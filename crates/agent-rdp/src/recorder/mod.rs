//! Session video recorder.
//!
//! Connects to the daemon's WebSocket debug stream as an ordinary viewer,
//! receives the JPEG frames it broadcasts, and pipes them into ffmpeg at a
//! constant frame rate (re-sending the latest frame when the desktop has not
//! produced a new one) so the output has correct wall-clock timing.
//!
//! ffmpeg runs in its own process group, so a Ctrl+C in the terminal reaches
//! only the recorder, which then closes ffmpeg's stdin and lets it finalize the
//! file. If the recorder itself dies, ffmpeg sees EOF on stdin and finalizes
//! anyway; if ffmpeg dies, the fragmented output is still playable.

pub mod ffmpeg;

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, warn};

pub use ffmpeg::{Codec, Container, EncoderChoice};
use ffmpeg::{EncodeSettings, Encoder};

/// Recording options.
#[derive(Debug, Clone)]
pub struct RecordOptions {
    /// WebSocket URL of the daemon stream, e.g. `ws://localhost:9224`.
    pub url: String,
    pub output: PathBuf,
    pub fps: u32,
    /// Constant quality on the CRF / QP scale (0 = lossless-ish, 51 = worst).
    pub quality: u8,
    pub codec: Codec,
    pub encoder: EncoderChoice,
    /// Explicit ffmpeg binary.
    pub ffmpeg: Option<String>,
    pub vaapi_device: String,
    /// Stop automatically after this long.
    pub duration: Option<Duration>,
    /// How long to wait for the first frame from the stream.
    pub first_frame_timeout: Duration,
}

/// Information reported once encoding has started.
#[derive(Debug, Clone)]
pub struct RecordingStarted {
    pub ffmpeg: PathBuf,
    pub encoder: String,
    pub hardware: bool,
    pub container: Container,
    pub width: u32,
    pub height: u32,
}

/// Why the recording ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Ctrl+C / SIGTERM.
    Interrupted,
    /// `--duration` elapsed.
    DurationReached,
    /// The WebSocket stream closed (daemon exited or RDP disconnected).
    StreamClosed,
    /// ffmpeg exited on its own.
    EncoderExited,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interrupted => "interrupted",
            Self::DurationReached => "duration_reached",
            Self::StreamClosed => "stream_closed",
            Self::EncoderExited => "encoder_exited",
        }
    }
}

/// Result of a finished recording.
#[derive(Debug, Clone)]
pub struct RecordingSummary {
    pub started: RecordingStarted,
    pub path: PathBuf,
    pub frames: u64,
    pub duration: Duration,
    pub stop_reason: StopReason,
    /// Whether ffmpeg exited cleanly after finalizing the file.
    pub finalized: bool,
}

/// A decoded frame from the stream.
struct Frame {
    jpeg: Vec<u8>,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
struct WsMessage {
    #[serde(rename = "type")]
    kind: String,
    data: Option<String>,
    metadata: Option<WsFrameMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WsFrameMetadata {
    device_width: u32,
    device_height: u32,
}

/// Record the stream until interrupted, the duration elapses, or the stream ends.
pub async fn record(
    opts: RecordOptions,
    on_start: impl FnOnce(&RecordingStarted),
) -> anyhow::Result<RecordingSummary> {
    let fps = opts.fps.clamp(1, 120);
    let container = Container::from_path(&opts.output).ok_or_else(|| {
        anyhow::anyhow!(
            "unsupported output extension for '{}' (use .mp4, .mkv or .ts)",
            opts.output.display()
        )
    })?;

    let ffmpeg_path = ffmpeg::find_ffmpeg(opts.ffmpeg.as_deref()).await?;
    debug!("using ffmpeg at {}", ffmpeg_path.display());

    let stop_rx = spawn_signal_listener();

    // Connect to the stream and keep only the newest frame.
    let (ws, _) = tokio_tungstenite::connect_async(opts.url.as_str())
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "could not connect to {}: {} (was the session started with --stream-port?)",
                opts.url,
                e
            )
        })?;
    let (frame_tx, mut frame_rx) = watch::channel::<Option<Arc<Frame>>>(None);
    tokio::spawn(read_stream(ws, frame_tx));

    let first = tokio::time::timeout(opts.first_frame_timeout, async {
        loop {
            if let Some(f) = frame_rx.borrow_and_update().clone() {
                return Some(f);
            }
            if frame_rx.changed().await.is_err() {
                return None;
            }
        }
    })
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "no frame received within {:?} (is the RDP session connected?)",
            opts.first_frame_timeout
        )
    })?
    .ok_or_else(|| anyhow::anyhow!("stream closed before the first frame arrived"))?;

    // Output canvas: the first frame's size rounded up to even dimensions.
    let width = (first.width.max(2) + 1) & !1;
    let height = (first.height.max(2) + 1) & !1;

    let template = EncodeSettings {
        encoder: Encoder::new("libx264"),
        codec: opts.codec,
        quality: opts.quality,
        fps,
        width,
        height,
        vaapi_device: opts.vaapi_device.clone(),
    };
    let encoder = ffmpeg::select_encoder(&ffmpeg_path, &opts.encoder, &template).await?;
    let settings = EncodeSettings {
        encoder,
        ..template
    };

    let mut cmd = Command::new(&ffmpeg_path);
    cmd.args(settings.record_args(container, &opts.output))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    detach_from_console_signals(&mut cmd);
    let mut child = cmd.spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to open ffmpeg stdin"))?;

    let started = RecordingStarted {
        ffmpeg: ffmpeg_path,
        encoder: settings.encoder.name.clone(),
        hardware: settings.encoder.kind.is_hardware(),
        container,
        width,
        height,
    };
    on_start(&started);

    let start = Instant::now();
    let deadline = opts.duration.map(|d| tokio::time::Instant::now() + d);
    let mut ticker = tokio::time::interval(Duration::from_secs_f64(1.0 / f64::from(fps)));
    // Burst: if a write stalls, catch up with duplicate frames so the video
    // timeline keeps matching wall-clock time.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
    let mut latest = first;
    let mut frames: u64 = 0;
    let mut stop_rx = stop_rx;

    let stop_reason = loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Some(f) = frame_rx.borrow_and_update().clone() {
                    latest = f;
                }
                if let Err(e) = stdin.write_all(&latest.jpeg).await {
                    debug!("ffmpeg stdin closed: {}", e);
                    break StopReason::EncoderExited;
                }
                frames += 1;
            }
            res = frame_rx.changed() => {
                if res.is_err() {
                    break StopReason::StreamClosed;
                }
            }
            _ = stop_rx.changed() => break StopReason::Interrupted,
            _ = sleep_until(deadline) => break StopReason::DurationReached,
            _ = child.wait() => break StopReason::EncoderExited,
        }
    };
    let duration = start.elapsed();

    // Closing stdin tells ffmpeg to flush and finalize the file.
    let _ = stdin.flush().await;
    drop(stdin);
    let finalized = match tokio::time::timeout(Duration::from_secs(60), child.wait()).await {
        Ok(Ok(status)) => {
            if !status.success() {
                warn!("ffmpeg exited with {}", status);
            }
            status.success()
        }
        Ok(Err(e)) => {
            warn!("failed to wait for ffmpeg: {}", e);
            false
        }
        Err(_) => {
            warn!("ffmpeg did not finish in time; killing it (file is still playable)");
            let _ = child.kill().await;
            false
        }
    };

    Ok(RecordingSummary {
        started,
        path: opts.output,
        frames,
        duration,
        stop_reason,
        finalized,
    })
}

async fn sleep_until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending().await,
    }
}

/// Read stream messages, publishing each decoded frame. Dropping `tx` on
/// return signals that the stream ended.
async fn read_stream<S>(mut ws: S, tx: watch::Sender<Option<Arc<Frame>>>)
where
    S: futures_util::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    use base64::Engine;

    while let Some(msg) = ws.next().await {
        let text = match msg {
            Ok(Message::Text(t)) => t,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(e) => {
                debug!("stream error: {}", e);
                break;
            }
        };
        let Ok(parsed) = serde_json::from_str::<WsMessage>(text.as_str()) else {
            continue;
        };
        if parsed.kind != "frame" {
            continue;
        }
        let (Some(data), Some(meta)) = (parsed.data, parsed.metadata) else {
            continue;
        };
        match base64::engine::general_purpose::STANDARD.decode(data) {
            Ok(jpeg) => {
                let frame = Frame {
                    jpeg,
                    width: meta.device_width,
                    height: meta.device_height,
                };
                if tx.send(Some(Arc::new(frame))).is_err() {
                    break;
                }
            }
            Err(e) => debug!("bad frame data: {}", e),
        }
    }
}

/// Returns a receiver that changes when Ctrl+C (or SIGTERM on Unix) arrives.
fn spawn_signal_listener() -> watch::Receiver<bool> {
    let (tx, rx) = watch::channel(false);
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut term = signal(SignalKind::terminate()).ok();
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = async {
                    match term.as_mut() {
                        Some(t) => { t.recv().await; }
                        None => std::future::pending::<()>().await,
                    }
                } => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        let _ = tx.send(true);
    });
    rx
}

/// Keep terminal Ctrl+C from reaching ffmpeg directly; the recorder shuts it
/// down itself by closing stdin, which guarantees the file is finalized.
fn detach_from_console_signals(cmd: &mut Command) {
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
}
