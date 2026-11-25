//! Record command implementation - captures the WebSocket stream to a video file.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cli::RecordArgs;
use crate::output::Output;
use crate::recorder::{self, Codec, EncoderChoice, RecordOptions};

pub async fn run(
    session: &str,
    args: RecordArgs,
    output: &Output,
    timeout_ms: u64,
    stream_port: u16,
) -> anyhow::Result<()> {
    let Some(codec) = Codec::parse(&args.codec) else {
        output.print_error("invalid_codec", "Codec must be 'h264' or 'hevc'");
        std::process::exit(1);
    };

    let duration = match args.duration {
        Some(d) if !(d > 0.0 && d.is_finite()) => {
            output.print_error(
                "invalid_duration",
                "Duration must be a positive number of seconds",
            );
            std::process::exit(1);
        }
        Some(d) => Some(Duration::from_secs_f64(d)),
        None => None,
    };

    let port = args
        .port
        .unwrap_or(if stream_port != 0 { stream_port } else { 9224 });
    let url = args
        .url
        .unwrap_or_else(|| format!("ws://{}:{}", args.host, port));

    let path = PathBuf::from(
        args.output
            .unwrap_or_else(|| format!("agent-rdp-{}-{}.mp4", session, utc_timestamp())),
    );

    let path_display = path.display().to_string();
    let opts = RecordOptions {
        url,
        output: path,
        fps: args.fps,
        quality: args.quality,
        codec,
        encoder: EncoderChoice::parse(&args.encoder),
        ffmpeg: args.ffmpeg,
        vaapi_device: args.vaapi_device,
        duration,
        first_frame_timeout: Duration::from_millis(timeout_ms),
    };

    let json = output.is_json();
    let result = recorder::record(opts, |s| {
        if !json {
            eprintln!(
                "Recording {}x{} to {} with {} ({}, {})",
                s.width,
                s.height,
                path_display,
                s.encoder,
                if s.hardware { "hardware" } else { "software" },
                s.container.name(),
            );
            eprintln!("ffmpeg: {}. Press Ctrl+C to stop.", s.ffmpeg.display());
        }
    })
    .await;

    let summary = match result {
        Ok(s) => s,
        Err(e) => {
            output.print_error("record_failed", &e.to_string());
            std::process::exit(1);
        }
    };

    if json {
        let value = serde_json::json!({
            "success": summary.finalized,
            "data": {
                "type": "recording",
                "path": summary.path.display().to_string(),
                "frames": summary.frames,
                "durationSeconds": summary.duration.as_secs_f64(),
                "width": summary.started.width,
                "height": summary.started.height,
                "encoder": summary.started.encoder,
                "ffmpeg": summary.started.ffmpeg.display().to_string(),
                "hardware": summary.started.hardware,
                "container": summary.started.container.name(),
                "stopReason": summary.stop_reason.as_str(),
                "finalized": summary.finalized,
            }
        });
        println!("{}", value);
    } else {
        println!(
            "Saved {} ({} frames, {:.1}s, stopped: {}){}",
            summary.path.display(),
            summary.frames,
            summary.duration.as_secs_f64(),
            summary.stop_reason.as_str().replace('_', " "),
            if summary.finalized {
                ""
            } else {
                " - ffmpeg did not exit cleanly; the file is playable up to the last fragment"
            }
        );
    }

    if !summary.finalized {
        std::process::exit(1);
    }
    Ok(())
}

/// Current UTC time as `YYYYMMDDTHHMMSSZ`.
fn utc_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Days since 1970-01-01 to (year, month, day). Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::civil_from_days;

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(20_782), (2026, 11, 25));
    }
}
