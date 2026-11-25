//! ffmpeg discovery, encoder selection, and command-line construction.
//!
//! The recorder does not link against any codec library. It drives an external
//! `ffmpeg` binary (the [jellyfin-ffmpeg](https://github.com/jellyfin/jellyfin-ffmpeg)
//! build is preferred because it ships with every hardware encoder enabled) and
//! feeds it a stream of JPEG frames on stdin.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tracing::debug;

/// Output video codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

impl Codec {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "h264" | "avc" | "h.264" => Some(Self::H264),
            "hevc" | "h265" | "h.265" => Some(Self::Hevc),
            _ => None,
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
        }
    }

    fn software_encoder(self) -> &'static str {
        match self {
            Self::H264 => "libx264",
            Self::Hevc => "libx265",
        }
    }
}

/// Encoder families, each with its own constant-quality rate control flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderKind {
    /// libx264 / libx265 (CRF).
    Software,
    /// NVIDIA NVENC (VBR with a constant-quality target, `-cq`).
    Nvenc,
    /// Intel Quick Sync (ICQ, `-global_quality`).
    Qsv,
    /// VA-API (CQP).
    Vaapi,
    /// Apple VideoToolbox (constant quality `-q:v`, Apple Silicon only).
    VideoToolbox,
    /// AMD AMF (CQP).
    Amf,
    /// Rockchip MPP, available in jellyfin-ffmpeg (CQP).
    Rkmpp,
}

impl EncoderKind {
    pub fn is_hardware(self) -> bool {
        self != Self::Software
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::Software => "",
            Self::Nvenc => "nvenc",
            Self::Qsv => "qsv",
            Self::Vaapi => "vaapi",
            Self::VideoToolbox => "videotoolbox",
            Self::Amf => "amf",
            Self::Rkmpp => "rkmpp",
        }
    }

    /// Identify an encoder family from an ffmpeg encoder name.
    fn from_name(name: &str) -> Self {
        [
            Self::Nvenc,
            Self::Qsv,
            Self::Vaapi,
            Self::VideoToolbox,
            Self::Amf,
            Self::Rkmpp,
        ]
        .into_iter()
        .find(|k| name.ends_with(&format!("_{}", k.suffix())))
        .unwrap_or(Self::Software)
    }

    /// Hardware encoder families worth probing on the current platform, in
    /// order of preference.
    fn platform_candidates() -> &'static [EncoderKind] {
        if cfg!(target_os = "macos") {
            &[Self::VideoToolbox]
        } else if cfg!(target_os = "windows") {
            &[Self::Nvenc, Self::Qsv, Self::Amf]
        } else {
            &[Self::Nvenc, Self::Qsv, Self::Vaapi, Self::Rkmpp]
        }
    }
}

/// A concrete encoder choice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoder {
    /// ffmpeg encoder name, e.g. `h264_nvenc`.
    pub name: String,
    pub kind: EncoderKind,
}

impl Encoder {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let kind = EncoderKind::from_name(&name);
        Self { name, kind }
    }

    fn for_kind(codec: Codec, kind: EncoderKind) -> Self {
        let name = match kind {
            EncoderKind::Software => codec.software_encoder().to_string(),
            _ => format!("{}_{}", codec.prefix(), kind.suffix()),
        };
        Self { name, kind }
    }
}

/// Output container, chosen from the file extension.
///
/// All of them are written so that a file cut short by a crash, power loss or
/// `kill -9` is still playable up to roughly the last second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// Fragmented MP4: an empty `moov` up front followed by self-contained
    /// `moof`+`mdat` fragments. Plays in browsers, QuickTime, VLC, etc.
    FragmentedMp4,
    /// Matroska: a sequence of independent clusters.
    Matroska,
    /// MPEG transport stream: no header or index at all.
    MpegTs,
}

impl Container {
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "mp4" | "m4v" | "mov" => Some(Self::FragmentedMp4),
            "mkv" => Some(Self::Matroska),
            "ts" | "m2ts" => Some(Self::MpegTs),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::FragmentedMp4 => "fragmented-mp4",
            Self::Matroska => "matroska",
            Self::MpegTs => "mpegts",
        }
    }

    fn args(self, codec: Codec) -> Vec<String> {
        let mut args = strings(&[]);
        match self {
            Self::FragmentedMp4 => {
                args.extend(strings(&[
                    "-f",
                    "mp4",
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                    // Also cut a fragment every second (microseconds), even
                    // inside a GOP, to bound data loss on a crash.
                    "-frag_duration",
                    "1000000",
                ]));
                if codec == Codec::Hevc {
                    // Required by QuickTime/Safari for HEVC in MP4.
                    args.extend(strings(&["-tag:v", "hvc1"]));
                }
            }
            Self::Matroska => {
                args.extend(strings(&["-f", "matroska", "-cluster_time_limit", "1000"]));
            }
            Self::MpegTs => {
                args.extend(strings(&["-f", "mpegts"]));
            }
        }
        args
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// Everything needed to build the ffmpeg command line.
#[derive(Debug, Clone)]
pub struct EncodeSettings {
    pub encoder: Encoder,
    pub codec: Codec,
    /// Constant quality on the x264 CRF / QP scale (0 = best, 51 = worst).
    pub quality: u8,
    pub fps: u32,
    /// Output dimensions (must be even).
    pub width: u32,
    pub height: u32,
    /// DRM render node used by VA-API.
    pub vaapi_device: String,
}

impl EncodeSettings {
    /// Arguments that must appear before the first `-i` (hardware device setup).
    fn global_args(&self) -> Vec<String> {
        match self.encoder.kind {
            EncoderKind::Vaapi => vec![
                "-init_hw_device".into(),
                format!("vaapi=va:{}", self.vaapi_device),
                "-filter_hw_device".into(),
                "va".into(),
            ],
            _ => Vec::new(),
        }
    }

    /// Video filter: fit into a fixed canvas (handles odd sizes and mid-stream
    /// resolution changes) and convert to the encoder's pixel format.
    fn filter(&self) -> String {
        let (w, h) = (self.width, self.height);
        let pix_fmt = match self.encoder.kind {
            EncoderKind::Software => "yuv420p",
            _ => "nv12",
        };
        let mut vf = format!(
            "scale={w}:{h}:force_original_aspect_ratio=decrease:out_color_matrix=bt709:out_range=tv,\
             pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,format={pix_fmt}"
        );
        if self.encoder.kind == EncoderKind::Vaapi {
            vf.push_str(",hwupload");
        }
        vf
    }

    /// Encoder selection plus constant-quality rate control arguments.
    fn encoder_args(&self) -> Vec<String> {
        let q = self.quality.min(51);
        let qs = q.to_string();
        let qs = qs.as_str();
        let gop = (self.fps * 2).max(1).to_string();
        let mut args = vec![
            "-c:v".to_string(),
            self.encoder.name.clone(),
            "-g".into(),
            gop,
        ];
        args.extend(match self.encoder.kind {
            EncoderKind::Software => {
                let mut v = strings(&["-crf", qs, "-preset", "veryfast"]);
                // Keep the encoder pipeline short (frame threads and lookahead
                // otherwise hold seconds of video at low frame rates), so less
                // is lost if ffmpeg is killed. Costs little compression.
                let params = format!("rc-lookahead={}:sync-lookahead=0", self.fps.clamp(1, 10));
                match self.codec {
                    Codec::H264 => {
                        v.extend(["-x264-params".into(), format!("{params}:sliced-threads=1")])
                    }
                    Codec::Hevc => v.extend([
                        "-x265-params".into(),
                        format!("rc-lookahead={}:frame-threads=1", self.fps.clamp(1, 10)),
                    ]),
                }
                v
            }
            EncoderKind::Nvenc => strings(&["-preset", "p4", "-rc", "vbr", "-cq", qs, "-b:v", "0"]),
            EncoderKind::Qsv => strings(&["-global_quality", qs, "-look_ahead", "0"]),
            EncoderKind::Vaapi => strings(&["-rc_mode", "CQP", "-qp", qs]),
            EncoderKind::VideoToolbox => {
                // VideoToolbox uses 1..100 (higher = better); map from 0..51.
                let vt_q = (100i32 - i32::from(q) * 2).clamp(1, 100).to_string();
                strings(&["-q:v", &vt_q])
            }
            EncoderKind::Amf => {
                let mut v = strings(&["-rc", "cqp", "-qp_i", qs, "-qp_p", qs]);
                if self.codec == Codec::H264 {
                    v.extend(strings(&["-qp_b", qs]));
                }
                v
            }
            EncoderKind::Rkmpp => strings(&[
                "-rc_mode",
                "CQP",
                "-qp_init",
                qs,
                "-qp_min",
                qs,
                "-qp_max",
                qs,
                "-qp_min_i",
                qs,
                "-qp_max_i",
                qs,
            ]),
        });
        args.extend(strings(&[
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
        ]));
        args
    }

    /// Full ffmpeg arguments for recording JPEG frames from stdin to `output`.
    pub fn record_args(&self, container: Container, output: &Path) -> Vec<String> {
        let fps = self.fps.to_string();
        let mut args = strings(&["-hide_banner", "-nostdin", "-loglevel", "error", "-y"]);
        args.extend(self.global_args());
        // Minimal probing: by default ffmpeg buffers ~5 MB / 5 s of input before
        // opening the output, which would all be lost on a crash.
        args.extend(strings(&[
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-fpsprobesize",
            "0",
            "-f",
            "mjpeg",
            "-framerate",
            &fps,
            "-i",
            "pipe:0",
            "-an",
        ]));
        args.extend(["-vf".to_string(), self.filter()]);
        args.extend(self.encoder_args());
        args.extend(strings(&["-r", &fps]));
        // Push every packet to disk immediately so a crash loses as little as possible.
        args.extend(strings(&["-flush_packets", "1"]));
        args.extend(container.args(self.codec));
        args.push(output.to_string_lossy().into_owned());
        args
    }

    /// Arguments for a short throwaway encode that checks an encoder actually
    /// works here (driver present, device accessible, size supported, ...).
    fn probe_args(&self) -> Vec<String> {
        let mut args = strings(&["-hide_banner", "-nostdin", "-loglevel", "error"]);
        args.extend(self.global_args());
        args.extend([
            "-f".into(),
            "lavfi".into(),
            "-i".into(),
            format!(
                "testsrc2=size={}x{}:rate={}",
                self.width, self.height, self.fps
            ),
            "-frames:v".into(),
            "5".into(),
            "-vf".into(),
            self.filter(),
        ]);
        args.extend(self.encoder_args());
        args.extend(strings(&["-f", "null", "-"]));
        args
    }
}

/// Locate an ffmpeg binary.
///
/// Order: explicit path (`--ffmpeg` / `AGENT_RDP_FFMPEG`), an `ffmpeg` placed
/// next to the agent-rdp executable, a jellyfin-ffmpeg install, then `PATH`.
pub async fn find_ffmpeg(explicit: Option<&str>) -> anyhow::Result<PathBuf> {
    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        if runs(&path).await {
            return Ok(path);
        }
        anyhow::bail!("ffmpeg at '{}' could not be executed", p);
    }

    let exe_name = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        candidates.push(dir.join(exe_name));
        candidates.push(dir.join("ffmpeg").join(exe_name));
    }

    if cfg!(target_os = "linux") {
        candidates.push("/usr/lib/jellyfin-ffmpeg/ffmpeg".into());
    } else if cfg!(target_os = "macos") {
        candidates.push("/Applications/Jellyfin.app/Contents/MacOS/ffmpeg".into());
    } else if cfg!(windows) {
        for var in ["ProgramFiles", "ProgramFiles(x86)"] {
            if let Ok(pf) = std::env::var(var) {
                candidates.push(
                    PathBuf::from(pf)
                        .join("Jellyfin")
                        .join("Server")
                        .join(exe_name),
                );
            }
        }
    }

    for c in &candidates {
        if c.is_file() && runs(c).await {
            return Ok(c.clone());
        }
    }

    let on_path = PathBuf::from(exe_name);
    if runs(&on_path).await {
        return Ok(on_path);
    }

    anyhow::bail!(
        "ffmpeg not found. Install jellyfin-ffmpeg (https://github.com/jellyfin/jellyfin-ffmpeg/releases) \
         or another ffmpeg build, then put it on PATH, pass --ffmpeg, or set AGENT_RDP_FFMPEG"
    )
}

async fn runs(path: &Path) -> bool {
    let mut cmd = Command::new(path);
    cmd.arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    matches!(
        tokio::time::timeout(Duration::from_secs(10), cmd.status()).await,
        Ok(Ok(s)) if s.success()
    )
}

/// List the video encoder names this ffmpeg build knows about.
async fn list_encoders(ffmpeg: &Path) -> anyhow::Result<Vec<String>> {
    let out = Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .await?;
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text
        .lines()
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let flags = parts.next()?;
            // Video encoder lines look like " V....D libx264  description".
            if flags.len() == 6 && flags.starts_with('V') {
                parts.next().map(String::from)
            } else {
                None
            }
        })
        .collect())
}

/// Run a short test encode; returns ffmpeg's error output on failure.
async fn probe(ffmpeg: &Path, settings: &EncodeSettings) -> Result<(), String> {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(settings.probe_args())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    match tokio::time::timeout(Duration::from_secs(15), cmd.output()).await {
        Ok(Ok(out)) if out.status.success() => Ok(()),
        Ok(Ok(out)) => Err(String::from_utf8_lossy(&out.stderr).trim().to_string()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("probe timed out".into()),
    }
}

/// How the encoder should be picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderChoice {
    /// Try hardware encoders, fall back to software.
    Auto,
    /// Always use libx264 / libx265.
    Software,
    /// A specific ffmpeg encoder name, e.g. `h264_vaapi`.
    Named(String),
}

impl EncoderChoice {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "software" | "sw" | "cpu" => Self::Software,
            _ => Self::Named(s.to_string()),
        }
    }
}

/// Pick a working encoder. `template` supplies everything except the encoder.
pub async fn select_encoder(
    ffmpeg: &Path,
    choice: &EncoderChoice,
    template: &EncodeSettings,
) -> anyhow::Result<Encoder> {
    let with = |encoder: Encoder| EncodeSettings {
        encoder,
        ..template.clone()
    };

    let candidates: Vec<Encoder> = match choice {
        EncoderChoice::Named(name) => {
            let enc = Encoder::new(name.clone());
            probe(ffmpeg, &with(enc.clone()))
                .await
                .map_err(|e| anyhow::anyhow!("encoder '{}' is not usable: {}", name, e))?;
            return Ok(enc);
        }
        EncoderChoice::Software => Vec::new(),
        EncoderChoice::Auto => {
            let available = list_encoders(ffmpeg).await.unwrap_or_default();
            EncoderKind::platform_candidates()
                .iter()
                .filter(|&&k| k != EncoderKind::Vaapi || Path::new(&template.vaapi_device).exists())
                .map(|&k| Encoder::for_kind(template.codec, k))
                .filter(|e| available.iter().any(|n| n == &e.name))
                .collect()
        }
    };

    for enc in candidates {
        match probe(ffmpeg, &with(enc.clone())).await {
            Ok(()) => return Ok(enc),
            Err(e) => debug!("hardware encoder {} unavailable: {}", enc.name, e),
        }
    }

    let enc = Encoder::for_kind(template.codec, EncoderKind::Software);
    probe(ffmpeg, &with(enc.clone()))
        .await
        .map_err(|e| anyhow::anyhow!("software encoder '{}' is not usable: {}", enc.name, e))?;
    Ok(enc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(name: &str) -> EncodeSettings {
        EncodeSettings {
            encoder: Encoder::new(name),
            codec: Codec::H264,
            quality: 23,
            fps: 10,
            width: 1920,
            height: 1080,
            vaapi_device: "/dev/dri/renderD128".into(),
        }
    }

    fn has_pair(args: &[String], a: &str, b: &str) -> bool {
        args.windows(2).any(|w| w[0] == a && w[1] == b)
    }

    #[test]
    fn encoder_kind_from_name() {
        assert_eq!(Encoder::new("h264_nvenc").kind, EncoderKind::Nvenc);
        assert_eq!(Encoder::new("hevc_vaapi").kind, EncoderKind::Vaapi);
        assert_eq!(Encoder::new("h264_rkmpp").kind, EncoderKind::Rkmpp);
        assert_eq!(
            Encoder::new("h264_videotoolbox").kind,
            EncoderKind::VideoToolbox
        );
        assert_eq!(Encoder::new("libx264").kind, EncoderKind::Software);
    }

    #[test]
    fn container_from_extension() {
        assert_eq!(
            Container::from_path(Path::new("a.mp4")),
            Some(Container::FragmentedMp4)
        );
        assert_eq!(
            Container::from_path(Path::new("a.MKV")),
            Some(Container::Matroska)
        );
        assert_eq!(
            Container::from_path(Path::new("a.ts")),
            Some(Container::MpegTs)
        );
        assert_eq!(Container::from_path(Path::new("a.avi")), None);
        assert_eq!(Container::from_path(Path::new("noext")), None);
    }

    #[test]
    fn software_uses_crf_and_fragmented_mp4() {
        let args = settings("libx264").record_args(Container::FragmentedMp4, Path::new("out.mp4"));
        assert!(has_pair(&args, "-c:v", "libx264"));
        assert!(has_pair(&args, "-crf", "23"));
        assert!(has_pair(
            &args,
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof"
        ));
        assert!(has_pair(&args, "-i", "pipe:0"));
        assert_eq!(args.last().unwrap(), "out.mp4");
    }

    #[test]
    fn hardware_rate_control() {
        let nv = settings("h264_nvenc").encoder_args();
        assert!(has_pair(&nv, "-cq", "23") && has_pair(&nv, "-rc", "vbr"));

        let va = settings("h264_vaapi");
        assert!(has_pair(&va.encoder_args(), "-rc_mode", "CQP"));
        assert!(va.filter().ends_with(",hwupload"));
        let va_args = va.record_args(Container::Matroska, Path::new("x.mkv"));
        assert!(has_pair(
            &va_args,
            "-init_hw_device",
            "vaapi=va:/dev/dri/renderD128"
        ));

        let qsv = settings("h264_qsv").encoder_args();
        assert!(has_pair(&qsv, "-global_quality", "23"));

        let vt = settings("h264_videotoolbox").encoder_args();
        assert!(has_pair(&vt, "-q:v", "54"));

        let rk = settings("h264_rkmpp").encoder_args();
        assert!(has_pair(&rk, "-rc_mode", "CQP") && has_pair(&rk, "-qp_init", "23"));
    }

    #[test]
    fn hevc_mp4_gets_hvc1_tag() {
        let mut s = settings("libx265");
        s.codec = Codec::Hevc;
        let args = s.record_args(Container::FragmentedMp4, Path::new("o.mp4"));
        assert!(has_pair(&args, "-tag:v", "hvc1"));
    }
}
