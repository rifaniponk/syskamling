//! Thin facade over the Docker Engine API used by the UI layer.

mod client;
mod types;

pub use client::{DockerError, UnixClient};
pub use types::{ContainerSummary, DfContainer, ImageSummary, Port, SystemDf, VolumeEntry};

use tokio::io::{AsyncRead, AsyncReadExt};

pub struct Docker {
    client: UnixClient,
}

impl Docker {
    pub fn new(socket: impl Into<String>) -> Self {
        Self {
            client: UnixClient::new(socket),
        }
    }

    /// List all containers, newest first (Engine default).
    pub async fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        self.client
            .get_json("/containers/json?all=true", 200)
            .await
    }

    pub async fn list_images(&self) -> Result<Vec<ImageSummary>, DockerError> {
        self.client.get_json("/images/json", 200).await
    }

    pub async fn list_volumes(&self) -> Result<Vec<VolumeEntry>, DockerError> {
        let v: serde_json::Value = self.client.get_json("/volumes", 200).await?;
        Ok(serde_json::from_value(v["Volumes"].clone())
            .map_err(|e| DockerError::Decode(e.to_string()))?)
    }

    pub async fn system_df(&self) -> Result<SystemDf, DockerError> {
        self.client.get_json("/system/df", 200).await
    }

    pub async fn ping(&self) -> Result<(), DockerError> {
        self.client.get_json("/_ping", 200).await
    }

    pub async fn start(&self, id: &str) -> Result<(), DockerError> {
        self.client.call_empty("POST", &format!("/containers/{id}/start"), 204).await
    }

    pub async fn stop(&self, id: &str) -> Result<(), DockerError> {
        self.client.call_empty("POST", &format!("/containers/{id}/stop"), 204).await
    }

    pub async fn restart(&self, id: &str) -> Result<(), DockerError> {
        self.client.call_empty("POST", &format!("/containers/{id}/restart"), 204).await
    }

    pub async fn remove(&self, id: &str) -> Result<(), DockerError> {
        self.client
            .call_empty("DELETE", &format!("/containers/{id}?force=true&v=true"), 204)
            .await
    }

    /// Open the raw multiplexed log stream for a container.
    pub async fn open_log_stream(
        &self,
        id: &str,
        follow: bool,
        tail: usize,
    ) -> Result<impl AsyncRead, DockerError> {
        self.client
            .open_stream(
                &format!("/containers/{id}/logs?stdout=true&stderr=true&follow={follow}&tail={tail}"),
            )
            .await
    }
}

/// A line of container output, already de-multiplexed from the Docker stream framing.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLine {
    pub stream: StreamKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamKind {
    Stdout,
    Stderr,
}

impl StreamKind {
    pub fn label(&self) -> &'static str {
        match self {
            StreamKind::Stdout => "stdout",
            StreamKind::Stderr => "stderr",
        }
    }
}

/// Incremental decoder for the Docker stream format:
/// `[STREAM(1) | 0(3) | SIZE(4 BE) | PAYLOAD]` per frame.
///
/// `tail` limits how many lines the decoder will emit (matching the API's
/// `tail` parameter on the server, but applied locally for safety).
pub struct LogStreamDecoder {
    buf: Vec<u8>,
    pub tail: usize,
    pub emitted: usize,
}

impl LogStreamDecoder {
    pub fn new(tail: usize) -> Self {
        Self { buf: Vec::new(), tail, emitted: 0 }
    }

    /// Feed raw socket bytes; returns complete lines.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<LogLine> {
        self.buf.extend_from_slice(chunk);
        let mut lines = Vec::new();
        loop {
            if self.buf.len() < 8 {
                break;
            }
            let kind = match self.buf[0] {
                1 => StreamKind::Stdout,
                2 => StreamKind::Stderr,
                3 => StreamKind::Stderr,
                _ => StreamKind::Stdout,
            };
            let len = u32::from_be_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;
            if self.buf.len() < 8 + len {
                break;
            }
            let payload = self.buf[8..8 + len].to_vec();
            self.buf.drain(..8 + len);
            let text = String::from_utf8_lossy(&payload).trim_end_matches('\n').to_string();
            for l in text.split('\n') {
                if self.tail > 0 && self.emitted >= self.tail {
                    // Trim from the front so the *last* tail lines win.
                    while lines.len() + self.emitted >= self.tail + lines.len() {
                        break;
                    }
                }
                lines.push(LogLine { stream: kind, text: l.to_string() });
                self.emitted += 1;
            }
        }
        lines
    }
}

/// Read a whole stream to EOF (used for non-followed log fetches).
pub async fn read_all_logs<R: AsyncRead + Unpin>(
    mut reader: R,
    tail: usize,
) -> Result<Vec<LogLine>, DockerError> {
    let mut dec = LogStreamDecoder::new(tail);
    let mut chunk = [0u8; 16 * 1024];
    let mut all: Vec<LogLine> = Vec::new();
    loop {
        let n = reader.read(&mut chunk).await.map_err(DockerError::Io)?;
        if n == 0 {
            break;
        }
        all.extend(dec.push(&chunk[..n]));
    }
    if tail > 0 && all.len() > tail {
        all.drain(..all.len() - tail);
    }
    Ok(all)
}
