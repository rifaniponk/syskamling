//! Thin facade over the Docker Engine API used by the UI layer.

mod client;
mod types;

pub use client::{DockerError, UnixClient};
pub use types::{ContainerSummary, ImageSummary, VolumeEntry};

use tokio::net::UnixStream;

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
        self.client.get_json("/containers/json?all=true").await
    }

    pub async fn list_images(&self) -> Result<Vec<ImageSummary>, DockerError> {
        self.client.get_json("/images/json").await
    }

    pub async fn list_volumes(&self) -> Result<Vec<VolumeEntry>, DockerError> {
        let v: serde_json::Value = self.client.get_json("/volumes").await?;
        Ok(serde_json::from_value(v["Volumes"].clone())
            .map_err(|e| DockerError::Decode(e.to_string()))?)
    }

    /// Report the daemon's API version; empty string if it doesn't say.
    pub async fn ping(&self) -> Result<String, DockerError> {
        let body = self.client.get_bytes("/_ping").await?;
        Ok(String::from_utf8_lossy(&body).trim().to_string())
    }

    pub async fn start(&self, id: &str) -> Result<(), DockerError> {
        self.client
            .call_empty("POST", &format!("/containers/{id}/start"))
            .await
            .map(|_| ())
    }

    pub async fn stop(&self, id: &str) -> Result<(), DockerError> {
        // 304 = "not running": already stopped, which is the desired end state.
        match self
            .client
            .call_empty("POST", &format!("/containers/{id}/stop"))
            .await
        {
            Ok(_) => Ok(()),
            Err(DockerError::Api(304, _)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    pub async fn restart(&self, id: &str) -> Result<(), DockerError> {
        self.client
            .call_empty("POST", &format!("/containers/{id}/restart"))
            .await
            .map(|_| ())
    }

    pub async fn remove(&self, id: &str) -> Result<(), DockerError> {
        self.client
            .call_empty("DELETE", &format!("/containers/{id}?force=true&v=true"))
            .await
            .map(|_| ())
    }

    /// Open the raw multiplexed log stream for a container (follow mode).
    pub async fn open_log_stream(
        &self,
        id: &str,
        follow: bool,
        tail: usize,
    ) -> Result<UnixStream, DockerError> {
        self.client
            .open_stream(&format!(
                "/containers/{id}/logs?stdout=true&stderr=true&follow={follow}&tail={tail}"
            ))
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

/// Incremental decoder for the Docker stream format:
/// `[STREAM(1) | 0(3) | SIZE(4 BE) | PAYLOAD]` per frame.
pub struct LogStreamDecoder {
    buf: Vec<u8>,
}

impl LogStreamDecoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
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
            let len =
                u32::from_be_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;
            if self.buf.len() < 8 + len {
                break;
            }
            let payload = self.buf[8..8 + len].to_vec();
            self.buf.drain(..8 + len);
            let text = String::from_utf8_lossy(&payload).trim_end_matches('\n').to_string();
            for l in text.split('\n') {
                lines.push(LogLine { stream: kind, text: l.to_string() });
            }
        }
        lines
    }
}

