use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body::{Body, SizeHint};
use http_body_util::BodyExt;
use hyper::Uri;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioIo;
use serde::de::DeserializeOwned;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

/// A minimal HTTP/1.1 client that dials a fresh Unix-domain connection per
/// request, like the Docker CLI does. No pooling, no TLS, no proxy — the
/// socket is local and the cost is negligible.
pub struct UnixClient {
    socket: String,
}

impl UnixClient {
    pub fn new(socket: impl Into<String>) -> Self {
        Self {
            socket: socket.into(),
        }
    }

    pub fn socket(&self) -> &str {
        &self.socket
    }

    fn client(&self) -> Client<UnixConnector, EmptyBody> {
        Client::builder(TokioExecutor).build(UnixConnector {
            socket: self.socket.clone(),
        })
    }

    /// GET a path and deserialize the JSON body.
    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, DockerError> {
        let bytes = self.get_bytes(path).await?;
        serde_json::from_slice(&bytes).map_err(|e| DockerError::Decode(e.to_string()))
    }

    /// GET a path, returning the raw body.
    pub async fn get_bytes(&self, path: &str) -> Result<Vec<u8>, DockerError> {
        self.get_bytes_opts(path, &[]).await
    }

    /// GET with extra headers (used for API-version overrides).
    pub async fn get_bytes_opts(
        &self,
        path: &str,
        headers: &[(&str, String)],
    ) -> Result<Vec<u8>, DockerError> {
        let mut b = hyper::Request::builder()
            .method("GET")
            .uri(uri(path))
            .header("Host", "localhost");
        for (k, v) in headers {
            b = b.header(*k, v.clone());
        }
        let req = b
            .body(EmptyBody)
            .map_err(|e| DockerError::Request(e.to_string()))?;
        let res = self.client().request(req).await.map_err(map_hyper_err)?;
        let status = res.status();
        let body = res
            .into_body()
            .collect()
            .await
            .map_err(map_hyper_err)?
            .to_bytes();
        check_status(status, &body)?;
        Ok(body.to_vec())
    }

    /// GET a path with an explicit `Api-Version` header, returning the raw
    /// body plus the `API-Version` response header (empty if absent). Some
    /// endpoints (e.g. `/system/df`) are only routed when the daemon is told
    /// which API version the client speaks.
    pub async fn get_with_version(
        &self,
        path: &str,
        version: &str,
    ) -> Result<(Vec<u8>, String), DockerError> {
        let (status, headers, body) = self.request("GET", path, version).await?;
        check_status(status, &body)?;
        let header_version = headers
            .get("API-Version")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        Ok((body, header_version))
    }

    /// Raw HTTP request through hyper; returns status, response headers and
    /// the fully-buffered body.
    async fn request(
        &self,
        method: &str,
        path: &str,
        version: &str,
    ) -> Result<(hyper::StatusCode, hyper::HeaderMap, Vec<u8>), DockerError> {
        let mut b = hyper::Request::builder()
            .method(method)
            .uri(uri(path))
            .header("Host", "localhost");
        if !version.is_empty() {
            b = b.header("Api-Version", version.to_string());
        }
        let req = b
            .body(EmptyBody)
            .map_err(|e| DockerError::Request(e.to_string()))?;
        let res = self.client().request(req).await.map_err(map_hyper_err)?;
        let status = res.status();
        let headers = res.headers().clone();
        let body = res
            .into_body()
            .collect()
            .await
            .map_err(map_hyper_err)?
            .to_bytes();
        Ok((status, headers, body.to_vec()))
    }

    /// POST/DELETE etc. with an empty body; returns the raw response body.
    pub async fn call_empty(&self, method: &str, path: &str) -> Result<Vec<u8>, DockerError> {
        let req = hyper::Request::builder()
            .method(method)
            .uri(uri(path))
            .header("Host", "localhost")
            .body(EmptyBody)
            .map_err(|e| DockerError::Request(e.to_string()))?;
        let res = self.client().request(req).await.map_err(map_hyper_err)?;
        let status = res.status();
        let body = res
            .into_body()
            .collect()
            .await
            .map_err(map_hyper_err)?
            .to_bytes();
        check_status(status, &body)?;
        Ok(body.to_vec())
    }

    /// Open a raw streaming request (e.g. `/containers/{id}/logs?follow=1`).
    ///
    /// Writes the request bytes directly, then returns the socket so the
    /// caller can read the response head and body framing itself. The daemon
    /// answers stream endpoints with `Transfer-Encoding: chunked` followed by
    /// the Docker multiplexed stream (when TTY is off). `Accept: application/json`
    /// selects newline-delimited JSON for stats/events endpoints.
    pub async fn open_stream(&self, path: &str, json: bool) -> Result<UnixStream, DockerError> {
        self.open_stream_opts(path, json, "").await
    }

    /// As `open_stream`, but with an explicit `Api-Version` header.
    pub async fn open_stream_opts(
        &self,
        path: &str,
        json: bool,
        version: &str,
    ) -> Result<UnixStream, DockerError> {
        let mut stream = UnixStream::connect(self.socket()).await.map_err(|e| {
            DockerError::Connect(format!(
                "cannot reach Docker daemon at {socket}: {e}",
                socket = self.socket
            ))
        })?;
        let accept = if json {
            "application/json"
        } else {
            "application/vnd.docker.raw-stream"
        };
        let vhead = if version.is_empty() {
            String::new()
        } else {
            format!("Api-Version: {version}\r\n")
        };
        let req =
            format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nAccept: {accept}\r\n{vhead}\r\n");
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| DockerError::Io(e.to_string()))?;
        Ok(stream)
    }
}

fn uri(path: &str) -> Uri {
    format!("http://localhost{path}")
        .parse()
        .unwrap_or_else(|_| Uri::from_static("http://localhost/"))
}

fn check_status(status: hyper::StatusCode, body: &[u8]) -> Result<(), DockerError> {
    if status.is_success() {
        return Ok(());
    }
    let text = String::from_utf8_lossy(body).to_string();
    // The daemon reports errors as {"message": "..."}; keep it short.
    let message = serde_json::from_str::<BTreeMap<String, String>>(&text)
        .ok()
        .and_then(|m| m.get("message").cloned())
        .unwrap_or(text);
    Err(DockerError::Api(status.as_u16(), message))
}

fn map_hyper_err(e: impl std::fmt::Display) -> DockerError {
    DockerError::Transport(e.to_string())
}

#[derive(Debug, thiserror::Error)]
pub enum DockerError {
    #[error("{0}")]
    Connect(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("request error: {0}")]
    Request(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("daemon returned {0}: {1}")]
    Api(u16, String),
    #[error("bad JSON from daemon: {0}")]
    Decode(String),
}

#[derive(Clone, Copy)]
struct TokioExecutor;

impl hyper::rt::Executor<UnpinBoxFuture> for TokioExecutor {
    fn execute(&self, fut: UnpinBoxFuture) {
        tokio::spawn(fut);
    }
}

type UnpinBoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

// ---------------------------------------------------------------------------
// Hyper glue: connect through a Unix socket, ignore the URI authority.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct UnixConnector {
    socket: String,
}

impl tower_service::Service<Uri> for UnixConnector {
    type Response = TokioIo<UnixStream>;
    type Error = std::io::Error;
    type Future = Pin<Box<dyn Future<Output = Result<TokioIo<UnixStream>, std::io::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: Uri) -> Self::Future {
        let socket = self.socket.clone();
        Box::pin(async move { UnixStream::connect(socket).await.map(TokioIo::new) })
    }
}

/// An empty request body.
struct EmptyBody;

impl Body for EmptyBody {
    type Data = Bytes;
    type Error = std::convert::Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        Poll::Ready(None)
    }

    fn is_end_stream(&self) -> bool {
        true
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::with_exact(0)
    }
}
