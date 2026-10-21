use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use http_body::{Body, SizeHint};
use http_body_util::BodyExt;
use hyper::Uri;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioIo;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
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

    fn socket(&self) -> &str {
        &self.socket
    }

    fn client(&self) -> Client<UnixStream, UnixHyperIo> {
        Client::builder(TokioExecutor).build(UnixHyperConnector {
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
        let req = hyper::Request::builder()
            .method("GET")
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
    /// the Docker multiplexed stream (when TTY is off).
    pub async fn open_stream(&self, path: &str) -> Result<UnixStream, DockerError> {
        let mut stream = UnixStream::connect(self.socket()).await.map_err(|e| {
            DockerError::Connect(format!(
                "cannot reach Docker daemon at {socket}: {e}",
                socket = self.socket
            ))
        })?;
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: localhost\r\nAccept: application/vnd.docker.raw-stream\r\n\r\n"
        );
        stream.write_all(req.as_bytes()).await.map_err(|e| DockerError::Io(e.to_string()))?;
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

fn map_hyper_err(e: hyper_util::client::legacy::Error) -> DockerError {
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

// ---------------------------------------------------------------------------
// Hyper glue: connect through a Unix socket, ignore the URI authority.
// ---------------------------------------------------------------------------

struct UnixHyperConnector {
    socket: String,
}

impl tower_service::Service<Uri> for UnixHyperConnector {
    type Response = UnixStream;
    type Error = std::io::Error;
    type Future = Pin<Box<dyn Future<Output = Result<UnixStream, std::io::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: Uri) -> Self::Future {
        let socket = self.socket.clone();
        Box::pin(async move { UnixStream::connect(socket).await })
    }
}

#[derive(Clone, Copy)]
struct TokioExecutor;

impl hyper::rt::Executor<UnpinBoxFuture> for TokioExecutor {
    fn execute(&self, fut: UnpinBoxFuture) {
        tokio::spawn(fut);
    }
}

type UnpinBoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Adapt tokio's UnixStream to hyper's AsyncRead/AsyncWrite.
struct UnixHyperIo(UnixStream);

impl hyper::rt::Read for UnixHyperIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        mut buf: hyper::rt::ReadBufCursor<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        let n = unsafe {
            let mut tbuf = ReadBuf::uninit(buf.as_mut());
            match tokio::io::AsyncRead::poll_read(Pin::new(&mut self.get_mut().0), cx, &mut tbuf) {
                Poll::Ready(Ok(())) => tbuf.filled().len(),
                other => return other,
            }
        };
        unsafe {
            buf.advance_unchecked(n);
        }
        Poll::Ready(Ok(()))
    }
}

impl hyper::rt::Write for UnixHyperIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        tokio::io::AsyncWrite::poll_write(Pin::new(&mut self.get_mut().0), cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), std::io::Error>> {
        tokio::io::AsyncWrite::poll_flush(Pin::new(&mut self.get_mut().0), cx)
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        tokio::io::AsyncWrite::poll_shutdown(Pin::new(&mut self.get_mut().0), cx)
    }
}

impl AsyncRead for UnixHyperIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        tokio::io::AsyncRead::poll_read(Pin::new(&mut self.get_mut().0), cx, buf)
    }
}

impl AsyncWrite for UnixHyperIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        tokio::io::AsyncWrite::poll_write(Pin::new(&mut self.get_mut().0), cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        tokio::io::AsyncWrite::poll_flush(Pin::new(&mut self.get_mut().0), cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        tokio::io::AsyncWrite::poll_shutdown(Pin::new(&mut self.get_mut().0), cx)
    }
}

/// An empty request body.
struct EmptyBody;

impl Body for EmptyBody {
    type Data = bytes::Bytes;
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
