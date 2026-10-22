//! Strata monitor page: pulls a Strata inference server's `/metrics` and
//! `/health` over HTTP and turns them into the same tiles its web monitor
//! shows — no 3D PC animation, just the statistics.
//!
//! The endpoint and API key live in `~/.config/syskamling/config.json`;
//! the GUI's Strata tab edits them in place.

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
use serde::Deserialize;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Persisted settings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StrataConfig {
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default)]
    pub api_key: String,
}

fn default_endpoint() -> String {
    "http://127.0.0.1:9191".to_string()
}

impl Default for StrataConfig {
    fn default() -> Self {
        Self {
            endpoint: default_endpoint(),
            api_key: String::new(),
        }
    }
}

impl StrataConfig {
    /// Normalized base URL without a trailing slash; the paths we fetch are
    /// relative (`metrics`, `health`).
    pub fn base(&self) -> String {
        self.endpoint.trim().trim_end_matches('/').to_string()
    }
}

/// Where the config lives; `None` when $HOME is unavailable.
pub fn config_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|h| h.join(".config").join("syskamling").join("config.json"))
}

#[derive(serde::Deserialize)]
struct FileShape {
    #[serde(default)]
    strata: Option<StrataConfig>,
}

/// Load `strata` settings; defaults when the file is missing or malformed.
pub fn load() -> StrataConfig {
    let Some(path) = config_path() else {
        return StrataConfig::default();
    };
    let Ok(txt) = std::fs::read_to_string(&path) else {
        return StrataConfig::default();
    };
    match serde_json::from_str::<FileShape>(&txt) {
        Ok(f) => f.strata.unwrap_or_default(),
        Err(_) => StrataConfig::default(),
    }
}

/// Merge our section into the shared config file, keeping other keys.
pub fn save(cfg: &StrataConfig) -> Result<(), String> {
    let path = config_path().ok_or("HOME is not set")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut root: BTreeMap<String, Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let me = serde_json::to_value(cfg).map_err(|e| e.to_string())?;
    root.insert("strata".into(), me);
    let pretty = serde_json::to_string_pretty(&root).map_err(|e| e.to_string())?;
    std::fs::write(&path, pretty).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// The /metrics document, trimmed to what the page shows
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    // engine
    pub model: String,
    pub max_context: i64,
    pub images: bool,
    pub batch: i64,
    pub batch_groups: i64,
    pub expert_slots: Option<i64>,
    pub expert_cache_mib: Option<i64>,
    pub conversation_cache_mib: Option<i64>,
    pub conversation_cache_slots: Option<i64>,
    // live
    pub state: String,
    pub queued: i64,
    pub phase: Option<String>,
    pub prompt_tokens: Option<i64>,
    pub prompt_read: Option<i64>,
    pub prompt_total: Option<i64>,
    pub generated: Option<i64>,
    pub max_tokens: Option<i64>,
    pub tok_s: Option<f64>,
    pub prefill_tok_s_mean: Option<f64>,
    pub parallel: i64,
    pub running: i64,
    pub waiting: i64,
    // hardware (now)
    pub gpu_util: Option<f64>,
    pub gpu_mem_used: Option<f64>,
    pub gpu_mem_total: Option<f64>,
    pub gpu_temp: Option<f64>,
    pub gpu_power: Option<f64>,
    pub gpu_power_limit: Option<f64>,
    pub gpu_pcie_gen: Option<i64>,
    pub gpu_pcie_gen_max: Option<i64>,
    pub gpu_pcie_width: Option<i64>,
    pub gpu_pcie_rx_mb: Option<f64>,
    pub cpu: Option<f64>,
    pub ram_used: Option<f64>,
    pub ram_total: Option<f64>,
    pub disk_read_mb: Option<f64>,
    pub disk_write_mb: Option<f64>,
    // hardware static
    pub gpu_name: Option<String>,
    pub cpu_name: Option<String>,
    pub cores: Option<i64>,
    pub threads: Option<i64>,
    pub gpu_note: Option<String>,
    // history series (1 sample/s, ~a minute kept by the server)
    pub hist: BTreeMap<String, Vec<f64>>,
    // latency summary
    pub ttft: Latency,
    pub itl: Latency,
    pub e2e: Latency,
    // totals since server start
    pub tot_requests: i64,
    pub tot_prompt_tokens: i64,
    pub tot_reused: i64,
    pub tot_output_tokens: i64,
    pub tot_prompt_ms: f64,
    pub tot_decode_ms: f64,
    pub totals_since: Option<f64>,
    // conversation cache card
    pub conv: Option<ConvCache>,
    // the last requests, newest first
    pub requests: Vec<RequestRow>,
    pub requests_kept: i64,
}

#[derive(Clone, Debug, Default)]
pub struct Latency {
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub mean: Option<f64>,
    pub count: i64,
}

#[derive(Clone, Debug, Default)]
pub struct ConvCache {
    pub enabled: bool,
    pub budget_mib: Option<i64>,
    pub slots: Option<i64>,
    pub parked: Option<i64>,
    pub bytes: Option<i64>,
    pub parks: Option<i64>,
    pub restores: Option<i64>,
    pub evictions: Option<i64>,
    pub requests: i64,
    pub requests_reused: i64,
    pub reused_tokens: i64,
    pub prompt_tokens: i64,
    pub last_reused: Option<i64>,
    pub last_prompt: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct RequestRow {
    pub time: f64,
    pub finish: String,
    pub prompt_tokens: Option<i64>,
    pub reused: Option<i64>,
    pub output_tokens: Option<i64>,
    pub prompt_ms: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub hit_rate: Option<f64>,
    pub pcie_share: Option<f64>,
    pub duration_s: Option<f64>,
}

/// Prefill rate of a finished request: the tokens it actually read (the
/// prompt minus what the cache gave back) over its prompt time — the same
/// arithmetic the web monitor uses.
pub fn prefill_rate(r: &RequestRow) -> Option<f64> {
    let fresh = r.prompt_tokens? - r.reused.unwrap_or(0);
    let ms = r.prompt_ms?;
    if fresh > 0 && ms > 0.0 {
        Some(fresh as f64 / (ms / 1000.0))
    } else {
        None
    }
}

/// The context used by the running request, else the last one — the ring
/// gauge's number.
pub fn context_used(m: &Metrics, last: Option<&RequestRow>) -> i64 {
    if m.state == "reading" || m.state == "generating" {
        m.prompt_tokens.unwrap_or(0) + m.generated.unwrap_or(0)
    } else if let Some(r) = last {
        r.prompt_tokens.unwrap_or(0) + r.output_tokens.unwrap_or(0)
    } else {
        0
    }
}

fn num(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

fn ndef(v: Option<&Value>, d: i64) -> i64 {
    num(v).map(|x| x as i64).unwrap_or(d)
}

fn sstr(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// The web monitor's prefill figure: the live rate while working, else what
/// the last request achieved.
pub fn prefill_now(m: &Metrics, last: Option<&RequestRow>) -> Option<f64> {
    if m.state == "reading" || m.state == "generating" {
        m.prefill_tok_s_mean
    } else {
        last.and_then(prefill_rate)
    }
}

/// The web monitor's decode figure: the live rate while generating, else the
/// last request's.
pub fn decode_now(m: &Metrics, last: Option<&RequestRow>) -> Option<f64> {
    if m.state == "generating" {
        m.tok_s
    } else {
        last.and_then(|r| r.decode_tok_s)
    }
}

/// Parse the `/metrics` JSON body. Unknown or missing fields stay defaults,
/// so older and newer servers both render.
pub fn parse_metrics(body: &[u8]) -> Result<Metrics, String> {
    let root: Value = serde_json::from_slice(body).map_err(|e| format!("bad JSON: {e}"))?;
    let g = |k: &str| root.get(k);
    let engine = g("engine").cloned().unwrap_or(Value::Null);
    let live = g("live").cloned().unwrap_or(Value::Null);
    let hw = g("hardware").cloned().unwrap_or(Value::Null);
    let st = g("hardware_static").cloned().unwrap_or(Value::Null);
    let hist = g("history").cloned().unwrap_or(Value::Null);
    let lat = g("latency").cloned().unwrap_or(Value::Null);
    let totals = g("totals").cloned().unwrap_or(Value::Null);
    let conv = g("conversation_cache").cloned().unwrap_or(Value::Null);

    let mut m = Metrics::default();
    m.model = sstr(engine.get("model")).unwrap_or_default();
    m.max_context = ndef(engine.get("max_context"), 0);
    m.images = engine.get("images").and_then(|v| v.as_bool()).unwrap_or(false);
    m.batch = ndef(engine.get("batch"), 0);
    m.batch_groups = ndef(engine.get("batch_groups"), 0);
    m.expert_slots = num(engine.get("expert_slots")).map(|x| x as i64);
    m.expert_cache_mib = num(engine.get("expert_cache_mib")).map(|x| x as i64);
    m.conversation_cache_mib = num(engine.get("conversation_cache_mib")).map(|x| x as i64);
    m.conversation_cache_slots = num(engine.get("conversation_cache_slots")).map(|x| x as i64);

    m.state = sstr(live.get("state")).unwrap_or_else(|| "idle".into());
    m.queued = ndef(live.get("queued"), 0);
    m.phase = sstr(live.get("phase"));
    m.prompt_tokens = num(live.get("prompt_tokens")).map(|x| x as i64);
    m.prompt_read = num(live.get("prompt_read")).map(|x| x as i64);
    m.prompt_total = num(live.get("prompt_total")).map(|x| x as i64);
    m.generated = num(live.get("generated")).map(|x| x as i64);
    m.max_tokens = num(live.get("max_tokens")).map(|x| x as i64);
    m.tok_s = num(live.get("tok_s"));
    m.prefill_tok_s_mean = num(live.get("prefill_tok_s_mean"));
    m.parallel = ndef(live.get("parallel"), 0);
    m.running = ndef(live.get("running"), 0);
    m.waiting = ndef(live.get("waiting"), 0);

    for (key, slot) in [
        ("gpu_util", &mut m.gpu_util),
        ("gpu_mem_used", &mut m.gpu_mem_used),
        ("gpu_mem_total", &mut m.gpu_mem_total),
        ("gpu_temp", &mut m.gpu_temp),
        ("gpu_power", &mut m.gpu_power),
        ("gpu_power_limit", &mut m.gpu_power_limit),
        ("gpu_pcie_rx_mb", &mut m.gpu_pcie_rx_mb),
        ("cpu", &mut m.cpu),
        ("ram_used", &mut m.ram_used),
        ("ram_total", &mut m.ram_total),
        ("disk_read_mb", &mut m.disk_read_mb),
        ("disk_write_mb", &mut m.disk_write_mb),
    ] {
        *slot = num(hw.get(key));
    }
    m.gpu_pcie_gen = num(hw.get("gpu_pcie_gen")).map(|x| x as i64);
    m.gpu_pcie_gen_max = num(hw.get("gpu_pcie_gen_max")).map(|x| x as i64);
    m.gpu_pcie_width = num(hw.get("gpu_pcie_width")).map(|x| x as i64);

    m.gpu_name = sstr(st.get("gpu_name"));
    m.cpu_name = sstr(st.get("cpu_name"));
    m.cores = num(st.get("cores")).map(|x| x as i64);
    m.threads = num(st.get("threads")).map(|x| x as i64);
    m.gpu_note = sstr(st.get("gpu_note"));

    if let Value::Object(map) = &hist {
        for (k, v) in map {
            if let Value::Array(items) = v {
                let s: Vec<f64> = items.iter().filter_map(|x| num(Some(x))).collect();
                if !s.is_empty() {
                    m.hist.insert(k.clone(), s);
                }
            }
        }
    }

    for (key, slot) in [("ttft", &mut m.ttft), ("itl", &mut m.itl), ("e2e", &mut m.e2e)] {
        let o = lat.get(key);
        slot.p50 = num(o.and_then(|o| o.get("p50")));
        slot.p90 = num(o.and_then(|o| o.get("p90")));
        slot.mean = num(o.and_then(|o| o.get("mean")));
        slot.count = ndef(o.and_then(|o| o.get("count")), 0);
    }

    m.tot_requests = ndef(totals.get("requests"), 0);
    m.tot_prompt_tokens = ndef(totals.get("prompt_tokens"), 0);
    m.tot_reused = ndef(totals.get("reused"), 0);
    m.tot_output_tokens = ndef(totals.get("output_tokens"), 0);
    m.tot_prompt_ms = num(totals.get("prompt_ms")).unwrap_or(0.0);
    m.tot_decode_ms = num(totals.get("decode_ms")).unwrap_or(0.0);
    m.totals_since = num(totals.get("since"));

    if conv.is_object() {
        let mut c = ConvCache::default();
        c.enabled = conv.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
        c.budget_mib = num(conv.get("budget_mib")).map(|x| x as i64);
        c.slots = num(conv.get("slots")).map(|x| x as i64);
        c.parked = num(conv.get("parked")).map(|x| x as i64);
        c.bytes = num(conv.get("bytes")).map(|x| x as i64);
        c.parks = num(conv.get("parks")).map(|x| x as i64);
        c.restores = num(conv.get("restores")).map(|x| x as i64);
        c.evictions = num(conv.get("evictions")).map(|x| x as i64);
        c.requests = ndef(conv.get("requests"), 0);
        c.requests_reused = ndef(conv.get("requests_reused"), 0);
        c.reused_tokens = ndef(conv.get("reused_tokens"), 0);
        c.prompt_tokens = ndef(conv.get("prompt_tokens"), 0);
        c.last_reused = num(conv.get("last_reused")).map(|x| x as i64);
        c.last_prompt = num(conv.get("last_prompt")).map(|x| x as i64);
        m.conv = Some(c);
    }

    if let Value::Array(items) = g("requests").cloned().unwrap_or(Value::Null) {
        for r in items {
            m.requests.push(RequestRow {
                time: num(r.get("time")).unwrap_or(0.0),
                finish: sstr(r.get("finish")).unwrap_or_default(),
                prompt_tokens: num(r.get("prompt_tokens")).map(|x| x as i64),
                reused: num(r.get("reused")).map(|x| x as i64),
                output_tokens: num(r.get("output_tokens")).map(|x| x as i64),
                prompt_ms: num(r.get("prompt_ms")),
                decode_tok_s: num(r.get("decode_tok_s")),
                hit_rate: num(r.get("hit_rate")),
                pcie_share: num(r.get("pcie_share")),
                duration_s: num(r.get("duration_s")),
            });
        }
    }
    m.requests_kept = ndef(g("requests_kept"), m.requests.len() as i64);
    Ok(m)
}

#[derive(Deserialize)]
pub struct Health {
    pub status: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub max_context: i64,
    #[serde(default)]
    pub images: bool,
    #[serde(default)]
    pub api_key: bool,
    #[serde(default)]
    pub loaded: bool,
    #[serde(default)]
    pub service: String,
}

// ---------------------------------------------------------------------------
// Minimal TCP HTTP/1.1 client (mirrors docker::UnixClient over TCP)
// ---------------------------------------------------------------------------

pub struct HttpClient {
    base: String,
    api_key: String,
}

impl HttpClient {
    pub fn new(cfg: &StrataConfig) -> Self {
        Self {
            base: cfg.base(),
            api_key: cfg.api_key.trim().to_string(),
        }
    }

    fn client(&self) -> Client<HttpConnector, EmptyBody> {
        Client::builder(TokioExecutor).build(HttpConnector)
    }

    async fn get(&self, path: &str) -> Result<(u16, Vec<u8>), String> {
        let uri: Uri = format!("{}{path}", self.base)
            .parse()
            .map_err(|e| format!("bad endpoint {:?}: {e}", self.base))?;
        let host = uri
            .authority()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "localhost".to_string());
        let mut b = hyper::Request::builder()
            .method("GET")
            .uri(uri)
            .header("Host", host)
            .header("Accept", "application/json");
        if !self.api_key.is_empty() {
            b = b.header("Authorization", format!("Bearer {}", self.api_key));
        }
        let req = b.body(EmptyBody).map_err(|e| e.to_string())?;
        let res = self
            .client()
            .request(req)
            .await
            .map_err(|e| format!("{path}: {e}"))?;
        let status = res.status().as_u16();
        let body = res
            .into_body()
            .collect()
            .await
            .map_err(|e| e.to_string())?
            .to_bytes();
        Ok((status, body.to_vec()))
    }

    /// GET /health: identity and reachability; no key needed.
    pub async fn health(&self) -> Result<Health, String> {
        let (status, body) = self.get("/health").await?;
        if !(200..300).contains(&status) {
            return Err(format!("/health: HTTP {status}"));
        }
        serde_json::from_slice(&body).map_err(|e| format!("/health: bad JSON: {e}"))
    }

    /// GET /metrics: the monitor payload. A 401 becomes a clear message.
    pub async fn metrics(&self) -> Result<Metrics, String> {
        let (status, body) = self.get("/metrics").await?;
        if status == 401 {
            return Err("the server needs an API key (401) — set it in the fields above".into());
        }
        if !(200..300).contains(&status) {
            let text = String::from_utf8_lossy(&body).chars().take(160).collect::<String>();
            return Err(format!("/metrics: HTTP {status} {text}"));
        }
        parse_metrics(&body)
    }
}

#[derive(Clone)]
struct HttpConnector;

impl tower_service::Service<Uri> for HttpConnector {
    type Response = TokioIo<tokio::net::TcpStream>;
    type Error = std::io::Error;
    type Future = Pin<Box<dyn Future<Output = Result<TokioIo<tokio::net::TcpStream>, std::io::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        Box::pin(async move {
            let host = uri.host().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "endpoint has no host")
            })?;
            let port = uri.port_u16().unwrap_or(80);
            let sock = tokio::net::lookup_host((host, port))
                .await?
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no address for host"))?;
            tokio::net::TcpStream::connect(sock).await.map(TokioIo::new)
        })
    }
}

/// Blocking GET over a plain std TcpStream, mirroring docker::UnixClient.
/// Used by the GUI, whose tokio runtime must not be relied on for hyper's
/// connection tasks; the TUI keeps the async path above.
pub fn get_blocking(cfg: &StrataConfig, path: &str) -> Result<(u16, Vec<u8>), String> {
    let base = cfg.base();
    let rest = base.split_once("://").map(|(_, r)| r).unwrap_or(&base);
    let (authority, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    // host:port split: an unbracketed IPv6 literal has colons but no port
    let (host_only, explicit_port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.contains(':') && !p.is_empty() => (h.to_string(), p.parse::<u16>().ok()),
        _ => (authority.to_string(), None),
    };
    let port = explicit_port.unwrap_or(if base.starts_with("https") { 443 } else { 80 });
    // resolve first, then connect each candidate in turn. A bare IPv6 zone
    // literal ([::1]) carries no scope id and cannot be connected; keep only
    // loopback/unscoped addresses from the resolver.
    let addrs = match std::net::ToSocketAddrs::to_socket_addrs(&(host_only, port)) {
        Ok(a) => a.filter(|a| !(a.is_ipv6() && a.ip().to_string().starts_with("fe80"))).collect::<Vec<_>>(),
        Err(e) => return Err(format!("{path}: resolve: {e}")),
    };
    let mut connected = None;
    let mut last_err = None;
    for a in addrs {
        match std::net::TcpStream::connect_timeout(&a, std::time::Duration::from_millis(3000)) {
            Ok(s) => {
                connected = Some(s);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let mut stream = match connected {
        Some(s) => s,
        None => return Err(format!("{path}: {}", last_err.map(|e| e.to_string()).unwrap_or_else(|| "no address".into()))),
    };
    stream.set_nodelay(true).ok();
    // every read is bounded: the real server closes the connection, but a
    // misbehaving one must never wedge the UI worker thread forever
    stream.set_read_timeout(Some(std::time::Duration::from_millis(3000))).ok();
    // path is absolute ("/health"); prefix is "" or "/some/base"
    let req = format!(
        "GET {prefix}{path} HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
        cfg.api_key.trim()
    );
    use std::io::{Read, Write};
    stream.write_all(req.as_bytes()).map_err(|e| format!("{path}: {e}"))?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("{path}: {e}"))?;
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| format!("{path}: truncated response"))?;
    let head = String::from_utf8_lossy(&buf[..split]).to_string();
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| format!("{path}: bad status line"))?;
    Ok((status, buf[split + 4..].to_vec()))
}

#[derive(Clone, Copy)]
struct TokioExecutor;

#[cfg(test)]
mod blocking_tests {
    use super::*;

    fn serve_one(response: &'static str) -> (u16, Vec<u8>, String) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            use std::io::{Read, Write};
            let mut req = [0u8; 2048];
            let n = sock.read(&mut req).unwrap_or(0);
            let head = String::from_utf8_lossy(&req[..n]).to_string();
            let body = "{\"ok\": true}";
            let resp = format!("HTTP/1.1 {response}\r\nContent-Length: {}\r\n\r\n{body}", body.len());
            sock.write_all(resp.as_bytes()).unwrap();
            head
        });
        let cfg = StrataConfig {
            endpoint: format!("http://127.0.0.1:{port}/"),
            api_key: "k".into(),
        };
        let out = get_blocking(&cfg, "/metrics").unwrap();
        let head = t.join().unwrap();
        (out.0, out.1, head)
    }

    #[test]
    fn blocking_get_builds_the_right_path_and_keeps_the_key() {
        let (status, body, head) = serve_one("200 OK");
        assert_eq!(status, 200);
        assert_eq!(body, b"{\"ok\": true}");
        // endpoint had a trailing slash; path is absolute → exactly /metrics
        assert!(head.contains("GET /metrics HTTP/1.1"), "{head}");
        assert!(head.contains("Authorization: Bearer k"), "{head}");
    }

    #[test]
    fn blocking_get_surfaces_error_statuses() {
        let (status, _, _) = serve_one("401 Unauthorized");
        assert_eq!(status, 401);
    }
}

impl hyper::rt::Executor<UnpinBoxFuture> for TokioExecutor {
    fn execute(&self, fut: UnpinBoxFuture) {
        // hyper uses this for its connection task. Spawning requires a tokio
        // context; when called from a plain thread, drive the future inline
        // instead of losing the request.
        match tokio::runtime::Handle::try_current() {
            Ok(h) => {
                h.spawn(fut);
            }
            Err(_) => {
                let mut fut = std::pin::pin!(fut);
                let noop = std::task::Waker::noop();
                let mut cx = std::task::Context::from_waker(noop);
                loop {
                    if fut.as_mut().poll(&mut cx).is_ready() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
        }
    }
}

type UnpinBoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> &'static str {
        r#"{
          "engine": {"model": "qwen3.8-flash-next-iq3_s", "max_context": 262144, "images": true,
                     "batch": 4, "batch_groups": 2, "expert_slots": 4252, "expert_cache_mib": 8192},
          "live": {"state": "generating", "queued": 0, "phase": "decode", "prompt_tokens": 512,
                   "generated": 100, "max_tokens": 2048, "tok_s": 751.4, "prefill_tok_s_mean": 1170.5,
                   "parallel": 4, "running": 2, "waiting": 1},
          "hardware": {"gpu_util": 96.0, "gpu_mem_used": 15400000000.0, "gpu_mem_total": 16106127360.0,
                       "gpu_temp": 50.0, "gpu_power": 155.0, "gpu_power_limit": 360.0,
                       "gpu_pcie_gen": 5, "gpu_pcie_width": 16, "gpu_pcie_rx_mb": 11.1,
                       "cpu": 55.0, "ram_used": 64000000000.0, "ram_total": 68000000000.0,
                       "disk_read_mb": 0.5, "disk_write_mb": 890.0},
          "hardware_static": {"gpu_name": "NVIDIA GeForce RTX 5080", "cpu_name": "AMD Ryzen 9 9950X",
                              "cores": 16, "threads": 32},
          "history": {"tok_s": [700.0, 751.4], "gpu_util": [90.0, 96.0]},
          "latency": {"ttft": {"p50": 0.44, "p90": 1.2, "mean": 0.6, "count": 509},
                      "itl": {"p50": 0.0008, "p90": 0.001, "mean": 0.0009, "count": 509},
                      "e2e": {"p50": 12.0, "p90": 60.0, "mean": 20.0, "count": 509}},
          "totals": {"since": 1761000000.0, "requests": 509, "prompt_tokens": 946912, "reused": 139490,
                     "output_tokens": 106225, "prompt_ms": 800000.0, "decode_ms": 500000.0},
          "conversation_cache": {"enabled": false, "budget_mib": null, "slots": null, "requests": 509,
                                 "requests_reused": 220, "reused_tokens": 139490, "prompt_tokens": 946912,
                                 "last_reused": 44864, "last_prompt": 47968},
          "requests": [
            {"time": 1761000900.0, "finish": "stop", "prompt_tokens": 47968, "reused": 44864,
             "output_tokens": 861, "prompt_ms": 100.0, "decode_tok_s": 131.5, "hit_rate": 0.782,
             "pcie_share": null, "duration_s": 7.4},
            {"time": 1761000800.0, "finish": "length", "prompt_tokens": 51277, "reused": 53429,
             "output_tokens": 922, "prompt_ms": 416.0, "decode_tok_s": 416.0, "hit_rate": 0.75,
             "duration_s": 7.2}],
          "requests_kept": 509
        }"#
    }

    #[test]
    fn parses_the_monitor_payload() {
        let m = parse_metrics(sample().as_bytes()).unwrap();
        assert_eq!(m.model, "qwen3.8-flash-next-iq3_s");
        assert_eq!(m.max_context, 262144);
        assert_eq!(m.state, "generating");
        assert_eq!(m.gpu_name.as_deref(), Some("NVIDIA GeForce RTX 5080"));
        assert_eq!(m.gpu_util, Some(96.0));
        assert_eq!(m.ttft.p50, Some(0.44));
        assert_eq!(m.tot_requests, 509);
        assert_eq!(m.requests.len(), 2);
        assert_eq!(m.requests_kept, 509);
        assert_eq!(m.hist.get("tok_s").map(|v| v.len()), Some(2));
        let conv = m.conv.clone().unwrap();
        assert!(!conv.enabled);
        assert_eq!(conv.requests_reused, 220);
    }

    #[test]
    fn missing_fields_stay_defaults() {
        let m = parse_metrics(br#"{"live": {"state": "idle"}}"#).unwrap();
        assert_eq!(m.state, "idle");
        assert_eq!(m.max_context, 0);
        assert!(m.gpu_util.is_none());
        assert!(m.requests.is_empty());
    }

    #[test]
    fn prefill_uses_fresh_tokens_only() {
        let m = parse_metrics(sample().as_bytes()).unwrap();
        let r = &m.requests[0];
        // 47968 - 44864 = 3104 fresh tokens over 100 ms = 31040 t/s
        assert_eq!(prefill_rate(r), Some(31040.0));
        // live state wins over the last request's number
        assert_eq!(prefill_now(&m, Some(r)), Some(1170.5));
        assert_eq!(decode_now(&m, Some(r)), Some(751.4));
    }

    #[test]
    fn context_used_follows_the_web_monitor() {
        let m = parse_metrics(sample().as_bytes()).unwrap();
        assert_eq!(context_used(&m, m.requests.first()), 512 + 100);
        let mut idle = m.clone();
        idle.state = "idle".into();
        let last = idle.requests.first().unwrap();
        assert_eq!(context_used(&idle, Some(last)), 47968 + 861);
        assert_eq!(context_used(&idle, None), 0);
    }

    /// `load`/`save` read `$HOME`; these tests point it at a scratch dir.
    /// `set_var` is process-wide, so they hold a lock while they run and
    /// restore the original value afterwards — other tests (and any later
    /// one) must never observe the scratch HOME.
    static HOME_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct HomeGuard {
        original: Option<std::ffi::OsString>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    fn isolate_home() -> HomeGuard {
        HomeGuard {
            original: std::env::var_os("HOME"),
            _lock: HOME_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner()),
        }
    }

    #[test]
    fn config_round_trip_merges_other_keys() {
        let _home = isolate_home();
        let dir = std::env::temp_dir().join(format!("syskamling-strata-cfg-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".config/syskamling")).unwrap();
        std::env::set_var("HOME", &dir);
                let path = dir.join(".config/syskamling/config.json");
        std::fs::write(&path, r#"{"other": 7}"#).unwrap();
        let cfg = StrataConfig {
            endpoint: "http://127.0.0.1:9191/".into(),
            api_key: "secret".into(),
        };
        save(&cfg).unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        let loaded = load();
        assert_eq!(loaded.endpoint, "http://127.0.0.1:9191/");
        assert_eq!(loaded.api_key, "secret");
        assert!(txt.contains("\"other\"")); // foreign keys survive
        assert_eq!(cfg.base(), "http://127.0.0.1:9191");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_survives_a_broken_file() {
        let _home = isolate_home();
        let dir = std::env::temp_dir().join(format!("syskamling-strata-bad-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".config/syskamling")).unwrap();
        std::env::set_var("HOME", &dir);
                std::fs::write(dir.join(".config/syskamling/config.json"), "not json").unwrap();
        assert_eq!(load(), StrataConfig::default());
        std::fs::remove_dir_all(&dir).ok();
    }
}
