//! Real application icons: resolve freedesktop icon names ("firefox",
//! "org.gimp.GIMP") to pixels using the system icon theme, and keep a small
//! texture cache for the GUI. PNGs decode via `image`; plain SVGs go through
//! the tiny rasterizer below (rects, circles, ellipses, simple paths);
//! `.svgz` is gunzipped first. Anything exotic (gradients, filters, masks)
//! simply fails to raster, and the caller falls back to the drawn glyph.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

/// One decoded icon: RGBA8 pixels, square, premultiplied alpha.
#[derive(Clone)]
pub struct IconRgba {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

/// Icon name -> decoded pixels (None = looked, found nothing).
static PIXELS: LazyLock<Mutex<HashMap<String, Option<IconRgba>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Icon name -> on-disk file ("" = none).
static PATHS: LazyLock<Mutex<HashMap<String, Option<Found>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Debug)]
pub struct Found {
    pub path: PathBuf,
}

/// Where a theme keeps its files, most specific first.
pub fn theme_roots() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        v.push(PathBuf::from(home).join(".local/share/icons"));
    }
    if let Some(p) = std::env::var_os("XDG_DATA_DIRS") {
        v.extend(std::env::split_paths(&p).map(|d| d.join("icons")));
    }
    v.push(PathBuf::from("/usr/share/icons"));
    v.push(PathBuf::from("/usr/share/pixmaps"));
    v
}

/// Theme inheritance chain: breeze-dark -> breeze -> hicolor.
pub fn theme_chain(roots: &[PathBuf], theme: &str) -> Vec<(PathBuf, String)> {
    let mut chain = Vec::new();
    let mut t = theme.to_string();
    for _ in 0..6 {
        if t.is_empty() {
            break;
        }
        for r in roots {
            if r.join(&t).is_dir() {
                chain.push((r.clone(), t.clone()));
                break;
            }
        }
        // follow inherits= from index.theme
        let mut next = String::new();
        for (r, name) in &chain {
            if let Ok(txt) = fs::read_to_string(r.join(name).join("index.theme")) {
                for line in txt.lines() {
                    if let Some(v) = line.strip_prefix("Inherits=") {
                        next = v.split(',').next().unwrap_or("").trim().to_string();
                        break;
                    }
                }
                break;
            }
        }
        t = next;
    }
    // the fallback theme already lives under a root; don't re-join the name
    chain.push((PathBuf::from("/usr/share/icons"), "hicolor".into()));
    chain.push((PathBuf::from("/usr/share/pixmaps"), String::new()));
    chain
}

/// Best file for one icon name: prefer the current theme, scalable SVGs,
/// then the size bucket nearest `want`. An absolute path (from a snap or
/// AppImage .desktop entry) is used directly.
pub fn find_icon(name: &str, want: u32) -> Option<Found> {
    if name.starts_with('/') {
        let p = PathBuf::from(name);
        let f = p.is_file().then_some(Found { path: p });
        PATHS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(name.to_string(), f.clone());
        return f;
    }
    {
        let m = PATHS.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(hit) = m.get(name) {
            return hit.clone();
        }
    }
    let theme = std::env::var("XDG_ICON_THEME")
        .or_else(|_| std::env::var("GTK_THEME"))
        .unwrap_or_else(|_| "hicolor".into());
    let theme = theme.split(':').next().unwrap_or("hicolor").to_string();
    let roots = theme_roots();
    let chain = theme_chain(&roots, &theme);
    let base = name
        .rsplit('/')
        .next()
        .unwrap_or(name)
        .rsplit('.')
        .next()
        .unwrap_or(name);
    let mut best: Option<Found> = None;
    let mut best_score: i64 = i64::MIN;
    for (root, tname) in &chain {
        if tname.is_empty() {
            continue; // /usr/share/pixmaps is tried after the theme chain
        }
        let dir = root.join(tname);
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        // "48x48" ends in 'x' too: strip the Nx prefix before parsing
        let mut sizes: Vec<(u32, PathBuf)> = Vec::new();
        for e in rd.flatten() {
            let f = e.file_name().to_string_lossy().into_owned();
            if f == "scalable" || f == "symbolic" {
                sizes.push((u32::MAX, dir.join(f)));
                continue;
            }
            // bucket dirs look like "48x48", "64x64+", "32x32@2"
            let stem = f.trim_end_matches('+').trim_end_matches('@');
            if let Some((a, b)) = stem.split_once('x') {
                if a == b {
                    if let Ok(n) = a.parse::<u32>() {
                        sizes.push((n, dir.join(f)));
                    }
                }
            }
        }
        // nearest bucket first, scalable wins
        sizes.sort_by_key(|(n, _)| {
            if *n == u32::MAX {
                0i64
            } else {
                1 + (*n as i64 - want as i64).abs()
            }
        });
        for (n, sdir) in sizes {
            for sub in [
                "apps",
                "categories",
                "places",
                "mimetypes",
                "actions",
                "status",
            ] {
                let sd = sdir.join(sub);
                for ext in ["svg", "svgz", "png"] {
                    let p = sd.join(format!("{base}.{ext}"));
                    if p.is_file() {
                        let score = if ext == "svg" || ext == "svgz" {
                            1000
                        } else {
                            // prefer the bucket closest to `want`
                            500 - (n as i64 - want as i64).abs().min(400)
                        };
                        if score > best_score {
                            best_score = score;
                            best = Some(Found { path: p });
                        }
                        break;
                    }
                }
            }
            if best.is_some() && best_score >= 1000 {
                break; // scalable found in this theme
            }
        }
        if best.is_some() {
            break; // first theme that has the icon wins
        }
    }
    if best.is_none() {
        // last resort: bare files in pixmaps dirs (root with empty theme)
        for (root, tname) in &chain {
            if !tname.is_empty() {
                continue;
            }
            for ext in ["svg", "svgz", "png", "xpm"] {
                let p = root.join(format!("{base}.{ext}"));
                if p.is_file() {
                    best = Some(Found { path: p });
                    break;
                }
            }
            if best.is_some() {
                break;
            }
        }
    }
    PATHS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(name.to_string(), best.clone());
    best
}

/// Decode (and cache) the pixels for an icon name.
pub fn pixels(name: &str, want: u32) -> Option<IconRgba> {
    let hit = PIXELS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(name)
        .cloned();
    if let Some(hit) = hit {
        return hit;
    }
    let out = load_pixels(name, want);
    PIXELS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(name.to_string(), out.clone());
    out
}

fn load_pixels(name: &str, want: u32) -> Option<IconRgba> {
    let f = find_icon(name, want)?;
    let raw = fs::read(&f.path).ok()?;
    if f.path.extension().and_then(|e| e.to_str()) == Some("svgz") {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .ok()?;
        raster_svg(&String::from_utf8_lossy(&out), want.max(48))
    } else if f.path.extension().and_then(|e| e.to_str()) == Some("svg") {
        raster_svg(&String::from_utf8_lossy(&raw), want.max(48))
    } else {
        let img = image::load_from_memory(&raw).ok()?.to_rgba8();
        let (w, h) = img.dimensions();
        Some(IconRgba {
            w,
            h,
            px: premultiply(img.as_raw()),
        })
    }
}

fn premultiply(px: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(px.len());
    for p in px.as_chunks::<4>().0 {
        let a = p[3] as u32;
        out.push((p[0] as u32 * a / 255) as u8);
        out.push((p[1] as u32 * a / 255) as u8);
        out.push((p[2] as u32 * a / 255) as u8);
        out.push(p[3]);
    }
    out
}

// ---------------------------------------------------------------------------
// Tiny SVG rasterizer. Enough for the flat app icons that ship in hicolor
// (solid fills, simple path data). Returns None on anything it can't do
// (gradients, filters, masks, text, clip paths, transforms).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rgba(f32, f32, f32, f32);

fn parse_color(s: &str) -> Option<Rgba> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        let d = |a: &str, b: &str| {
            u8::from_str_radix(a, 16)
                .ok()
                .zip(u8::from_str_radix(b, 16).ok())
        };
        return match hex.len() {
            3 => Some(Rgba(
                u8::from_str_radix(&hex[0..1], 16).ok()? as f32 / 15.0,
                u8::from_str_radix(&hex[1..2], 16).ok()? as f32 / 15.0,
                u8::from_str_radix(&hex[2..3], 16).ok()? as f32 / 15.0,
                1.0,
            )),
            6 => {
                let (r, g) = d(&hex[0..2], &hex[2..4])?;
                let (b, _) = d(&hex[4..6], "00")?;
                Some(Rgba(
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                    1.0,
                ))
            }
            8 => {
                let (r, g) = d(&hex[0..2], &hex[2..4])?;
                let (b, a) = d(&hex[4..6], &hex[6..8])?;
                Some(Rgba(
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                    a as f32 / 255.0,
                ))
            }
            _ => None,
        };
    }
    let named = match s.to_lowercase().as_str() {
        "black" => (0.0, 0.0, 0.0),
        "white" => (1.0, 1.0, 1.0),
        "red" => (1.0, 0.0, 0.0),
        "green" => (0.0, 0.5, 0.0),
        "blue" => (0.0, 0.0, 1.0),
        "yellow" => (1.0, 1.0, 0.0),
        "orange" => (1.0, 0.647, 0.0),
        "purple" => (0.5, 0.0, 0.5),
        "gray" | "grey" => (0.5, 0.5, 0.5),
        "silver" => (0.75, 0.75, 0.75),
        "none" => return None,
        _ => return None,
    };
    Some(Rgba(named.0, named.1, named.2, 1.0))
}

/// One shape in scanline order: a list of spans per y-band is computed at
/// raster time; here we keep the primitives.
enum Shape {
    Rect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        rx: f32,
        fill: Rgba,
        op: f32,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
        fill: Rgba,
        op: f32,
    },
    Poly {
        pts: Vec<(f32, f32)>,
        closed: bool,
        fill: Option<Rgba>,
        stroke: Option<(f32, Rgba)>,
        op: f32,
    },
}

fn attrs(tag: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes: Vec<char> = tag.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i].is_whitespace() || bytes[i] == '/') {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i] != '=' && !bytes[i].is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != '=' {
            continue;
        }
        let key: String = bytes[start..i].iter().collect();
        i += 1;
        if i >= bytes.len() || (bytes[i] != '"' && bytes[i] != '\'') {
            continue;
        }
        let q = bytes[i];
        i += 1;
        let vstart = i;
        while i < bytes.len() && bytes[i] != q {
            i += 1;
        }
        let val: String = bytes[vstart..i.min(bytes.len())].iter().collect();
        i += 1;
        out.push((key, val));
    }
    out
}

fn get<'a>(a: &'a [(String, String)], k: &str) -> Option<&'a str> {
    a.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.as_str())
}

fn fnum(a: &[(String, String)], k: &str) -> Option<f32> {
    get(a, k).and_then(|v| v.trim().trim_end_matches("px").trim().parse().ok())
}

/// CSS-ish fill resolution: fill attr, then style="fill:…".
fn fill_of(a: &[(String, String)]) -> Option<Rgba> {
    if let Some(v) = get(a, "fill") {
        if v.trim() == "none" {
            return None;
        }
        if let Some(c) = parse_color(v) {
            return Some(c);
        }
    }
    if let Some(st) = get(a, "style") {
        for part in st.split(';') {
            if let Some(v) = part.split(':').find_map(|p| {
                let p = p.trim();
                (!p.starts_with("fill") && !p.contains('#')).then_some(p)
            }) {
                let _ = v;
            }
            if let Some(v) = part.trim().strip_prefix("fill:") {
                if v.trim() == "none" {
                    return None;
                }
                if let Some(c) = parse_color(v) {
                    return Some(c);
                }
            }
        }
    }
    Some(Rgba(0.0, 0.0, 0.0, 1.0))
}

fn opacity_of(a: &[(String, String)]) -> f32 {
    let mut o = get(a, "opacity")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0f32);
    if let Some(v) = get(a, "fill-opacity").and_then(|v| v.parse::<f32>().ok()) {
        o *= v;
    }
    if let Some(st) = get(a, "style") {
        for part in st.split(';') {
            if let Some(v) = part.trim().strip_prefix("opacity:") {
                if let Ok(v) = v.trim().parse::<f32>() {
                    o *= v;
                }
            }
        }
    }
    o.clamp(0.0, 1.0)
}

/// Tokenize path data into (command, numbers).
fn path_tokens(d: &str) -> Vec<(char, Vec<f32>)> {
    let mut out = Vec::new();
    let mut cur: Option<(char, Vec<f32>)> = None;
    let mut num = String::new();
    let flush_num = |num: &mut String, cur: &mut Option<(char, Vec<f32>)>| {
        if !num.is_empty() {
            if let Some((_, v)) = cur.as_mut() {
                if let Ok(x) = num.parse::<f32>() {
                    v.push(x);
                }
            }
            num.clear();
        }
    };
    for c in d.chars() {
        if c.is_ascii_alphabetic() && c != 'e' && c != 'E' {
            flush_num(&mut num, &mut cur);
            if let Some(seg) = cur.take() {
                out.push(seg);
            }
            cur = Some((c, Vec::new()));
        } else if c == ',' || c.is_whitespace() {
            flush_num(&mut num, &mut cur);
        } else if (c == '-' || c == '.')
            && !num.is_empty()
            && !num.ends_with('e')
            && !num.ends_with('E')
        {
            flush_num(&mut num, &mut cur);
            num.push(c);
        } else if c.is_ascii_digit() || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E' {
            num.push(c);
        }
    }
    flush_num(&mut num, &mut cur);
    if let Some(seg) = cur {
        out.push(seg);
    }
    out
}

/// Flatten a path into polylines (absolute coords).
/// Flatten a path into polylines (absolute coords).
fn flatten(d: &str) -> Vec<Vec<(f32, f32)>> {
    let mut polys: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut cur: Vec<(f32, f32)> = Vec::new();
    let (mut cx, mut cy) = (0.0f32, 0.0f32);
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    let (mut px, mut py) = (0.0f32, 0.0f32);
    for (cmd, v) in path_tokens(d) {
        let abs = cmd.is_uppercase();
        let c = cmd.to_ascii_uppercase();
        // absolute target for the i-th coordinate pair (relative offsets
        // resolved against the current point)
        let g = |i: usize, cx: f32, cy: f32| -> (f32, f32) {
            let x = *v.get(i).unwrap_or(&0.0);
            let y = *v.get(i + 1).unwrap_or(&0.0);
            if abs {
                (x, y)
            } else {
                (cx + x, cy + y)
            }
        };
        match c {
            'M' => {
                if cur.len() > 1 {
                    polys.push(std::mem::take(&mut cur));
                }
                let (x, y) = g(0, cx, cy);
                cx = x;
                cy = y;
                sx = x;
                sy = y;
                cur.push((x, y));
                for i in (2..v.len()).step_by(2) {
                    let (x, y) = g(i, cx, cy);
                    cx = x;
                    cy = y;
                    cur.push((x, y));
                }
            }
            'L' | 'H' | 'V' => {
                let mut i = 0;
                while i < v.len() {
                    if c == 'H' {
                        cx = if abs { v[i] } else { cx + v[i] };
                    } else if c == 'V' {
                        cy = if abs { v[i] } else { cy + v[i] };
                    } else {
                        let (x, y) = g(i, cx, cy);
                        cx = x;
                        cy = y;
                        i += 1;
                    }
                    cur.push((cx, cy));
                    i += 1;
                }
            }
            'C' => {
                let mut i = 0;
                while i + 5 < v.len() {
                    let (x1, y1) = g(i, cx, cy);
                    let (x2, y2) = g(i + 2, cx, cy);
                    let (x, y) = g(i + 4, cx, cy);
                    let (x0, y0) = (cx, cy);
                    for k in 1..=12 {
                        let t = k as f32 / 12.0;
                        let mt = 1.0 - t;
                        let bx = mt * mt * mt * x0
                            + 3.0 * mt * mt * t * x1
                            + 3.0 * mt * t * t * x2
                            + t * t * t * x;
                        let by = mt * mt * mt * y0
                            + 3.0 * mt * mt * t * y1
                            + 3.0 * mt * t * t * y2
                            + t * t * t * y;
                        cur.push((bx, by));
                    }
                    cx = x;
                    cy = y;
                    px = x2;
                    py = y2;
                    i += 6;
                }
            }
            'S' => {
                let mut i = 0;
                while i + 3 < v.len() {
                    let (x1, y1) = (2.0 * cx - px, 2.0 * cy - py);
                    let (x2, y2) = g(i, cx, cy);
                    let (x, y) = g(i + 2, cx, cy);
                    let (x0, y0) = (cx, cy);
                    for k in 1..=12 {
                        let t = k as f32 / 12.0;
                        let mt = 1.0 - t;
                        let bx = mt * mt * mt * x0
                            + 3.0 * mt * mt * t * x1
                            + 3.0 * mt * t * t * x2
                            + t * t * t * x;
                        let by = mt * mt * mt * y0
                            + 3.0 * mt * mt * t * y1
                            + 3.0 * mt * t * t * y2
                            + t * t * t * y;
                        cur.push((bx, by));
                    }
                    cx = x;
                    cy = y;
                    px = x2;
                    py = y2;
                    i += 4;
                }
            }
            'Q' => {
                let mut i = 0;
                while i + 3 < v.len() {
                    let (qx, qy) = g(i, cx, cy);
                    let (x, y) = g(i + 2, cx, cy);
                    let (x0, y0) = (cx, cy);
                    for k in 1..=12 {
                        let t = k as f32 / 12.0;
                        let mt = 1.0 - t;
                        let bx = mt * mt * x0 + 2.0 * mt * t * qx + t * t * x;
                        let by = mt * mt * y0 + 2.0 * mt * t * qy + t * t * y;
                        cur.push((bx, by));
                    }
                    cx = x;
                    cy = y;
                    px = qx;
                    py = qy;
                    i += 4;
                }
            }
            'A' => {
                // approximate arcs with line segments (no rotation support)
                let mut i = 0;
                while i + 6 < v.len() {
                    let mut rx = v[i].abs().max(1e-4);
                    let mut ry = v[i + 1].abs().max(1e-4);
                    let laf = v[i + 3] != 0.0;
                    let sf = v[i + 4] != 0.0;
                    let (x, y) = g(i + 5, cx, cy);
                    let (x0, y0) = (cx, cy);
                    // center parameterization (simplified, rot=0)
                    let dx = (x0 - x) / 2.0;
                    let dy = (y0 - y) / 2.0;
                    let rad = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
                    if rad > 1.0 {
                        let s = rad.sqrt();
                        // radius scale-up per spec
                        rx *= s;
                        ry *= s;
                    }
                    let sign = if laf != sf { 1.0 } else { -1.0 };
                    let num = (rx * rx * (ry * ry) - rx * rx * dy * dy - ry * ry * dx * dx)
                        .max(0.0)
                        .sqrt();
                    let den = rx * rx * dy * dy + ry * ry * dx * dx;
                    let co = if den > 0.0 { sign * num / den } else { 0.0 };
                    let cxc = co * rx * dy + (x0 + x) / 2.0;
                    let cyc = -co * ry * dx + (y0 + y) / 2.0;
                    let ang = |vx: f32, vy: f32| -> f32 {
                        let m = (vx * vx + vy * vy).sqrt().max(1e-9);
                        let a = (vx / m).clamp(-1.0, 1.0).acos();
                        if vy < 0.0 {
                            2.0 * std::f32::consts::PI - a
                        } else {
                            a
                        }
                    };
                    let th0 = ang((x0 - cxc) / rx, (y0 - cyc) / ry);
                    let mut dth = ang((x - cxc) / rx, (y - cyc) / ry) - th0;
                    if sf && dth < 0.0 {
                        dth += 2.0 * std::f32::consts::PI;
                    }
                    if !sf && dth > 0.0 {
                        dth -= 2.0 * std::f32::consts::PI;
                    }
                    let steps = (dth.abs() / 0.2).ceil().max(2.0) as i32;
                    for k in 1..=steps {
                        let t = th0 + dth * k as f32 / steps as f32;
                        cur.push((cxc + rx * t.cos(), cyc + ry * t.sin()));
                    }
                    cx = x;
                    cy = y;
                    i += 7;
                }
            }
            'Z' => {
                if !cur.is_empty() {
                    cur.push((sx, sy));
                    polys.push(std::mem::take(&mut cur));
                }
                cx = sx;
                cy = sy;
            }
            _ => {}
        }
    }
    if cur.len() > 1 {
        polys.push(cur);
    }
    polys
}

pub fn raster_svg(svg: &str, target: u32) -> Option<IconRgba> {
    if svg.contains("<text")
        || svg.contains("filter:")
        || svg.contains("<filter")
        || svg.contains("clip-path")
    {
        return None;
    }
    // viewBox
    let open = svg.find("<svg")?;
    let open_end = svg[open..].find('>')? + open;
    let head = &svg[open + 4..open_end];
    let vb = attrs(head)
        .into_iter()
        .find(|(k, _)| k == "viewBox")
        .and_then(|(_, v)| {
            let n: Vec<f32> = v
                .split([',', ' '])
                .filter_map(|p| p.trim().parse().ok())
                .collect();
            (n.len() == 4).then_some((n[0], n[1], n[2], n[3]))
        });
    let (vb_x, vb_y, vb_w, vb_h) = vb.or_else(|| {
        let a = attrs(head);
        let w = fnum(&a, "width")?;
        let h = fnum(&a, "height")?;
        Some((0.0, 0.0, w, h))
    })?;
    if vb_w <= 0.0 || vb_h <= 0.0 {
        return None;
    }
    let n = target.clamp(16, 256) as usize;
    let scale = n as f32 / vb_w.max(vb_h);
    let tf = |x: f32, y: f32| -> (f32, f32) { ((x - vb_x) * scale, (y - vb_y) * scale) };

    // collect shapes
    let mut shapes: Vec<Shape> = Vec::new();
    let tags: Vec<(String, String, bool)> = {
        let mut out = Vec::new();
        let bytes: Vec<char> = svg.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == '<'
                && i + 1 < bytes.len()
                && bytes[i + 1] != '!'
                && bytes[i + 1] != '?'
                && bytes[i + 1] != '/'
            {
                let end = i + 1;
                let gt = match bytes[end..].iter().position(|c| *c == '>') {
                    Some(p) => p + end,
                    None => break,
                };
                let inner: String = bytes[end..gt].iter().collect();
                let selfclose = inner.ends_with('/');
                let name: String = inner
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                    .collect();
                let rest: String = inner.chars().skip(name.len()).collect();
                out.push((name, rest, selfclose));
                i = gt + 1;
            } else {
                i += 1;
            }
        }
        out
    };
    for (name, rest, _sc) in tags {
        let a = attrs(&rest);
        let op = opacity_of(&a);
        let stroke: Option<(f32, Rgba)> = get(&a, "stroke").and_then(parse_color).map(|c| {
            (
                get(&a, "stroke-width")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1.0f32),
                c,
            )
        });
        match name.as_str() {
            "rect" => {
                let fill = fill_of(&a);
                if let Some(f) = fill {
                    shapes.push(Shape::Rect {
                        x: fnum(&a, "x").unwrap_or(0.0),
                        y: fnum(&a, "y").unwrap_or(0.0),
                        w: match fnum(&a, "width") {
                            Some(v) => v,
                            None => continue,
                        },
                        h: match fnum(&a, "height") {
                            Some(v) => v,
                            None => continue,
                        },
                        rx: fnum(&a, "rx").unwrap_or(0.0),
                        fill: f,
                        op,
                    });
                }
            }
            "circle" => {
                let fill = fill_of(&a);
                if let Some(f) = fill {
                    let r = match fnum(&a, "r") {
                        Some(v) => v,
                        None => continue,
                    };
                    shapes.push(Shape::Ellipse {
                        cx: fnum(&a, "cx").unwrap_or(0.0),
                        cy: fnum(&a, "cy").unwrap_or(0.0),
                        rx: r,
                        ry: r,
                        fill: f,
                        op,
                    });
                }
            }
            "ellipse" => {
                let fill = fill_of(&a);
                if let Some(f) = fill {
                    let (rx, ry) = match (fnum(&a, "rx"), fnum(&a, "ry")) {
                        (Some(x), Some(y)) => (x, y),
                        _ => continue,
                    };
                    shapes.push(Shape::Ellipse {
                        cx: fnum(&a, "cx").unwrap_or(0.0),
                        cy: fnum(&a, "cy").unwrap_or(0.0),
                        rx,
                        ry,
                        fill: f,
                        op,
                    });
                }
            }
            "polygon" | "polyline" | "path" => {
                let fill = fill_of(&a);
                let pts = if name == "path" {
                    get(&a, "d").map(flatten).unwrap_or_default()
                } else {
                    let nums: Vec<f32> = get(&a, "points")
                        .unwrap_or("")
                        .split([',', ' ', '\n'])
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    let mut p = Vec::new();
                    for i in (0..nums.len()).step_by(2) {
                        if i + 1 < nums.len() {
                            p.push((nums[i], nums[i + 1]));
                        }
                    }
                    vec![p]
                };
                for poly in pts {
                    if poly.len() < 2 {
                        continue;
                    }
                    let closed = name != "polyline";
                    shapes.push(Shape::Poly {
                        pts: poly,
                        closed,
                        fill,
                        stroke,
                        op,
                    });
                }
            }
            _ => {}
        }
    }
    if shapes.is_empty() {
        return None;
    }

    // rasterize: painter's order, even-odd fill per polygon
    let mut buf = vec![0u8; n * n * 4];
    for sh in &shapes {
        match sh {
            Shape::Rect {
                x,
                y,
                w,
                h,
                rx,
                fill,
                op,
            } => {
                let (x0, y0) = tf(*x, *y);
                let (x1, y1) = tf(*x + w, *y + h);
                let r = rx * scale;
                for py in (y0.floor() as i32).max(0)..(y1.ceil() as i32).min(n as i32) {
                    for px in (x0.floor() as i32).max(0)..(x1.ceil() as i32).min(n as i32) {
                        let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                        if r > 0.0 {
                            let inside_tl = fx < x0 + r
                                && fy < y0 + r
                                && ((fx - x0 - r) * (fx - x0 - r) + (fy - y0 - r) * (fy - y0 - r))
                                    > r * r;
                            let inside_tr = fx > x1 - r
                                && fy < y0 + r
                                && ((fx - x1 + r) * (fx - x1 + r) + (fy - y0 - r) * (fy - y0 - r))
                                    > r * r;
                            let inside_bl = fx < x0 + r
                                && fy > y1 - r
                                && ((fx - x0 - r) * (fx - x0 - r) + (fy - y1 + r) * (fy - y1 + r))
                                    > r * r;
                            let inside_br = fx > x1 - r
                                && fy > y1 - r
                                && ((fx - x1 + r) * (fx - x1 + r) + (fy - y1 + r) * (fy - y1 + r))
                                    > r * r;
                            if inside_tl || inside_tr || inside_bl || inside_br {
                                continue;
                            }
                        }
                        blend(&mut buf, n, px, py, *fill, *op);
                    }
                }
            }
            Shape::Ellipse {
                cx,
                cy,
                rx,
                ry,
                fill,
                op,
            } => {
                let (ccx, ccy) = tf(*cx, *cy);
                let (rx, ry) = (*rx * scale, *ry * scale);
                for py in
                    ((ccy - ry).floor() as i32).max(0)..((ccy + ry).ceil() as i32).min(n as i32)
                {
                    for px in
                        ((ccx - rx).floor() as i32).max(0)..((ccx + rx).ceil() as i32).min(n as i32)
                    {
                        let dx = (px as f32 + 0.5 - ccx) / rx;
                        let dy = (py as f32 + 0.5 - ccy) / ry;
                        if dx * dx + dy * dy <= 1.0 {
                            blend(&mut buf, n, px, py, *fill, *op);
                        }
                    }
                }
            }
            Shape::Poly {
                pts,
                closed,
                fill,
                stroke,
                op,
            } => {
                let t: Vec<(f32, f32)> = pts.iter().map(|(x, y)| tf(*x, *y)).collect();
                if let Some(f) = fill {
                    if *closed {
                        let mut ys: Vec<f32> = t.iter().map(|p| p.1).collect();
                        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let (y0, y1) = (ys[0].floor() as i32, ys[ys.len() - 1].ceil() as i32);
                        for py in y0.max(0)..y1.min(n as i32) {
                            let sy = py as f32 + 0.5;
                            let mut xs: Vec<f32> = Vec::new();
                            for i in 0..t.len() - 1 {
                                let (a, b) = (t[i], t[i + 1]);
                                if (a.1 <= sy && b.1 > sy) || (b.1 <= sy && a.1 > sy) {
                                    xs.push(a.0 + (b.0 - a.0) * (sy - a.1) / (b.1 - a.1));
                                }
                            }
                            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            for pair in xs.chunks(2) {
                                if pair.len() == 2 {
                                    for px in (pair[0].floor() as i32).max(0)
                                        ..(pair[1].ceil() as i32).min(n as i32)
                                    {
                                        blend(&mut buf, n, px, py, *f, *op);
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some((sw, sc)) = stroke {
                    let sw = sw.max(0.5) * scale;
                    for i in 0..t.len() - 1 {
                        stroke_line(&mut buf, n, t[i], t[i + 1], sw, *sc, *op);
                    }
                }
            }
        }
    }
    Some(IconRgba {
        w: n as u32,
        h: n as u32,
        px: buf,
    })
}

fn blend(buf: &mut [u8], n: usize, x: i32, y: i32, c: Rgba, op: f32) {
    if x < 0 || y < 0 || x as usize >= n || y as usize >= n || op <= 0.0 {
        return;
    }
    let i = ((y as usize) * n + x as usize) * 4;
    let a = c.3 * op;
    let px = &mut buf[i..i + 4];
    let (sr, sg, sb, sa) = (
        px[0] as f32 / 255.0,
        px[1] as f32 / 255.0,
        px[2] as f32 / 255.0,
        px[3] as f32 / 255.0,
    );
    let oa = a + sa * (1.0 - a);
    if oa <= 0.0 {
        return;
    }
    px[0] = (((c.0 * a + sr * sa * (1.0 - a)) / oa) * 255.0) as u8;
    px[1] = (((c.1 * a + sg * sa * (1.0 - a)) / oa) * 255.0) as u8;
    px[2] = (((c.2 * a + sb * sa * (1.0 - a)) / oa) * 255.0) as u8;
    px[3] = (oa * 255.0) as u8;
}

fn stroke_line(buf: &mut [u8], n: usize, a: (f32, f32), b: (f32, f32), w: f32, c: Rgba, op: f32) {
    let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt().max(1e-6);
    let steps = (len * 2.0).ceil() as i32;
    for k in 0..=steps {
        let t = k as f32 / steps as f32;
        let cx = a.0 + (b.0 - a.0) * t;
        let cy = a.1 + (b.1 - a.1) * t;
        let r = w / 2.0;
        for py in ((cy - r).floor() as i32).max(0)..=(cy + r).ceil() as i32 {
            for px in ((cx - r).floor() as i32).max(0)..=(cx + r).ceil() as i32 {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                if dx * dx + dy * dy <= r * r {
                    blend(buf, n, px, py, c, op);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GUI texture cache: icon name -> egui texture, allocated lazily on the UI
// thread. The scan thread only supplies names.
// ---------------------------------------------------------------------------

pub struct TexCache {
    tex: HashMap<String, Option<egui::TextureHandle>>,
    epoch: u64,
}

impl TexCache {
    pub fn new() -> Self {
        Self {
            tex: HashMap::new(),
            epoch: 0,
        }
    }

    /// Texture for an icon name, or None when not decoded yet / no icon
    /// exists (caller draws the family glyph instead).
    pub fn get(&self, name: &str) -> Option<&egui::TextureHandle> {
        if name.is_empty() {
            return None;
        }
        self.tex.get(name).and_then(|o| o.as_ref())
    }

    /// Register a decoded texture.
    pub fn insert(&mut self, name: &str, tex: egui::TextureHandle) {
        self.tex.insert(name.to_string(), Some(tex));
    }

    /// Remember that an icon name has no usable icon, so it isn't retried.
    pub fn mark_missing(&mut self, name: &str) {
        self.tex.entry(name.to_string()).or_insert(None);
    }

    /// How many decoded (Some) textures the cache holds.
    #[cfg(test)]
    pub fn tex_count(&self) -> usize {
        self.tex.values().filter(|o| o.is_some()).count()
    }

    pub fn known(&self, name: &str) -> bool {
        self.tex.contains_key(name)
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Same, but only from already-decoded pixels (never blocks on disk I/O).
pub fn tint_cached(name: &str) -> Option<egui::Color32> {
    if name.is_empty() {
        return None;
    }
    let hit = PIXELS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(name)
        .cloned();
    if let Some(hit) = hit {
        return tint_of(&hit?);
    }
    // dashboard dots need a colour now, not after the next process scan:
    // decode the icon right here (theme lookup is cached, one decode ~1ms)
    let out = load_pixels(name, 48);
    PIXELS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(name.to_string(), out.clone());
    tint_of(&out?)
}

fn tint_of(i: &IconRgba) -> Option<egui::Color32> {
    let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
    for p in i.px.as_chunks::<4>().0 {
        if p[3] > 200 {
            r += p[0] as u64;
            g += p[1] as u64;
            b += p[2] as u64;
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some(egui::Color32::from_rgb(
        (r / n) as u8,
        (g / n) as u8,
        (b / n) as u8,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_icon_locates_a_real_system_icon() {
        // this machine has these; skip cleanly on machines that don't
        let f = find_icon("firefox", 48);
        if f.is_none() {
            return;
        }
        let f = f.unwrap();
        assert!(f.path.is_file());
        let ext = f.path.extension().unwrap().to_str().unwrap();
        assert!(matches!(ext, "svg" | "svgz" | "png"), "unexpected {ext}");
    }

    #[test]
    fn raster_svg_draws_shapes() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
            <rect x="4" y="4" width="56" height="56" rx="8" fill="#e0701f"/>
            <circle cx="32" cy="32" r="14" fill="white"/>
        </svg>"##;
        let img = raster_svg(svg, 64).expect("raster");
        assert_eq!((img.w, img.h), (64, 64));
        let at = |x: u32, y: u32| -> (u8, u8, u8, u8) {
            let i = ((y as usize) * 64 + x as usize) * 4;
            (img.px[i], img.px[i + 1], img.px[i + 2], img.px[i + 3])
        };
        // centre is the white circle, corner is the orange rounded rect
        let (r, g, b, a) = at(32, 32);
        assert!(
            a > 200 && r > 200 && g > 200 && b > 200,
            "centre {r},{g},{b},{a}"
        );
        let (r, g, b, a) = at(12, 32);
        assert!(
            a > 200 && r > 180 && g > 80 && b < 90,
            "side {r},{g},{b},{a}"
        );
        // outside the rounded rect: transparent
        assert!(at(0, 0).3 == 0);
    }

    #[test]
    fn raster_svg_paths_flatten_curves_and_arcs() {
        let svg = r##"<svg viewBox="0 0 100 100"><path d="M10 50 C30 10 70 10 90 50 A40 40 0 0 1 10 50 Z" fill="#2266aa"/></svg>"##;
        let img = raster_svg(svg, 100).expect("raster");
        let at = |x: usize, y: usize| img.px[(y * 100 + x) * 4 + 3];
        assert!(at(50, 30) > 200, "inside top");
        assert!(at(50, 80) > 200, "inside bottom arc");
        assert!(at(2, 2) == 0, "outside");
    }

    #[test]
    fn parse_color_accepts_hex_and_names() {
        assert_eq!(parse_color("#ff8800").unwrap().0, 1.0);
        assert_eq!(parse_color("#f80").unwrap().1, 0x88 as f32 / 255.0);
        assert_eq!(parse_color("white").unwrap().0, 1.0);
        assert!(parse_color("url(#grad)").is_none());
    }

    #[test]
    fn pixels_cache_round_trips() {
        let a = pixels("definitely-not-an-icon-xyz", 48);
        assert!(a.is_none());
        // the miss is remembered: a repeat lookup must not hit the disk again
        assert!(pixels("definitely-not-an-icon-xyz", 48).is_none());
        assert!(tint_cached("definitely-not-an-icon-xyz").is_none());
    }

    #[test]
    fn svgz_decodes_through_the_gzip_path() {
        // build a gzipped svg and run it through the same decode the loader uses
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(
            br##"<svg viewBox="0 0 32 32"><rect width="32" height="32" fill="red"/></svg>"##,
        )
        .unwrap();
        let gz = e.finish().unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&gz[..]), &mut out).unwrap();
        let img = raster_svg(&String::from_utf8_lossy(&out), 32).unwrap();
        assert!(img.px[3] > 200);
    }

    /// On a live desktop, the names the .desktop layer produces must
    /// actually decode: this is what the Processes rows and dashboard
    /// dots paint.
    #[test]
    fn live_desktop_names_decode_to_pixels() {
        let mut names: Vec<String> = Vec::new();
        for r in crate::procs::ProcScanner::new().scan() {
            if !r.app_icon.is_empty() && !names.contains(&r.app_icon) {
                names.push(r.app_icon.clone());
            }
        }
        let ok: Vec<String> = names
            .iter()
            .filter(|n| pixels(n, 48).is_some())
            .cloned()
            .collect();
        assert!(
            ok.len() >= 5,
            "only {}/{} live icons decoded: {names:?}",
            ok.len(),
            names.len()
        );
        // every decoded icon must yield a usable tint for the dashboard dot
        for n in &ok {
            assert!(tint_cached(n).is_some(), "{n} decoded but has no tint");
        }
    }
}
