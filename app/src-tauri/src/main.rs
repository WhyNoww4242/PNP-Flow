// pnpflow desktop shell (Tauri 2). UNBUILT: see app/README.md.
//
// The commands mirror the dev server's HTTP API (app/server.py, hud/server.py) so that
// app/ui/js/bridge.js can call either one with the same result shapes:
//   list_runs, run_listing, read_run_file, read_run_file_base64, read_run_log,
//   list_cases, launch_run, stop_run
//
// Configuration (environment):
//   PNPFLOW_ROOT    folder with run.py (default: the first ancestor of the current dir with run.py)
//   PNPFLOW_PYTHON  command that runs Python, split on whitespace. Default "python".
//                   On Windows with the solver in WSL:  "wsl -d Debian -- bash tools/wsl_run.sh python"
//                   (the solver then sees the root as /mnt/c/..., so keep PNPFLOW_ROOT on C:).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::Engine;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

const RUN_FILES: [&str; 5] = ["summary.json", "fields.json", "profiles.json", "progress.json", "stdout.log"];

struct Procs(Mutex<HashMap<String, (Child, Vec<String>, f64)>>);

fn re(pat: &'static str, cell: &'static OnceLock<Regex>) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("valid regex"))
}
fn name_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$", &R)
}
fn case_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$", &R)
}
fn key_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(r"^[A-Za-z_][A-Za-z0-9_.\-+]{0,63}$", &R)
}
/// Run-relative files the UI may read (same allowlist as app/server.py).
fn file_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    re(r"^(summary\.json|fields\.json|profiles\.json|progress\.json|stdout\.log|plots/[A-Za-z0-9_.\-+]{1,128}\.(png|jpg|jpeg|svg)|figures/[A-Za-z0-9_.\-+]{1,128}\.png|svg/[A-Za-z0-9_.\-+]{1,128}\.svg|mesh/(view_a\.png|view_b\.png|diagnose\.json))$", &R)
}

fn root() -> PathBuf {
    if let Ok(r) = std::env::var("PNPFLOW_ROOT") {
        return PathBuf::from(r);
    }
    let mut d = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if d.join("run.py").is_file() {
            return d;
        }
        if !d.pop() {
            return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        }
    }
}
fn results_dir() -> PathBuf {
    root().join("results")
}
fn python_cmd() -> Vec<String> {
    std::env::var("PNPFLOW_PYTHON")
        .unwrap_or_else(|_| "python".into())
        .split_whitespace()
        .map(String::from)
        .collect()
}
fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}
fn mtime(p: &Path) -> f64 {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Folder of run `name`, or an error when the name is unsafe or the folder escapes results/.
fn run_path(name: &str) -> Result<PathBuf, String> {
    if !name_re().is_match(name) || name.contains("..") {
        return Err("invalid run name".into());
    }
    let base = results_dir().canonicalize().map_err(|e| format!("results folder: {e}"))?;
    let p = base.join(name);
    let real = p.canonicalize().map_err(|_| format!("no such run: {name}"))?;
    if real.parent() != Some(base.as_path()) {
        return Err("forbidden".into());
    }
    Ok(real)
}

/// Absolute path of an allowed file inside a run folder.
fn run_file(name: &str, file: &str) -> Result<PathBuf, String> {
    if !file_re().is_match(file) {
        return Err("unknown run file".into());
    }
    let dir = run_path(name)?;
    let real = dir.join(file).canonicalize().map_err(|_| format!("not found: {file}"))?;
    if !real.starts_with(&dir) {
        return Err("forbidden".into());
    }
    Ok(real)
}

fn read_json(p: &Path) -> Value {
    fs::read_to_string(p)
        .ok()
        .and_then(|s| serde_json::from_str(&s.replace("NaN", "null").replace("-Infinity", "null").replace("Infinity", "null")).ok())
        .unwrap_or(Value::Null)
}

fn proc_status(procs: &tauri::State<Procs>, name: &str) -> Value {
    let mut map = procs.0.lock().unwrap();
    match map.get_mut(name) {
        None => Value::Null,
        Some((child, _, started)) => {
            let code = child.try_wait().ok().flatten();
            json!({"pid": child.id(), "alive": code.is_none(), "returncode": code.and_then(|c| c.code()), "started": started})
        }
    }
}

fn list_dir(dir: &Path, sub: &str) -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(dir.join(sub))
        .map(|it| it.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    out.retain(|f| file_re().is_match(&format!("{sub}/{f}")));
    out.sort();
    out
}

fn thumb_for(dir: &Path) -> Value {
    let figs = list_dir(dir, "figures");
    for want in ["field_phi.png", "field_cplus.png"] {
        if figs.iter().any(|f| f == want) {
            return json!(format!("figures/{want}"));
        }
    }
    if let Some(f) = figs.iter().find(|f| f.starts_with("field")).or(figs.first()) {
        return json!(format!("figures/{f}"));
    }
    let plots: Vec<String> = list_dir(dir, "plots").into_iter().filter(|p| p.ends_with(".png")).collect();
    for want in ["field_phi.png", "field_cplus.png"] {
        if plots.iter().any(|f| f == want) {
            return json!(format!("plots/{want}"));
        }
    }
    plots.iter().find(|p| p.starts_with("field_")).or(plots.first()).map(|p| json!(format!("plots/{p}"))).unwrap_or(Value::Null)
}

fn run_entry(procs: &tauri::State<Procs>, name: &str, dir: &Path) -> Value {
    let summary = read_json(&dir.join("summary.json"));
    let progress = read_json(&dir.join("progress.json"));
    let has_summary = summary.is_object();
    let proc = proc_status(procs, name);
    let pstate = progress.get("state").and_then(|s| s.as_str()).unwrap_or("");
    let alive = proc.get("alive").and_then(|a| a.as_bool());
    let state = match alive {
        Some(true) => if pstate.is_empty() { "launched".to_string() } else { pstate.to_string() },
        Some(false) if pstate != "done" && pstate != "failed" => {
            let ok = proc.get("returncode").and_then(|c| c.as_i64()) == Some(0);
            if ok && has_summary { "done".into() } else { "failed".into() }
        }
        _ if !pstate.is_empty() => pstate.to_string(),
        _ => if has_summary { "done".into() } else { "unknown".into() },
    };
    let order = |s: &str| match s { "fail" => 3, "warn" => 2, "pass" => 1, _ => 0 };
    let worst = summary.get("checks").and_then(|c| c.as_array()).and_then(|cs| {
        cs.iter().filter_map(|c| c.get("status").and_then(|s| s.as_str())).max_by_key(|s| order(s)).map(String::from)
    });
    let mut latest = mtime(dir);
    for f in ["progress.json", "summary.json", "fields.json", "stdout.log"] {
        latest = latest.max(mtime(&dir.join(f)));
    }
    let mut prog = Map::new();
    for k in ["state", "stage", "n_stages", "iteration", "res_norm", "message", "time_s"] {
        if let Some(v) = progress.get(k) {
            prog.insert(k.into(), v.clone());
        }
    }
    json!({
        "name": name,
        "case": summary.get("case").cloned().unwrap_or(Value::Null),
        "created": summary.get("created").cloned().unwrap_or(Value::Null),
        "mtime": latest,
        "state": state,
        "converged": summary.pointer("/solver/converged").cloned().unwrap_or(Value::Null),
        "time_s": summary.pointer("/solver/time_s").cloned().unwrap_or(Value::Null),
        "n_nodes": summary.pointer("/mesh/n_nodes").cloned().unwrap_or(Value::Null),
        "n_elements": summary.pointer("/mesh/n_elements").cloned().unwrap_or(Value::Null),
        "checks": worst,
        "files": RUN_FILES.iter().filter(|f| dir.join(f).exists()).collect::<Vec<_>>(),
        "progress": prog,
        "proc": proc,
        "thumb": thumb_for(dir),
    })
}

#[tauri::command]
fn list_runs(procs: tauri::State<Procs>) -> Result<Value, String> {
    let base = results_dir();
    let mut runs: Vec<Value> = Vec::new();
    if let Ok(it) = fs::read_dir(&base) {
        for e in it.filter_map(|e| e.ok()) {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Ok(dir) = run_path(&name) {
                if dir.is_dir() {
                    runs.push(run_entry(&procs, &name, &dir));
                }
            }
        }
    }
    runs.sort_by(|a, b| b["mtime"].as_f64().unwrap_or(0.0).partial_cmp(&a["mtime"].as_f64().unwrap_or(0.0)).unwrap());
    Ok(json!({"runs": runs, "results": base.to_string_lossy()}))
}

#[tauri::command]
fn run_listing(procs: tauri::State<Procs>, name: String) -> Result<Value, String> {
    let dir = run_path(&name)?;
    let mut files = Map::new();
    for f in RUN_FILES {
        let p = dir.join(f);
        if let Ok(m) = fs::metadata(&p) {
            files.insert(f.into(), json!({"size": m.len(), "mtime": mtime(&p)}));
        }
    }
    let mesh: Vec<&str> = ["mesh/view_a.png", "mesh/view_b.png", "mesh/diagnose.json"].into_iter().filter(|f| dir.join(f).is_file()).collect();
    Ok(json!({
        "name": name, "files": files, "plots": list_dir(&dir, "plots"), "figures": list_dir(&dir, "figures"),
        "svgs": list_dir(&dir, "svg"), "mesh_files": mesh, "thumb": thumb_for(&dir),
        "entry": run_entry(&procs, &name, &dir), "launch_params": {},
    }))
}

/// Text content of a run file (JSON or log); the front end parses JSON itself.
#[tauri::command]
fn read_run_file(name: String, file: String) -> Result<String, String> {
    fs::read_to_string(run_file(&name, &file)?).map_err(|e| e.to_string())
}

/// Binary run file (images) as base64, shown through a data: URL.
#[tauri::command]
fn read_run_file_base64(name: String, file: String) -> Result<String, String> {
    let bytes = fs::read(run_file(&name, &file)?).map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

#[tauri::command]
fn read_run_log(procs: tauri::State<Procs>, name: String, tail: Option<u64>) -> Result<Value, String> {
    let p = run_path(&name)?.join("stdout.log");
    let n = tail.unwrap_or(65536).clamp(1024, 4 << 20);
    let (text, size) = match fs::File::open(&p) {
        Err(_) => (String::new(), 0),
        Ok(mut f) => {
            let size = f.metadata().map(|m| m.len()).unwrap_or(0);
            if size > n {
                let _ = f.seek(SeekFrom::Start(size - n));
            }
            let mut buf = Vec::new();
            let _ = f.read_to_end(&mut buf);
            (String::from_utf8_lossy(&buf).into_owned(), size)
        }
    };
    Ok(json!({"text": text, "size": size, "proc": proc_status(&procs, &name)}))
}

#[tauri::command]
fn list_cases() -> Result<Value, String> {
    let cmd = python_cmd();
    let out = Command::new(&cmd[0]).args(&cmd[1..]).arg("run.py").arg("--list").current_dir(root())
        .output().map_err(|e| format!("could not start Python ({}): {e}", cmd.join(" ")))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (i, j) = (text.find('{'), text.rfind('}'));
    match (i, j) {
        (Some(i), Some(j)) if j > i => serde_json::from_str(&text[i..=j]).map_err(|e| format!("run.py --list: {e}")),
        _ => Err(format!("run.py --list printed no JSON: {}", String::from_utf8_lossy(&out.stderr).chars().rev().take(400).collect::<String>().chars().rev().collect::<String>())),
    }
}

#[derive(Deserialize)]
struct LaunchRequest {
    case: String,
    name: String,
    #[serde(default)]
    params: Map<String, Value>,
    #[serde(default)]
    overwrite: bool,
}

#[tauri::command]
fn launch_run(procs: tauri::State<Procs>, request: LaunchRequest) -> Result<Value, String> {
    if !case_re().is_match(&request.case) {
        return Err("invalid case name".into());
    }
    if !name_re().is_match(&request.name) || request.name.contains("..") {
        return Err("invalid run name (letters, digits, _ . - only)".into());
    }
    if proc_status(&procs, &request.name).get("alive").and_then(|a| a.as_bool()) == Some(true) {
        return Err(format!("run '{}' is already running", request.name));
    }
    let dir = results_dir().join(&request.name);
    let non_empty = fs::read_dir(&dir).map(|mut it| it.next().is_some()).unwrap_or(false);
    if non_empty && !request.overwrite {
        return Err(format!("results/{} already exists", request.name));
    }
    let py = python_cmd();
    let mut args: Vec<String> = py[1..].to_vec();
    args.extend(["-u".into(), "run.py".into(), request.case.clone(), "--name".into(), request.name.clone()]);
    for (k, v) in &request.params {
        let v = match v { Value::String(s) => s.clone(), other => other.to_string() };
        if !key_re().is_match(k) || v.contains(|c: char| c == '\r' || c == '\n' || c == '\0') {
            return Err(format!("invalid parameter {k}"));
        }
        args.extend(["--set".into(), format!("{k}={v}")]);
    }
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    if request.overwrite {
        let _ = fs::remove_file(dir.join("progress.json"));
    }
    let mut log = fs::File::create(dir.join("stdout.log")).map_err(|e| e.to_string())?;
    let cmdline: Vec<String> = std::iter::once(py[0].clone()).chain(args.iter().cloned()).collect();
    let _ = writeln!(log, "# launched by pnpflow app\n# {}", cmdline.join(" "));
    let err = log.try_clone().map_err(|e| e.to_string())?;
    let child = Command::new(&py[0]).args(&args).current_dir(root()).env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null()).stdout(log).stderr(err)
        .spawn().map_err(|e| format!("spawn failed: {e}"))?;
    let pid = child.id();
    procs.0.lock().unwrap().insert(request.name.clone(), (child, cmdline.clone(), now()));
    Ok(json!({"ok": true, "name": request.name, "pid": pid, "cmd": cmdline}))
}

#[tauri::command]
fn stop_run(procs: tauri::State<Procs>, name: String) -> Result<Value, String> {
    let mut map = procs.0.lock().unwrap();
    match map.get_mut(&name) {
        Some((child, _, _)) if child.try_wait().ok().flatten().is_none() => {
            child.kill().map_err(|e| e.to_string())?;
            Ok(json!({"ok": true}))
        }
        _ => Err(format!("no running process for '{name}' launched by this app")),
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Procs(Mutex::new(HashMap::new())))
        .invoke_handler(tauri::generate_handler![
            list_runs, run_listing, read_run_file, read_run_file_base64, read_run_log,
            list_cases, launch_run, stop_run
        ])
        .run(tauri::generate_context!())
        .expect("error while running the pnpflow app");
}
