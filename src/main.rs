use std::env;
use std::fs;
use std::io::{self, Write};
use std::process::Command;
use std::thread;
use std::time::Duration;

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{Event, KeyCode, KeyEvent};
use crossterm::execute;
use crossterm::style::{Color, SetForegroundColor};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size};
use yaml_rust::YamlLoader;

// ── Cleanup guard: ensures terminal is restored even on panic ──────────

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), Show);
        let _ = disable_raw_mode();
        let _ = write!(io::stdout(), "\x1b[?25h\x1b[2J\x1b[H");
        let _ = io::stdout().flush();
    }
}

// ── Data structures ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct GpuInfo {
    id: usize,
    temperature: f64,
    fan_speed: f64,
    power_usage: u32,
    power_cap: u32,
    memory_used: u32,
    memory_total: u32,
    gpu_util: f64,
}

#[derive(Debug, Default)]
struct InferenceStats {
    progress: f64,
    time_seconds: f64,
    tokens_per_second: f64,
    n_decoded: u32,
    gen_speed_tps: f64,
    latency_ms_tok: f64,
    draft_acceptance: f64,
    n_decoded_max: u32,
    ctx_n_tokens: u32,
    ctx_used: u32,
}

#[derive(Debug)]
struct Config {
    temp_low: String,
    temp_medium: String,
    temp_high: String,
    temp_critical: String,
    power: String,
    memory: String,
    util_low: String,
    util_medium: String,
    util_high: String,
    title: String,
    bar_empty: String,
    log_file: Option<String>,
    log_lines: usize,
    log_height: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            temp_low: "Cyan".to_string(),
            temp_medium: "Green".to_string(),
            temp_high: "Yellow".to_string(),
            temp_critical: "Red".to_string(),
            power: "Green".to_string(),
            memory: "Cyan".to_string(),
            util_low: "Green".to_string(),
            util_medium: "Yellow".to_string(),
            util_high: "Red".to_string(),
            title: "Cyan".to_string(),
            bar_empty: "DarkGrey".to_string(),
            log_file: None,
            log_lines: 10,
            log_height: 10,
        }
    }
}

// ── Color parsing ─────────────────────────────────────────────────────────

fn parse_color_str(color_str: &str) -> Color {
    match color_str {
        "Red" => Color::Red,
        "Green" => Color::Green,
        "Yellow" => Color::Yellow,
        "Blue" => Color::Blue,
        "Magenta" => Color::Magenta,
        "Cyan" => Color::Cyan,
        "White" => Color::White,
        "Black" => Color::Black,
        "DarkRed" => Color::DarkRed,
        "DarkGreen" => Color::DarkGreen,
        "DarkYellow" => Color::DarkYellow,
        "DarkBlue" => Color::DarkBlue,
        "DarkMagenta" => Color::DarkMagenta,
        "DarkCyan" => Color::DarkCyan,
        "DarkGrey" => Color::DarkGrey,
        "Grey" => Color::Grey,
        _ if color_str.starts_with("RGB") => {
            let parts: Vec<&str> = color_str
                .trim_start_matches("RGB(")
                .trim_end_matches(")")
                .split(',')
                .collect();
            if parts.len() == 3 {
                return Color::Rgb {
                    r: parts[0].trim().parse().unwrap_or(255),
                    g: parts[1].trim().parse().unwrap_or(255),
                    b: parts[2].trim().parse().unwrap_or(255),
                };
            }
            Color::White
        }
        _ => Color::White,
    }
}

fn get_temp_color(temp: f64, config: &Config) -> Color {
    if temp >= 85.0 {
        parse_color_str(&config.temp_critical)
    } else if temp >= 70.0 {
        parse_color_str(&config.temp_high)
    } else if temp >= 50.0 {
        parse_color_str(&config.temp_medium)
    } else {
        parse_color_str(&config.temp_low)
    }
}

fn get_util_color(util: f64, config: &Config) -> Color {
    if util >= 90.0 {
        parse_color_str(&config.util_high)
    } else if util >= 70.0 {
        parse_color_str(&config.util_medium)
    } else {
        parse_color_str(&config.util_low)
    }
}

// ── Config loading ────────────────────────────────────────────────────────

impl Config {
    fn yaml_str(docs: &[yaml_rust::Yaml], key: &str, default: &str) -> String {
        docs.get(0)
            .and_then(|d| d[key].as_str())
            .unwrap_or(default)
            .to_string()
    }

    fn yaml_usize(docs: &[yaml_rust::Yaml], key: &str, default: usize) -> usize {
        docs.get(0)
            .and_then(|d| d[key].as_i64())
            .map(|v| v as usize)
            .unwrap_or(default)
    }

    fn load() -> Self {
        let config_path = env::var("HOME")
            .map(|h| format!("{}/.config/nv-smi/config.yaml", h))
            .unwrap_or_default();

        if !fs::metadata(&config_path).is_ok() {
            return Config::default();
        }

        let content = fs::read_to_string(&config_path).unwrap_or_default();
        let docs = YamlLoader::load_from_str(&content).unwrap_or_default();

        Config {
            temp_low: Self::yaml_str(&docs, "temp_low", "Cyan"),
            temp_medium: Self::yaml_str(&docs, "temp_medium", "Green"),
            temp_high: Self::yaml_str(&docs, "temp_high", "Yellow"),
            temp_critical: Self::yaml_str(&docs, "temp_critical", "Red"),
            power: Self::yaml_str(&docs, "power", "Green"),
            memory: Self::yaml_str(&docs, "memory", "Cyan"),
            util_low: Self::yaml_str(&docs, "util_low", "Green"),
            util_medium: Self::yaml_str(&docs, "util_medium", "Yellow"),
            util_high: Self::yaml_str(&docs, "util_high", "Red"),
            title: Self::yaml_str(&docs, "title", "Cyan"),
            bar_empty: Self::yaml_str(&docs, "bar_empty", "DarkGrey"),
            log_file: docs
                .get(0)
                .and_then(|d| d["log_file"].as_str())
                .map(|s| s.to_string()),
            log_lines: Self::yaml_usize(&docs, "log_lines", 10),
            log_height: Self::yaml_usize(&docs, "log_height", 10),
        }
    }
}

// ── GPU info ──────────────────────────────────────────────────────────────

struct LlamaServerInfo {
    model: String,
    params: Vec<(String, String)>,
    context_len: u32,
    n_parallel: u32,
    multimodal: bool,
    embedding: bool,
}

fn get_nvidia_smi() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(o) => parse_gpus_csv(&String::from_utf8_lossy(&o.stdout)),
        Err(_) => Vec::new(),
    }
}

fn parse_gpus_csv(output: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    for (id, line) in output.lines().enumerate() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 7 {
            continue;
        }
        gpus.push(GpuInfo {
            id,
            temperature: parts[0].parse().unwrap_or(0.0),
            fan_speed: parts[1].parse().unwrap_or(0.0),
            power_usage: parts[2].parse().unwrap_or(0),
            power_cap: parts[3].parse().unwrap_or(0),
            memory_used: parts[4].parse().unwrap_or(0),
            memory_total: parts[5].parse().unwrap_or(0),
            gpu_util: parts[6].parse().unwrap_or(0.0),
        });
    }
    gpus
}

fn get_llama_server_info(config_log_file: Option<&str>) -> Option<LlamaServerInfo> {
    let output = Command::new("ps")
        .arg("aux")
        .output()
        .ok()?;
    let ps_text = String::from_utf8_lossy(&output.stdout);

    // If we have a config log_file, prefer the llama-server that matches it
    let mut exact_match: Option<LlamaServerInfo> = None;

    for line in ps_text.lines() {
        if !line.contains("llama-server") || line.contains("grep") {
            continue;
        }

        let args: Vec<&str> = line.split_whitespace().collect();
        let mut model = String::new();
        let mut params: Vec<(String, String)> = Vec::new();
        let mut proc_log_file: Option<String> = None;
        let mut multimodal: bool = false;
        let mut embedding: bool = false;

        let start = match args.iter().position(|&a| a.contains("llama-server")) {
            Some(pos) => pos + 1,
            None => return None,
        };

        let mut i = start;
        let mut context_len: u32 = 0;
        let mut n_parallel: u32 = 1;
        while i < args.len() {
            if !args[i].starts_with('-') {
                i += 1;
                continue;
            }

            match args[i] {
                "-m" => {
                    if i + 1 < args.len() {
                        model = args[i + 1].to_string();
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "-ngl" | "-t" | "-tb" | "--top-k" | "--top-p"
                | "--repeat-penalty" | "--temp" | "--port" | "--cache-reuse" => {
                    let key = args[i].trim_start_matches('-').to_string();
                    let val = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        args[i + 1].to_string()
                    } else {
                        "N/A".to_string()
                    };
                    params.push((key, val));
                    i += 2;
                }
                "-c" => {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        context_len = parse_u32(args[i + 1]);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "-np" => {
                    if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        n_parallel = parse_u32(args[i + 1]);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "-fa" => {
                    let val = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        args[i + 1].to_string()
                    } else {
                        "N/A".to_string()
                    };
                    params.push(("fa".to_string(), val));
                    i += 2;
                }
                "-ctk" | "-ctv" => {
                    let key = args[i].trim_start_matches('-').to_string();
                    let val = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        args[i + 1].to_string()
                    } else {
                        "N/A".to_string()
                    };
                    params.push((key, val));
                    i += 2;
                }
                "--cont-batching" | "--cache-idle-slots" => {
                    let key = args[i].trim_start_matches('-').to_string();
                    params.push((key, "on".to_string()));
                    i += 1;
                }
                "--log-file" => {
                    let val = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        args[i + 1].to_string()
                    } else {
                        "N/A".to_string()
                    };
                    proc_log_file = Some(val.clone());
                    params.push(("log-file".to_string(), val));
                    i += 2;
                }
                "--mmproj" => {
                    multimodal = true;
                    let val = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        args[i + 1].to_string()
                    } else {
                        "N/A".to_string()
                    };
                    params.push(("mmproj".to_string(), val));
                    i += 2;
                }
                "--embedding" => {
                    embedding = true;
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }

        if model.is_empty() {
            continue;
        }

        let info = LlamaServerInfo { model, params, context_len, n_parallel, multimodal, embedding };

        // If this process's --log-file matches our config, prefer it
        if let Some(config_lf) = config_log_file {
            if let Some(proc_lf) = proc_log_file {
                let proc_canonical = proc_lf.trim_start_matches('~').replace("$HOME", &std::env::var("HOME").unwrap_or_default());
                if proc_canonical == config_lf || proc_lf == config_lf {
                    exact_match = Some(info);
                }
            }
        } else {
            // No config log_file: first match wins (original behavior)
            return Some(info);
        }
    }

    exact_match
}

fn get_embedding_models(config_log_file: Option<&str>) -> Option<Vec<String>> {
    let output = Command::new("ps")
        .arg("aux")
        .output()
        .ok()?;
    let ps_text = String::from_utf8_lossy(&output.stdout);
    let mut models = Vec::new();

    for line in ps_text.lines() {
        if !line.contains("llama-server") || line.contains("grep") {
            continue;
        }

        let args: Vec<&str> = line.split_whitespace().collect();
        let mut model = String::new();
        let mut proc_log_file: Option<String> = None;
        let mut is_embedding = false;

        let start = match args.iter().position(|&a| a.contains("llama-server")) {
            Some(pos) => pos + 1,
            None => continue,
        };

        let mut i = start;
        while i < args.len() {
            if !args[i].starts_with('-') {
                i += 1;
                continue;
            }
            match args[i] {
                "-m" => {
                    model = if i + 1 < args.len() { args[i + 1].to_string() } else { String::new() };
                    i += if !model.is_empty() { 2 } else { 1 };
                }
                "--log-file" => {
                    proc_log_file = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        Some(args[i + 1].to_string())
                    } else {
                        None
                    };
                    i += 2;
                }
                "--embedding" => {
                    is_embedding = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }

        // Skip non-embedding or the main model (matches config log_file)
        if !is_embedding || model.is_empty() {
            continue;
        }
        if let Some(config_lf) = config_log_file {
            if let Some(proc_lf) = &proc_log_file {
                if proc_lf == config_lf {
                    continue;
                }
            }
        }
        models.push(model);
    }

    if models.is_empty() {
        None
    } else {
        Some(models)
    }
}

// ── Small parsing helpers ──────────────────────────────────────────────────

fn parse_float(value: &str) -> f64 {
    value
        .replace("%", "")
        .replace("C", "")
        .parse()
        .unwrap_or(0.0)
}

fn parse_u32(value: &str) -> u32 {
    value.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)
}

fn parse_gpus(output: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut gpu_id: usize = 0;
    for line in output.lines() {
        if line.starts_with('|') {
            let inner = line.trim_start_matches('|').trim_end_matches('|');
            let tokens: Vec<&str> = inner.split_whitespace().collect();
            if !tokens.is_empty() && tokens[0].contains('%') {
                if tokens.len() >= 13 {
                    gpus.push(GpuInfo {
                        id: gpu_id,
                        temperature: parse_float(tokens[1]),
                        fan_speed: parse_float(tokens[0]),
                        power_usage: parse_u32(tokens[3]),
                        power_cap: parse_u32(tokens[5]),
                        memory_used: parse_u32(tokens[7]),
                        memory_total: parse_u32(tokens[9]),
                        gpu_util: parse_float(tokens[11]),
                    });
                    gpu_id += 1;
                }
            }
        }
    }
    gpus
}

// ── Bar / color rendering helpers ─────────────────────────────────────────

fn format_bar(label: &str, value: f64, max: f64, color: Color, empty_color: Color) -> String {
    let percent = if max > 0.0 {
        (value / max) * 100.0
    } else {
        0.0
    };
    let width = 25;
    let filled = ((percent / 100.0) * width as f64).round() as usize;

    let mut s = format!("{:10}", label);
    for i in 0..width {
        if i < filled {
            s += &format!("{}", SetForegroundColor(color));
        } else {
            s += &format!("{}", SetForegroundColor(empty_color));
        }
        if i < filled {
            s.push('\u{2588}');
        } else {
            s.push('\u{2591}');
        }
    }
    s += "\x1b[0m";
    s
}

fn format_colored(color: Color, text: &str) -> String {
    let c = match color {
        Color::Red => "31",
        Color::Green => "32",
        Color::Yellow => "33",
        Color::Blue => "34",
        Color::Magenta => "35",
        Color::Cyan => "36",
        Color::White | Color::Grey => "37",
        Color::Black => "30",
        Color::DarkGrey => "90",
        _ => "37",
    };
    format!("\x1b[{}m{}\x1b[0m", c, text)
}

fn println_line(text: &str) {
    print!("{}", text);
    print!("\x1b[K\n");
}

// ── ANSI strip ────────────────────────────────────────────────────────────

fn strip_ansi_codes(text: &str) -> String {
    let mut result = String::new();
    let mut in_escape = false;

    for c in text.chars() {
        if c == '\x1b' {
            in_escape = true;
            continue;
        }
        if in_escape {
            if c == '[' {
                continue;
            }
            if in_escape && (c.is_ascii_digit() || c == ';' || c == 'H' || c == 'm' || c == 'J' || c == 'K') {
                if c == 'H' || c == 'm' || c == 'J' || c == 'K' {
                    in_escape = false;
                }
                continue;
            }
            in_escape = false;
        }
        result.push(c);
    }

    result
}

// ── Log reading helpers ────────────────────────────────────────────────────

fn get_last_lines(path: &str, n: usize) -> Vec<String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let lines: Vec<String> = content
        .lines()
        .map(|l| strip_ansi_codes(l))
        .collect();
    let start = if lines.len() > n { lines.len() - n } else { 0 };
    lines[start..].to_vec()
}

fn read_new_log_lines(path: &str, last_line_count: &mut usize) -> Vec<String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let all_lines: Vec<String> = content
        .lines()
        .map(|l| strip_ansi_codes(l))
        .collect();

    let total = all_lines.len();

    if total < *last_line_count {
        *last_line_count = 0;
    }

    let new_count = total - *last_line_count;

    if new_count == 0 {
        return Vec::new();
    }

    *last_line_count = total;
    all_lines.iter().skip(*last_line_count - new_count).cloned().collect()
}

// ── Slot parsing ───────────────────────────────────────────────────────────

fn get_slot_id(line: &str) -> Option<usize> {
    // Match patterns like: "[thread-id] slot print_timing: id 3" or "slot update_slots: id 3"
    if !line.contains("slot") {
        return None;
    }

    // Look for ": id N" or "= id N" after "slot"
    let slot_pos = line.find("slot")?;
    let rest = &line[slot_pos..];

    // Try ": id" first, then fallback to "id " pattern
    let id_start = rest.find(": id ").or_else(|| rest.find("id "))?;
    let after = &rest[id_start + 4..]; // skip "id "
    let trimmed = after.trim_start();
    let num_str: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();

    if num_str.is_empty() {
        return None;
    }
    num_str.parse::<usize>().ok()
}

// ── Inference stats parsing ────────────────────────────────────────────────

fn parse_inference_stats(line: &str) -> Option<InferenceStats> {
    if !line.contains("print_timing") {
        return None;
    }

    // Extract progress (match "progress = " specifically to avoid "processing")
    let progress = if line.contains("progress = ") {
        extract_float_after(line, "progress = ")
    } else {
        0.0
    };

    // Extract total_time from "total time = N ms" section
    let total_time = extract_float_after(line, "total time = ") / 1000.0;

    // Extract tokens_per_second from last "tokens per second" occurrence
    let mut tps: f64 = 0.0;
    if let Some(tps_pos) = line.rfind("tokens per second") {
        let before = &line[..tps_pos];
        if let Some(paren) = before.rfind('(') {
            let section = &line[paren + 1..tps_pos];
            let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
            tps = nums.parse().unwrap_or(0.0);
        }
    }

    // Extract latency (ms per token) from last "(X.XX ms per token)"
    let mut lat_tok: f64 = 0.0;
    if let Some(mpt_pos) = line.rfind("ms per token") {
        let before = &line[..mpt_pos];
        if let Some(paren) = before.rfind('(') {
            let section = &line[paren + 1..mpt_pos];
            let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
            lat_tok = nums.parse().unwrap_or(0.0);
        }
    }

    // Extract gen_speed from eval-time's "(N.N tokens per second)" specifically
    let mut gen_speed: f64 = 0.0;
    if let Some(eval_pos) = line.find("eval time") {
        if let Some(tps_pos) = line[eval_pos..].find("tokens per second") {
            let abs_tps_pos = eval_pos + tps_pos;
            let before = &line[..abs_tps_pos];
            if let Some(paren) = before.rfind('(') {
                let section = &line[paren + 1..abs_tps_pos];
                let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
                gen_speed = nums.parse().unwrap_or(0.0);
            }
        }
    }

    // Extract draft acceptance
    let draft_acceptance = if line.contains("draft acceptance = ") {
        extract_float_after(line, "draft acceptance = ")
    } else {
        0.0
    };

    // Extract n_decoded
    let n_decoded = if line.contains("n_decoded") {
        extract_number_after(line, "n_decoded =")
    } else if line.contains("stop processing") {
        extract_number_after(line, "n_tokens")
    } else {
        0
    };

    Some(InferenceStats {
        progress,
        time_seconds: total_time,
        tokens_per_second: tps,
        n_decoded,
        gen_speed_tps: if gen_speed > 0.0 { gen_speed } else { tps },
        latency_ms_tok: lat_tok,
        draft_acceptance,
        n_decoded_max: 0,
        ctx_n_tokens: 0,
        ctx_used: 0,
    })
}

fn parse_generation_stats(line: &str) -> Option<(u32, f64)> {
    // Extract gen_speed from multiple formats:
    // llama.cpp: "tg = 29.31 t/s" or "eval time = 100 ms / 5 tokens (27.07 tokens per second)"
    // forks:     "eval time = ... (20.5 tokens per second)"
    let mut gen_speed: f64 = 0.0;

    // Try "tg = N.XX t/s" (llama.cpp standard format)
    if line.contains("tg = ") {
        gen_speed = extract_float_after(line, "tg = ");
    }
    // Try eval-time's tokens per second
    if gen_speed == 0.0 && line.contains("eval time") && !line.contains("prompt eval time") {
        if let Some(eval_pos) = line.find("eval time") {
            if let Some(tps_pos) = line[eval_pos..].find("tokens per second") {
                let abs_tps_pos = eval_pos + tps_pos;
                let before = &line[..abs_tps_pos];
                if let Some(paren) = before.rfind('(') {
                    let section = &line[paren + 1..abs_tps_pos];
                    let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
                    gen_speed = nums.parse().unwrap_or(0.0);
                }
            }
        }
    }

    // Path 1: n_decoded = N
    if line.contains("n_decoded") {
        let n_decoded = extract_number_after(line, "n_decoded =");
        if n_decoded > 0 || gen_speed > 0.0 {
            return Some((n_decoded, gen_speed));
        }
    }

    // Path 2: stop processing with n_tokens
    if line.contains("stop processing") && line.contains("n_tokens") {
        let n_decoded = extract_number_after(line, "n_tokens");
        if n_decoded > 0 || gen_speed > 0.0 {
            return Some((n_decoded, gen_speed));
        }
    }

    // Path 3: eval time tokens count
    if line.contains("eval time") && !line.contains("prompt eval time") {
        let n_decoded = extract_number_after(line, "ms /");
        if n_decoded > 0 && gen_speed > 0.0 {
            return Some((n_decoded, gen_speed));
        }
    }

    None
}

fn parse_prompt_tps(line: &str) -> Option<f64> {
    // Extract from "prompt eval time = ... (N.NN ms per token, N.NN tokens per second)"
    if line.contains("prompt eval time") && line.contains("tokens per second") {
        if let Some(pe_pos) = line.find("prompt eval time") {
            if let Some(tps_pos) = line[pe_pos..].find("tokens per second") {
                let abs_tps_pos = pe_pos + tps_pos;
                let before = &line[..abs_tps_pos];
                // Extract the float immediately before "tokens per second"
                let mut digits: Vec<char> = Vec::new();
                let mut i = abs_tps_pos as isize - 1;
                let before_chars: Vec<char> = before.chars().collect();
                loop {
                    if i < 0 || i as usize >= before_chars.len() {
                        break;
                    }
                    let c = before_chars[i as usize];
                    if c.is_ascii_digit() || c == '.' {
                        digits.insert(0, c);
                        i -= 1;
                    } else if c.is_whitespace() {
                        i -= 1;
                    } else {
                        break;
                    }
                }
                let nums: String = digits.into_iter().collect();
                let tps = nums.parse::<f64>().unwrap_or(0.0);
                if tps > 0.0 {
                    return Some(tps);
                }
            }
        }
    }
    None
}

fn parse_prompt_processing_tps(line: &str) -> Option<f64> {
    // Extract from "prompt processing, n_tokens = N, progress = X, t = Y s / Z tokens per second"
    if line.contains("prompt processing") && line.contains("tokens per second") {
        if let Some(pp_pos) = line.find("prompt processing") {
            if let Some(tps_pos) = line[pp_pos..].find("tokens per second") {
                let abs_tps_pos = pp_pos + tps_pos;
                let before = &line[..abs_tps_pos];
                // Extract float immediately before "tokens per second"
                let mut digits: Vec<char> = Vec::new();
                let mut i = abs_tps_pos as isize - 1;
                let before_chars: Vec<char> = before.chars().collect();
                loop {
                    if i < 0 || i as usize >= before_chars.len() {
                        break;
                    }
                    let c = before_chars[i as usize];
                    if c.is_ascii_digit() || c == '.' {
                        digits.insert(0, c);
                        i -= 1;
                    } else if c.is_whitespace() {
                        i -= 1;
                    } else {
                        break;
                    }
                }
                let nums: String = digits.into_iter().collect();
                let tps = nums.parse::<f64>().unwrap_or(0.0);
                if tps > 0.0 {
                    return Some(tps);
                }
            }
        }
    }
    None
}

fn parse_latency(line: &str) -> Option<f64> {
    // Extract from eval time "(N.N ms per token)" only, not prompt eval time
    if line.contains("prompt eval time") || !line.contains("eval time") {
        return None;
    }
    if let Some(eval_pos) = line.find("eval time") {
        if let Some(mpt_pos) = line[eval_pos..].find("ms per token") {
            let abs_mpt_pos = eval_pos + mpt_pos;
            let before = &line[..abs_mpt_pos];
            if let Some(paren) = before.rfind('(') {
                let section = &line[paren + 1..abs_mpt_pos];
                let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
                let lat_tok = nums.parse::<f64>().unwrap_or(0.0);
                if lat_tok > 0.0 {
                    return Some(lat_tok);
                }
            }
        }
    }
    None
}

fn extract_number_after(line: &str, marker: &str) -> u32 {
    if let Some(pos) = line.find(marker) {
        let rest = &line[pos + marker.len()..];
        let mut start = None;
        let mut end = None;

        for (i, c) in rest.chars().enumerate() {
            if c.is_ascii_digit() {
                if start.is_none() {
                    start = Some(i);
                }
                end = Some(i + 1);
            } else if start.is_some() {
                break;
            }
        }

        if let (Some(s), Some(e)) = (start, end) {
            let nums: String = rest[s..e].chars().collect();
            nums.parse().unwrap_or(0)
        } else {
            0
        }
    } else {
        0
    }
}

fn extract_float_after(line: &str, marker: &str) -> f64 {
    if let Some(pos) = line.find(marker) {
        let rest = &line[pos + marker.len()..];
        let mut start = None;
        let mut end = None;
        let mut found_digit = false;

        for (i, c) in rest.chars().enumerate() {
            if c.is_ascii_digit() {
                if !found_digit {
                    start = Some(i);
                    found_digit = true;
                }
                end = Some(i + 1);
            } else if c == '.' && start.is_some() {
                end = Some(i + 1);
            } else if found_digit && !c.is_ascii_digit() && c != '.' {
                break;
            }
        }

        if let (Some(s), Some(e)) = (start, end) {
            let nums: String = rest[s..e].chars().collect();
            nums.parse().unwrap_or(0.0)
        } else {
            0.0
        }
    } else {
        0.0
    }
}

fn parse_ctx_usage(line: &str) -> Option<(u32, u32)> {
    // New format: "stop processing: n_tokens = N"
    if line.contains("stop processing") && line.contains("n_tokens") {
        let n_tokens = extract_number_after(line, "n_tokens");
        if n_tokens > 0 {
            return Some((n_tokens, 0));
        }
    }
    // New format: "created context checkpoint ... n_tokens = N"
    if line.contains("created context checkpoint") && line.contains("n_tokens") {
        let n_tokens = extract_number_after(line, "n_tokens");
        if n_tokens > 0 {
            return Some((n_tokens, 0));
        }
    }
    // Old format: "slot update_slots ... restored context checkpoint (pos_max = N, n_past = N)"
    if line.contains("slot update_slots") {
        let n_past = extract_number_after(line, "n_past =");
        let pos_max = extract_number_after(line, "pos_max =");
        if n_past > 0 {
            return Some((n_past, 0));
        } else if pos_max > 0 {
            return Some((pos_max + 1, 0));
        }
    }
    None
}

// ── Rendering ──────────────────────────────────────────────────────────────

fn render_inference_bars(stats: &InferenceStats, skip_progress: bool, config: &Config) -> Vec<String> {
    let bar_empty = parse_color_str(&config.bar_empty);
    let max_decoded = if stats.n_decoded_max > 0 { stats.n_decoded_max as f64 } else { 1000.0 };
    let ctx_percent = if stats.ctx_n_tokens > 0 {
        (stats.ctx_used as f64 / stats.ctx_n_tokens as f64) * 100.0
    } else {
        0.0
    };
    let bar_color = Color::DarkCyan;
    let mut bars: Vec<String> = Vec::new();
    if !skip_progress {
        bars.push(format!("{} {:.0}%",
            format_bar("Progress", stats.progress * 100.0, 100.0, bar_color, bar_empty),
            stats.progress * 100.0));
    }
    bars.push(format!("{} {}/s",
        format_bar("Prompt t/s", stats.tokens_per_second, 3000.0, bar_color, bar_empty),
        stats.tokens_per_second as u32));
    bars.push(format!("{} {}/s",
        format_bar("Gen t/s", stats.gen_speed_tps, 100.0, bar_color, bar_empty),
        stats.gen_speed_tps as u32));
    bars.push(format!("{} {}",
        format_bar("Decoded", stats.n_decoded as f64, max_decoded, bar_color, bar_empty),
        stats.n_decoded));
    bars.push(format!("{} {:.1}%",
        format_bar("Draft", stats.draft_acceptance * 100.0, 100.0, bar_color, bar_empty),
        stats.draft_acceptance * 100.0));
    bars.push(format!("{} {} / {} tokens",
        format_bar("Context", ctx_percent, 100.0, bar_color, bar_empty),
        stats.ctx_used, stats.ctx_n_tokens));
    bars.push(format!("{} {:.2}ms",
        format_bar("Latency", stats.latency_ms_tok, 5.0, bar_color, bar_empty),
        stats.latency_ms_tok));
    let mins = stats.time_seconds as u64 / 60;
    let secs = (stats.time_seconds as u64 % 60);
    bars.push(format!("{} {:02}:{:02}",
        format_colored(bar_color, "Time"), mins, secs));
    bars
}

fn render_log_window(lines: &[String], count: usize) -> Vec<String> {
    let total = lines.len();
    let start = if total > count { total - count } else { 0 };
    lines[start..].to_vec()
}

fn gpu_list_eq(a: &[GpuInfo], b: &[GpuInfo]) -> bool {
    if a.len() != b.len() { return false; }
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.temperature == y.temperature && x.fan_speed == y.fan_speed
            && x.power_usage == y.power_usage && x.power_cap == y.power_cap
            && x.memory_used == y.memory_used && x.memory_total == y.memory_total
            && x.gpu_util == y.gpu_util
    })
}

// ── Render: GPU section ────────────────────────────────────────────────────

fn render_gpu_section(gpus: &[GpuInfo], config: &Config, bar_empty: Color, y: &mut u16) {
    for gpu in gpus {
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_colored(Color::Magenta, &format!("GPU {}", gpu.id)));
        println_line("");
        *y += 1;

        let _ = execute!(io::stdout(), MoveTo(0, *y));
        println_line("");
        *y += 1;

        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_bar("Temp", gpu.temperature, 100.0, get_temp_color(gpu.temperature, config), bar_empty));
        print!(" {}\u{00B0}C\x1b[K", gpu.temperature as u32);
        *y += 1;

        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_bar("Fan", gpu.fan_speed, 100.0, Color::DarkCyan, bar_empty));
        print!(" {:.0}%\x1b[K", gpu.fan_speed);
        *y += 1;

        let power_color = parse_color_str(&config.power);
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        let power_pct = if gpu.power_cap > 0 {
            (gpu.power_usage as f64) / (gpu.power_cap as f64) * 100.0
        } else {
            0.0
        };
        print!("{}", format_bar("Power", power_pct, 100.0, power_color, bar_empty));
        print!(" {}/{}W\x1b[K", gpu.power_usage, gpu.power_cap);
        *y += 1;

        let power_left = gpu.power_cap.saturating_sub(gpu.power_usage);
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_bar("Pwr Left", power_left as f64, gpu.power_cap as f64, power_color, bar_empty));
        print!(" {}W\x1b[K", power_left);
        *y += 1;

        let mem_color = parse_color_str(&config.memory);
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        let mem_pct = if gpu.memory_total > 0 {
            (gpu.memory_used as f64) / (gpu.memory_total as f64) * 100.0
        } else {
            0.0
        };
        print!("{}", format_bar("Memory", mem_pct, 100.0, mem_color, bar_empty));
        print!(" {}/{}MiB\x1b[K", gpu.memory_used, gpu.memory_total);
        *y += 1;

        let mem_free = gpu.memory_total.saturating_sub(gpu.memory_used);
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_bar("Mem Free", mem_free as f64, gpu.memory_total as f64, mem_color, bar_empty));
        print!(" {}MiB\x1b[K", mem_free);
        *y += 1;

        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_bar("Util", gpu.gpu_util, 100.0, get_util_color(gpu.gpu_util, config), bar_empty));
        print!(" {}%\x1b[K", gpu.gpu_util as u32);
        *y += 1;

        if gpu.id + 1 < gpus.len() {
            let _ = execute!(io::stdout(), MoveTo(0, *y));
            println_line("");
            *y += 1;
            let _ = execute!(io::stdout(), MoveTo(0, *y));
            println_line("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
            *y += 1;
        }
    }
}

// ── Render: header (title + model info) ─────────────────────────────────────

fn render_header(llama_info: &Option<LlamaServerInfo>, embedding_models: &Option<Vec<String>>, config: &Config, y: &mut u16) {
    let _ = execute!(io::stdout(), MoveTo(0, *y));
    print!("{}", format_colored(parse_color_str(&config.title), "NV-SMI"));
    println_line("");
    *y += 1;

    if let Some(ref info) = llama_info {
        let model_name = info.model.rsplit('/').next().unwrap_or("unknown");
        let tags = [
            (info.multimodal, Color::Cyan, "[VISION]"),
            (info.embedding, Color::Magenta, "[EMBEDDING]"),
        ]
        .iter()
        .filter_map(|(show, color, text)| if *show { Some(format_colored(*color, text)) } else { None })
        .collect::<Vec<_>>()
        .join(" ");
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_colored(Color::Yellow, &format!("Model: {}", model_name)));
        if !tags.is_empty() {
            print!(" {}", tags);
        }
        println_line("");
        *y += 1;

        if !info.params.is_empty() {
            for (key, val) in &info.params {
                let _ = execute!(io::stdout(), MoveTo(0, *y));
                print!("{}", format_colored(Color::White, &format!("{}={}", key, val)));
                println_line("");
                *y += 1;
            }
        }

        if let Some(models) = embedding_models {
            let _ = execute!(io::stdout(), MoveTo(0, *y));
            print!("{}", format_colored(Color::Magenta, "Embedding model(s):"));
            println_line("");
            *y += 1;
            for model in models {
                let model_name = model.rsplit('/').next().unwrap_or("unknown");
                let _ = execute!(io::stdout(), MoveTo(0, *y));
                print!("  {}", format_colored(Color::White, model_name));
                println_line("");
                *y += 1;
            }
        }

        let _ = execute!(io::stdout(), MoveTo(0, *y));
        println_line("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
        *y += 1;
    } else {
        let _ = execute!(io::stdout(), MoveTo(0, *y));
        print!("{}", format_colored(Color::White, "(no llama-server running)"));
        println_line("");
        *y += 1;

        let _ = execute!(io::stdout(), MoveTo(0, *y));
        println_line("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
        *y += 1;
    }

    let _ = execute!(io::stdout(), MoveTo(0, *y));
    println_line("");
    *y += 1;
}

// ── Render: slot bars ───────────────────────────────────────────────────────

fn render_slot_bars(
    all_slot_stats: &[InferenceStats],
    all_idle: bool,
    config: &Config,
    y: &mut u16,
) {
    let mut bar_y = *y + 1;

    for (slot_id, slot_stats) in all_slot_stats.iter().enumerate() {
        let _ = execute!(io::stdout(), MoveTo(0, bar_y));
        if all_idle {
            print!("{}", format_colored(Color::Grey, &format!("SLOT {} BARS  \u{2014}  IDLE", slot_id)));
        } else {
            print!("{}", format_colored(Color::Yellow, &format!("SLOT {} BARS", slot_id)));
        }
        println_line("");
        bar_y += 1;

        let bar_lines = render_inference_bars(slot_stats, all_idle, config);
        for (i, line) in bar_lines.iter().enumerate() {
            let row = bar_y + (i * 2) as u16;
            let _ = execute!(io::stdout(), MoveTo(0, row));
            print!("{}\x1b[K", line);
        }
        bar_y += (bar_lines.len() * 2) as u16;

        let _ = execute!(io::stdout(), MoveTo(0, bar_y));
        println_line("");
        bar_y += 1;
    }

    *y = bar_y;
}

// ── Main loop ───────────────────────────────────────────────────────────────

fn main() {
    let config = Config::load();
    let _guard = TerminalGuard;
    let _ = enable_raw_mode();

    let mut prev_gpus: Vec<GpuInfo> = Vec::new();
    let mut prev_llama: Option<String> = None;
    let mut prev_log_lines: Vec<String> = Vec::new();
    let mut prev_height: u16 = 0;
    let mut persist_progress: f64 = 0.0;
    let mut persist_gen_speed: f64 = 0.0;
    let mut persist_n_decoded: u32 = 0;
    let mut persist_draft_acceptance: f64 = 0.0;
    let mut persist_prompt_tps: f64 = 0.0;
    let mut persist_time_seconds: f64 = 0.0;
    let mut persist_latency: f64 = 0.0;
    let mut prev_n_parallel: usize = 0;
    let mut log_line_count: usize = 0;

    if let Some(ref log_file) = config.log_file {
        if let Ok(content) = fs::read_to_string(log_file) {
            log_line_count = content.lines().count();
        }
    }

    let mut slot_n_decoded: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut slot_gen_speed: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut slot_draft: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut slot_progress: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut slot_ctx_used: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut slot_prompt_tps: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut slot_time_seconds: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut slot_latency: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

    loop {
        let gpus = get_nvidia_smi();

        let llama_info = get_llama_server_info(config.log_file.as_deref());
        let embedding_models = get_embedding_models(config.log_file.as_deref());
        let llama_key = llama_info.as_ref().map(|i| i.model.clone());
        let log_changed = if let Some(ref log_file) = config.log_file {
            get_last_lines(log_file, config.log_lines) != prev_log_lines
        } else {
            false
        };

        let changed = !gpu_list_eq(&prev_gpus, &gpus)
            || prev_llama.as_ref() != llama_key.as_ref()
            || log_changed;

        if !changed {
            prev_gpus = gpus;
            prev_llama = llama_key;
            if let Some(ref log_file) = config.log_file {
                prev_log_lines = get_last_lines(log_file, config.log_lines);
            }

            if crossterm::event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(KeyEvent { code, .. })) = crossterm::event::read() {
                    if code == KeyCode::Char('q') || code == KeyCode::Esc {
                        return;
                    }
                }
            }
            thread::sleep(Duration::from_secs(2));
            continue;
        }

        prev_gpus = gpus.clone();
        prev_llama = llama_key;
        if let Some(ref log_file) = config.log_file {
            prev_log_lines = get_last_lines(log_file, config.log_lines);
        }

        let _ = execute!(io::stdout(), Hide);

        let mut y: u16 = 0;

        render_header(&llama_info, &embedding_models, &config, &mut y);

        let bar_empty = parse_color_str(&config.bar_empty);

        render_gpu_section(&gpus, &config, bar_empty, &mut y);

        // ── Log + slots + raw log section ───────────────────────────────────

        if let Some(ref log_file) = config.log_file {
            let _ = execute!(io::stdout(), MoveTo(0, y));
            println_line("");
            y += 1;

            let _ = execute!(io::stdout(), MoveTo(0, y));
            print!("{}", format_colored(parse_color_str(&config.title), "LOG"));
            println_line("");
            y += 1;

            let _ = execute!(io::stdout(), MoveTo(0, y));
            println_line("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
            y += 1;

            let new_log_lines = read_new_log_lines(log_file, &mut log_line_count);
            let log_lines_data = get_last_lines(log_file, config.log_lines * 10);

            let mut all_idle = false;
            let mut slot_map: std::collections::HashMap<usize, Vec<&str>> = std::collections::HashMap::new();
            for line in &new_log_lines {
                if line.contains("all slots are idle") {
                    all_idle = true;
                }
                if let Some(slot_id) = get_slot_id(line) {
                    slot_map.entry(slot_id).or_insert_with(Vec::new).push(line.as_str());
                }
            }

            if all_idle {
                slot_map.clear();
                slot_n_decoded.clear();
                slot_gen_speed.clear();
                slot_draft.clear();
                slot_progress.clear();
                slot_ctx_used.clear();
                slot_prompt_tps.clear();
                slot_time_seconds.clear();
                slot_latency.clear();
                persist_progress = 0.0;
                persist_gen_speed = 0.0;
                persist_n_decoded = 0;
                persist_draft_acceptance = 0.0;
                persist_prompt_tps = 0.0;
                persist_time_seconds = 0.0;
                persist_latency = 0.0;
            }

            let mut max_decoded_from_config: u32 = 0;
            let mut n_parallel: u32 = 4;
            if let Some(ref info) = llama_info {
                if info.context_len > 0 && info.n_parallel > 0 {
                    max_decoded_from_config = info.context_len / info.n_parallel;
                    n_parallel = info.n_parallel;
                }
            }

            if n_parallel as usize != prev_n_parallel {
                slot_n_decoded.clear();
                slot_gen_speed.clear();
                slot_draft.clear();
                slot_progress.clear();
                slot_ctx_used.clear();
                slot_prompt_tps.clear();
                slot_time_seconds.clear();
                slot_latency.clear();
                prev_n_parallel = n_parallel as usize;
            }

            let mut all_slot_stats: Vec<InferenceStats> = Vec::new();

            for slot_id in 0..n_parallel as usize {
                let lines: Vec<&str> = slot_map.get(&slot_id).map(|v| v.as_slice()).unwrap_or(&[]).to_vec();
                let mut s: Option<InferenceStats> = None;
                let mut n_decoded: u32 = 0;
                let mut gen_speed: f64 = 0.0;
                let mut draft: f64 = 0.0;
                let mut progress: f64 = 0.0;
                let mut latency: f64 = 0.0;
                let mut prompt_tps: f64 = 0.0;
                let mut time_seconds: f64 = 0.0;

                for line in &lines {
                    if let Some(stats) = parse_inference_stats(line) {
                        progress = stats.progress;
                        draft = stats.draft_acceptance;
                        time_seconds = stats.time_seconds;
                        s = Some(stats);
                    }
                    if let Some((nd, gs)) = parse_generation_stats(line) {
                        n_decoded = nd;
                        gen_speed = gs;
                    }
                    if let Some(lat) = parse_latency(line) {
                        latency = lat;
                    }
                    if let Some(tps) = parse_prompt_tps(line) {
                        prompt_tps = tps;
                    }
                    if let Some(tps) = parse_prompt_processing_tps(line) {
                        prompt_tps = tps;
                    }
                    // Reset prompt_tps when task ends or new one starts
                    if line.contains("stop processing") || line.contains("launch_slot_") {
                        slot_prompt_tps.remove(&slot_id);
                    }
                    if let Some((ctx, _)) = parse_ctx_usage(line) {
                        if ctx > 0 {
                            slot_ctx_used.insert(slot_id, ctx);
                        }
                    }
                }

                if n_decoded > 0 { slot_n_decoded.insert(slot_id, n_decoded); }
                if gen_speed > 0.0 { slot_gen_speed.insert(slot_id, gen_speed); }
                if draft > 0.0 { slot_draft.insert(slot_id, draft); }
                if progress > 0.0 { slot_progress.insert(slot_id, progress); }
                if prompt_tps > 0.0 { slot_prompt_tps.insert(slot_id, prompt_tps); }
                if time_seconds > 0.0 { slot_time_seconds.insert(slot_id, time_seconds); }
                if latency > 0.0 { slot_latency.insert(slot_id, latency); }

                let mut stats_to_render = s.unwrap_or_default();
                stats_to_render.tokens_per_second = *slot_prompt_tps.get(&slot_id).unwrap_or(&0.0);
                stats_to_render.n_decoded = *slot_n_decoded.get(&slot_id).unwrap_or(&0);
                stats_to_render.gen_speed_tps = *slot_gen_speed.get(&slot_id).unwrap_or(&0.0);
                stats_to_render.latency_ms_tok = *slot_latency.get(&slot_id).unwrap_or(&0.0);
                stats_to_render.time_seconds = *slot_time_seconds.get(&slot_id).unwrap_or(&persist_time_seconds);
                stats_to_render.draft_acceptance = *slot_draft.get(&slot_id).unwrap_or(&0.0);
                stats_to_render.progress = *slot_progress.get(&slot_id).unwrap_or(&persist_progress);
                stats_to_render.n_decoded_max = max_decoded_from_config;
                stats_to_render.ctx_n_tokens = max_decoded_from_config;
                stats_to_render.ctx_used = *slot_ctx_used.get(&slot_id).unwrap_or(&0);
                all_slot_stats.push(stats_to_render);
            }

            persist_progress = if !slot_progress.is_empty() { slot_progress.values().cloned().fold(0.0f64, f64::max) } else { persist_progress };
            persist_gen_speed = if !slot_gen_speed.is_empty() { slot_gen_speed.values().cloned().fold(0.0f64, f64::max) } else { persist_gen_speed };
            persist_n_decoded = if !slot_n_decoded.is_empty() { *slot_n_decoded.values().max_by(|a, b| a.cmp(b)).unwrap() } else { persist_n_decoded };
            persist_draft_acceptance = if !slot_draft.is_empty() { slot_draft.values().cloned().fold(0.0f64, f64::max) } else { persist_draft_acceptance };
            persist_prompt_tps = if !slot_prompt_tps.is_empty() { slot_prompt_tps.values().cloned().fold(0.0f64, f64::max) } else { persist_prompt_tps };
            persist_time_seconds = if !slot_time_seconds.is_empty() { slot_time_seconds.values().cloned().fold(0.0f64, f64::max) } else { persist_time_seconds };
            persist_latency = if !slot_latency.is_empty() { slot_latency.values().cloned().fold(0.0f64, f64::max) } else { persist_latency };

            if all_slot_stats.is_empty() {
                let stats_to_render = InferenceStats {
                    tokens_per_second: persist_prompt_tps,
                    progress: persist_progress,
                    gen_speed_tps: persist_gen_speed,
                    n_decoded: persist_n_decoded,
                    draft_acceptance: persist_draft_acceptance,
                    time_seconds: persist_time_seconds,
                    latency_ms_tok: persist_latency,
                    n_decoded_max: max_decoded_from_config,
                    ctx_n_tokens: max_decoded_from_config,
                    ..Default::default()
                };

                let bar_lines = render_inference_bars(&stats_to_render, true, &config);
                for (i, line) in bar_lines.iter().enumerate() {
                    let row = y + 1 + (i * 2) as u16;
                    let _ = execute!(io::stdout(), MoveTo(0, row));
                    print!("{}\x1b[K", line);
                }
                y += 1 + (bar_lines.len() * 2) as u16;
            } else {
                render_slot_bars(&all_slot_stats, all_idle, &config, &mut y);
            }

            let _ = execute!(io::stdout(), MoveTo(0, y));
            println_line("");
            y += 1;

            let _ = execute!(io::stdout(), MoveTo(0, y));
            print!("{}", format_colored(parse_color_str(&config.title), "RAW LOG"));
            println_line("");
            y += 1;

            let _ = execute!(io::stdout(), MoveTo(0, y));
            println_line("\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
            y += 1;

            let log_display = render_log_window(&log_lines_data, config.log_lines);
            for (i, line) in log_display.iter().enumerate() {
                let _ = execute!(io::stdout(), MoveTo(0, y + i as u16));
                print!("{}\x1b[K", line);
            }
            y += log_display.len() as u16;
        }

        // ── Clear leftover lines ─────────────────────────────────────────────

        if let Ok((_term_w, term_h)) = size() {
            let term_h = term_h as u16;
            if y > term_h {
                y = term_h;
            }
            for cy in y..term_h {
                let _ = execute!(io::stdout(), MoveTo(0, cy));
                let _ = write!(io::stdout(), "{:120}\x1b[K", " ");
                let _ = io::stdout().flush();
            }
        } else {
            for cy in y..prev_height {
                let _ = execute!(io::stdout(), MoveTo(0, cy));
                print!("{:120}\x1b[K", " ");
            }
        }

        prev_height = y;

        let _ = execute!(io::stdout(), Show);
        let _ = io::stdout().flush();

        if crossterm::event::poll(Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(Event::Key(KeyEvent { code, .. })) = crossterm::event::read() {
                if code == KeyCode::Char('q') || code == KeyCode::Esc {
                    break;
                }
            }
        }

        thread::sleep(Duration::from_secs(2));
    }

    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), Show);
    print!("\x1b[?25h");
    print!("\x1b[2J\x1b[H");
}