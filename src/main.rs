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
use crossterm::terminal::{enable_raw_mode, disable_raw_mode};
use yaml_rust::YamlLoader;

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

#[derive(Debug)]
struct InferenceStats {
    progress: f64,
    time_seconds: f64,
    tokens_per_second: f64,
    n_decoded: u32,
    gen_speed_tps: f64,
    latency_ms_tok: f64,
    draft_acceptance: f64,
    n_decoded_max: u32,
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

impl Config {
    fn load() -> Self {
        let config_path = env::var("HOME")
            .map(|h| format!("{}/.config/nv-smi/config.yaml", h))
            .unwrap_or_default();

        if fs::metadata(&config_path).is_ok() {
            let content = fs::read_to_string(&config_path).unwrap_or_default();
            let docs = YamlLoader::load_from_str(&content).unwrap_or_default();

            Config {
                temp_low: docs
                    .get(0)
                    .and_then(|d| d["temp_low"].as_str())
                    .unwrap_or("Cyan")
                    .to_string(),
                temp_medium: docs
                    .get(0)
                    .and_then(|d| d["temp_medium"].as_str())
                    .unwrap_or("Green")
                    .to_string(),
                temp_high: docs
                    .get(0)
                    .and_then(|d| d["temp_high"].as_str())
                    .unwrap_or("Yellow")
                    .to_string(),
                temp_critical: docs
                    .get(0)
                    .and_then(|d| d["temp_critical"].as_str())
                    .unwrap_or("Red")
                    .to_string(),
                power: docs
                    .get(0)
                    .and_then(|d| d["power"].as_str())
                    .unwrap_or("Green")
                    .to_string(),
                memory: docs
                    .get(0)
                    .and_then(|d| d["memory"].as_str())
                    .unwrap_or("Cyan")
                    .to_string(),
                util_low: docs
                    .get(0)
                    .and_then(|d| d["util_low"].as_str())
                    .unwrap_or("Green")
                    .to_string(),
                util_medium: docs
                    .get(0)
                    .and_then(|d| d["util_medium"].as_str())
                    .unwrap_or("Yellow")
                    .to_string(),
                util_high: docs
                    .get(0)
                    .and_then(|d| d["util_high"].as_str())
                    .unwrap_or("Red")
                    .to_string(),
                title: docs
                    .get(0)
                    .and_then(|d| d["title"].as_str())
                    .unwrap_or("Cyan")
                    .to_string(),
                bar_empty: docs
                    .get(0)
                    .and_then(|d| d["bar_empty"].as_str())
                    .unwrap_or("DarkGrey")
                    .to_string(),
                log_file: docs
                    .get(0)
                    .and_then(|d| d["log_file"].as_str())
                    .map(|s| s.to_string()),
           log_lines: docs
                .get(0)
                .and_then(|d| d["log_lines"].as_i64())
                .map(|v| v as usize)
                .unwrap_or(10),
            log_height: docs
                .get(0)
                .and_then(|d| d["log_height"].as_i64())
                .map(|v| v as usize)
                .unwrap_or(10),
            }
        } else {
            Config::default()
        }
    }
}

fn get_nvidia_smi() -> String {
    let output = Command::new("nvidia-smi").output().unwrap();
    String::from_utf8_lossy(&output.stdout).to_string()
}

struct LlamaServerInfo {
    model: String,
    params: Vec<(String, String)>,
    context_len: u32,
    n_parallel: u32,
}

fn get_llama_server_info() -> Option<LlamaServerInfo> {
    let output = Command::new("ps")
        .arg("aux")
        .output()
        .ok()?;
    let ps_text = String::from_utf8_lossy(&output.stdout);

    for line in ps_text.lines() {
        if !line.contains("llama-server") || line.contains("grep") {
            continue;
        }

        let args: Vec<&str> = line.split_whitespace().collect();
        let mut model = String::new();
        let mut params: Vec<(String, String)> = Vec::new();

        // Only parse from ./llama-server onward (skip ps aux columns)
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
                    params.push(("log-file".to_string(), val));
                    i += 2;
                }
                _ => {
                    // Skip unrecognized flags
                    i += 1;
                }
            }
        }

        if !model.is_empty() {
            return Some(LlamaServerInfo { model, params, context_len, n_parallel });
        }
    }

    None
}

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
            s.push('█');
        } else {
            s.push('░');
        }
    }
    // Reset color so subsequent text (numbers, etc.) isn't colored DarkGrey.
    s += "\x1b[0m";
    s
}

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

fn get_slot_id(line: &str) -> Option<usize> {
    if !line.contains(" slot ") {
        return None;
    }
    let slot_start = line.find(" slot ").unwrap();
    let rest = &line[slot_start + 6..];
    if !rest.starts_with("print_timing: id") {
        return None;
    }
    let id_start = rest.find("id").unwrap() + 2;
    let after_id = &rest[id_start..];
    let trimmed = after_id.trim_start();
    let num_str: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_str.parse::<usize>().ok()
}

fn parse_inference_stats(line: &str) -> Option<InferenceStats> {
    if !line.contains("print_timing") {
        return None;
    }

    // Extract what's available from this line (not all fields present in every line).
    let progress = extract_float_after(line, "progress");
    let time_ms = extract_float_after(line, "ms /");

    // Try to get t/s from the "(X.XX tokens per second)" part.
    let mut tps: f64 = 0.0;
    if let Some(pos) = line.rfind("tokens per second") {
        let start = line.rfind('/').unwrap_or(pos - 20);
        let section = &line[start + 1..pos];
        let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
        tps = nums.parse().unwrap_or(0.0);
    }

    // Try to get latency from "(X.XX ms per token)".
    let mut lat_tok: f64 = 0.0;
    if let Some(pos) = line.find("ms per token") {
        let start = line.rfind('(').unwrap_or(pos - 15);
        let section = &line[start + 1..pos];
        let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
        lat_tok = nums.parse().unwrap_or(0.0);
    }

    // Try to get gen speed from "(X.XX tokens per second)" — this is eval time line.
    let mut gen_speed: f64 = 0.0;
    if line.contains("eval time") && !line.contains("prompt eval time") {
        if let Some(pos) = line.rfind("tokens per second") {
            // Find the last '(' before this position for "(X.XX tokens per second)".
            let mut last_start: usize = 0;
            for (i, ch) in line[..pos].char_indices() {
                if ch == '(' {
                    last_start = i;
                }
            }
            if last_start > 0 && last_start < pos {
                let section = &line[last_start + 1..pos];
                let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
                gen_speed = nums.parse().unwrap_or(0.0);
            }
        }
    }

    // Total time from "total time = X ms".
    let mut total_time: f64 = 0.0;
    if line.contains("total time") {
        total_time = extract_float_after(line, "= ") / 1000.0; // ms -> s
    }

    let mut draft_acceptance: f64 = 0.0;
    if line.contains("draft acceptance") {
        draft_acceptance = extract_float_after(line, "draft acceptance = ");
    }

    Some(InferenceStats {
        progress,
        time_seconds: if total_time > 0.0 { total_time } else { time_ms },
        tokens_per_second: tps,
        n_decoded: if line.contains("stop processing") || line.contains("n_decoded") {
            extract_number_after(line, "n_tokens")
        } else { 0 },
        gen_speed_tps: if gen_speed > 0.0 { gen_speed } else { tps },
        latency_ms_tok: lat_tok,
        draft_acceptance,
        n_decoded_max: 0,
    })
}

fn parse_generation_stats(line: &str) -> Option<(u32, f64)> {
    // Also check for "stop processing" lines with final token count.
    if line.contains("n_decoded") || (line.contains("stop processing") && line.contains("n_tokens")) {
        let n_decoded = extract_number_after(line, "n_tokens");
        let gen_speed = extract_float_after(line, "tg =");
        if n_decoded > 0 || gen_speed > 0.0 {
            return Some((n_decoded, gen_speed));
        }
    }

    // Legacy: look for n_decoded in eval-time lines.
    if line.contains("eval time") && !line.contains("prompt eval time") {
        let n_decoded = extract_number_after(line, "tokens");
        let gen_speed = extract_float_after(line, "tokens per second)");
        if n_decoded > 0 && gen_speed > 0.0 {
            return Some((n_decoded, gen_speed));
        }
    }

    None
}

fn parse_latency(line: &str) -> Option<f64> {
    // Also extract latency from "(X.XX ms per token)" in eval/prompt lines.
    if line.contains("ms per token") && !line.contains("verify ubatch") {
        let lat = extract_float_after(line, "ms per token");
        if lat > 0.0 { return Some(lat); }
    }

    // Legacy: verify ubatch latency.
    if !line.contains("verify ubatch") {
        return None;
    }

    let lat = extract_float_after(line, "(");
    if lat > 0.0 {
        Some(lat)
    } else {
        None
    }
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

fn render_inference_bars(stats: &InferenceStats) -> Vec<String> {
    let bar_empty = parse_color_str(&Config::default().bar_empty);
    let max_decoded = if stats.n_decoded_max > 0 { stats.n_decoded_max as f64 } else { 1000.0 };
    vec![
        format!("{} {:.0}%",
            format_bar("Progress", stats.progress * 100.0, 100.0, Color::Green, bar_empty),
            stats.progress * 100.0),
        format!("{} {}/s",
            format_bar("Prompt t/s", stats.tokens_per_second, 1000.0, Color::Cyan, bar_empty),
            stats.tokens_per_second as u32),
        format!("{} {}/s",
            format_bar("Gen t/s", stats.gen_speed_tps, 100.0, Color::Green, bar_empty),
            stats.gen_speed_tps as u32),
        format!("{} {}",
            format_bar("Decoded", stats.n_decoded as f64, max_decoded, Color::Magenta, bar_empty),
            stats.n_decoded),
        format!("{} {:.1}%",
            format_bar("Draft", stats.draft_acceptance * 100.0, 100.0, Color::Magenta, bar_empty),
            stats.draft_acceptance * 100.0),
        format!("{} {:.2}ms",
            format_bar("Latency", stats.latency_ms_tok, 5.0, Color::Yellow, bar_empty),
            stats.latency_ms_tok),
        format!("{} {:.2}s",
            format_bar("Time", stats.time_seconds, 300.0, Color::Yellow, bar_empty),
            stats.time_seconds),
    ]
}

fn render_log_window(lines: &[String], height: usize) -> Vec<String> {
    lines.iter().take(height).cloned().collect()
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
    // Clear rest of line to avoid leftover characters from longer previous content.
    print!("\x1b[K\n");
}

fn main() {
    let config = Config::load();

    let _ = enable_raw_mode();

    let mut prev_gpus: Vec<GpuInfo> = Vec::new();
    let mut prev_llama: Option<String> = None;
    let mut prev_log_lines: Vec<String> = Vec::new();
    let mut prev_height: u16 = 0;
   let mut persist_progress: f64 = 0.0;
    let mut persist_gen_speed: f64 = 0.0;
    let mut persist_n_decoded: u32 = 0;
    let mut persist_draft_acceptance: f64 = 0.0;

    loop {
        let output = get_nvidia_smi();
        let gpus = parse_gpus(&output);

        // Only redraw if something actually changed
        let llama_info = get_llama_server_info();
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
            // Data unchanged — skip redraw entirely (no flicker).
            // Just check for input and sleep.
            prev_gpus = gpus;
            prev_llama = llama_key;
            if let Some(ref log_file) = config.log_file {
                prev_log_lines = get_last_lines(log_file, config.log_lines);
            }

            if crossterm::event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(KeyEvent { code, .. })) = crossterm::event::read() {
                    if code == KeyCode::Char('q') || code == KeyCode::Esc {
                        let _ = disable_raw_mode();
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

        // Hide cursor before rendering to avoid visible flicker.
        execute!(io::stdout(), Hide).unwrap();

        let mut y: u16 = 0;

        // Title
        execute!(io::stdout(), MoveTo(0, y)).unwrap();
        print!("{}", format_colored(parse_color_str(&config.title), "NV-SMI"));
        println_line("");
        y += 1;

        if let Some(ref info) = llama_info {
            let model_name = info.model.rsplit('/').next().unwrap_or("unknown");
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(Color::Yellow, &format!("Model: {}", model_name)));
            println_line("");
            y += 1;

            if !info.params.is_empty() {
                let col_w = 14usize;
                let cols = 3usize;
                for chunk in info.params.chunks(cols) {
                    execute!(io::stdout(), MoveTo(0, y)).unwrap();
                    for (key, val) in chunk {
                        print!("{:width$} ", format!("{}={}", key, val), width = col_w);
                    }
                    println_line("");
                    y += 1;
                }
            }

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("─────────────────────────────────────");
            y += 1;
        } else {
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(Color::White, "(no llama-server running)"));
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("─────────────────────────────────────");
            y += 1;
        }

        execute!(io::stdout(), MoveTo(0, y)).unwrap();
        println_line("");
        y += 1;

        let bar_empty = parse_color_str(&config.bar_empty);
  

        for gpu in &gpus {
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(Color::Magenta, &format!("GPU {}", gpu.id)));
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Temp", gpu.temperature, 100.0, get_temp_color(gpu.temperature, &config), bar_empty));
            println!(" {}°C\x1b[K", gpu.temperature as u32);
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Fan", gpu.fan_speed, 100.0, Color::White, bar_empty));
            println!(" {:.0}%\x1b[K", gpu.fan_speed);
            y += 1;

            let power_color = parse_color_str(&config.power);
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Power", (gpu.power_usage as f64) / (gpu.power_cap as f64) * 100.0, 100.0, power_color, bar_empty));
            println!(" {}/{}W\x1b[K", gpu.power_usage, gpu.power_cap);
            y += 1;

            let power_left = if gpu.power_cap > gpu.power_usage { gpu.power_cap - gpu.power_usage } else { 0 };
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Pwr Left", power_left as f64, gpu.power_cap as f64, power_color, bar_empty));
            println!(" {}W\x1b[K", power_left);
            y += 1;

            let mem_color = parse_color_str(&config.memory);
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Memory", (gpu.memory_used as f64) / (gpu.memory_total as f64) * 100.0, 100.0, mem_color, bar_empty));
            println!(" {}/{}MiB\x1b[K", gpu.memory_used, gpu.memory_total);
            y += 1;

            let mem_free = if gpu.memory_total > gpu.memory_used { gpu.memory_total - gpu.memory_used } else { 0 };
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Mem Free", mem_free as f64, gpu.memory_total as f64, mem_color, bar_empty));
            println!(" {}MiB\x1b[K", mem_free);
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Util", gpu.gpu_util, 100.0, get_util_color(gpu.gpu_util, &config), bar_empty));
            println!(" {}%\x1b[K", gpu.gpu_util as u32);
            y += 1;

            if gpu.id + 1 < gpus.len() {
                execute!(io::stdout(), MoveTo(0, y)).unwrap();
                println_line("");
                y += 1;
                execute!(io::stdout(), MoveTo(0, y)).unwrap();
                println_line("─────────────────────────────────────");
                y += 1;
            }
        }

       // Render log window if configured
        if let Some(ref log_file) = config.log_file {
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(parse_color_str(&config.title), "LOG"));
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("═══════════════════════════════════════════");
            y += 1;

            let log_lines_data = get_last_lines(log_file, config.log_lines * 10);

            let mut slot_map: std::collections::HashMap<usize, Vec<&str>> = std::collections::HashMap::new();
            let mut slot_order: Vec<usize> = Vec::new();
            let mut all_non_slot_lines: Vec<String> = Vec::new();

            for line in &log_lines_data {
                if let Some(slot_id) = get_slot_id(line) {
                    if !slot_map.contains_key(&slot_id) {
                        slot_map.insert(slot_id, Vec::new());
                        slot_order.push(slot_id);
                    }
                    slot_map.get_mut(&slot_id).unwrap().push(line.as_str());
                } else {
                    all_non_slot_lines.push(line.clone());
                }
            }

            let mut max_decoded_from_config: u32 = 0;
            if let Some(ref info) = llama_info {
                if info.context_len > 0 && info.n_parallel > 0 {
                    max_decoded_from_config = info.context_len / info.n_parallel;
                }
            }

            let mut all_slot_stats: Vec<InferenceStats> = Vec::new();

            for slot_id in &slot_order {
                let lines: Vec<&str> = slot_map.get(slot_id).unwrap().clone();
                let mut s: Option<InferenceStats> = None;
                let mut n_decoded: u32 = 0;
                let mut gen_speed: f64 = 0.0;
                let mut latency: f64 = 0.0;

                for line in &lines {
                    if let Some(stats) = parse_inference_stats(line) {
                        s = Some(stats);
                    }
                    if let Some((nd, gs)) = parse_generation_stats(line) {
                        n_decoded = nd;
                        gen_speed = gs;
                    }
                    if let Some(lat) = parse_latency(line) {
                        latency = lat;
                    }
                }

                let mut stats_to_render = s.unwrap_or(InferenceStats {
                    progress: 0.0,
                    time_seconds: 0.0,
                    tokens_per_second: 0.0,
                    n_decoded: 0,
                    gen_speed_tps: 0.0,
                    latency_ms_tok: 0.0,
                    draft_acceptance: 0.0,
                    n_decoded_max: 0,
                });
                stats_to_render.n_decoded = n_decoded;
                stats_to_render.gen_speed_tps = gen_speed;
                stats_to_render.latency_ms_tok = latency;
                stats_to_render.n_decoded_max = max_decoded_from_config;
                all_slot_stats.push(stats_to_render);
            }

            let mut bar_y = y + 1;
            let mut slot_idx = 0;

            for slot_id in &slot_order {
                let slot_stats = &all_slot_stats[slot_idx];
                execute!(io::stdout(), MoveTo(0, bar_y)).unwrap();
                print!("{}", format_colored(Color::Yellow, &format!("SLOT {} BARS", slot_id)));
                println_line("");
                bar_y += 1;

                let bar_lines = render_inference_bars(slot_stats);
                for (i, line) in bar_lines.iter().enumerate() {
                    execute!(io::stdout(), MoveTo(0, bar_y + i as u16)).unwrap();
                    println!("{}\x1b[K", line);
                }
                bar_y += bar_lines.len() as u16;

                execute!(io::stdout(), MoveTo(0, bar_y)).unwrap();
                println_line("");
                bar_y += 1;
                slot_idx += 1;
            }

            if all_slot_stats.is_empty() {
                let mut stats_to_render = InferenceStats {
                    progress: persist_progress,
                    time_seconds: 0.0,
                    tokens_per_second: 0.0,
                    n_decoded: persist_n_decoded,
                    gen_speed_tps: persist_gen_speed,
                    latency_ms_tok: 0.0,
                    draft_acceptance: persist_draft_acceptance,
                    n_decoded_max: max_decoded_from_config,
                };

                let bar_lines = render_inference_bars(&stats_to_render);
                for (i, line) in bar_lines.iter().enumerate() {
                    execute!(io::stdout(), MoveTo(0, bar_y + i as u16)).unwrap();
                    println!("{}\x1b[K", line);
                }
                bar_y += bar_lines.len() as u16;
            }

            y = bar_y;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(parse_color_str(&config.title), "RAW LOG"));
            println_line("");
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println_line("──────────────────────────────────────────");
            y += 1;

             let all_log_lines: Vec<String> = {
                let mut result = Vec::new();
                for slot_id in &slot_order {
                    for line in slot_map.get(slot_id).unwrap() {
                        result.push(line.to_string());
                    }
                }
                for line in &all_non_slot_lines {
                    result.push(line.clone());
                }
                result
            };

            let log_display = render_log_window(&all_log_lines, config.log_height);
            for (i, line) in log_display.iter().enumerate() {
                execute!(io::stdout(), MoveTo(0, y + i as u16)).unwrap();
                println!("{}\x1b[K", line);
            }
            y += log_display.len() as u16;
        }

        // Clear leftover lines from a taller previous frame.
        for cy in y..prev_height {
            execute!(io::stdout(), MoveTo(0, cy)).unwrap();
            print!("{:120}\x1b[K", " ");
        }

        prev_height = y;

        // Show cursor again and flush.
        execute!(io::stdout(), Show).unwrap();
        io::stdout().flush().unwrap();

        // Check for key press
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
    print!("\x1b[?25h"); // show cursor
    print!("\x1b[2J\x1b[H");
}
