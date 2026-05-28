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
    n_tokens: u32,
    progress: f64,
    time_seconds: f64,
    tokens_per_second: f64,
    task_id: u32,
    slot_id: u32,
    n_decoded: u32,
    gen_speed_tps: f64,
    latency_ms_tok: f64,
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
                "-ngl" | "-np" | "-t" | "-tb" | "-c" | "--top-k" | "--top-p"
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
            return Some(LlamaServerInfo { model, params });
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

fn parse_inference_stats(line: &str) -> Option<InferenceStats> {
    if !line.contains("print_timing") {
        return None;
    }

    let n_tokens = extract_number_after(line, "n_tokens");
    let progress = extract_float_after(line, "progress");
    let time = extract_float_after(line, "t =");
    let tps = extract_tokens_per_second(line);
    let task_id = extract_number_after(line, "task");
    let slot_id = extract_number_after(line, "id");

    if n_tokens > 0 && progress > 0.0 && time > 0.0 && tps > 0.0 {
        Some(InferenceStats {
            n_tokens,
            progress,
            time_seconds: time,
            tokens_per_second: tps,
            task_id,
            slot_id,
            n_decoded: 0,
            gen_speed_tps: 0.0,
            latency_ms_tok: 0.0,
        })
    } else {
        None
    }
}

fn parse_generation_stats(line: &str) -> Option<(u32, f64)> {
    if !line.contains("n_decoded") {
        return None;
    }

    let n_decoded = extract_number_after(line, "n_decoded");
    let gen_speed = extract_float_after(line, "tg =");

    if n_decoded > 0 && gen_speed > 0.0 {
        Some((n_decoded, gen_speed))
    } else {
        None
    }
}

fn parse_latency(line: &str) -> Option<f64> {
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

fn extract_tokens_per_second(line: &str) -> f64 {
    if let Some(pos) = line.find("tokens per second") {
        let start = line.rfind('/').unwrap_or(pos - 20);
        let section = &line[start + 1..pos];
        let nums: String = section.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
        nums.parse().unwrap_or(0.0)
    } else {
        0.0
    }
}

fn render_inference_bars(stats: &InferenceStats) -> Vec<String> {
    let bar_empty = parse_color_str(&Config::default().bar_empty);
    vec![
        format!("{} {:.0}%",
            format_bar("Progress", stats.progress * 100.0, 100.0, Color::Green, bar_empty),
            stats.progress * 100.0),
        format!("{} {}/s",
            format_bar("Prompt t/s", stats.tokens_per_second, 1000.0, Color::Cyan, bar_empty),
            stats.tokens_per_second as u32),
        format!("{} {}/s",
            format_bar("Gen t/s", stats.gen_speed_tps, 50.0, Color::Green, bar_empty),
            stats.gen_speed_tps as u32),
        format!("{} {}",
            format_bar("Decoded", stats.n_decoded as f64, 1000.0, Color::Magenta, bar_empty),
            stats.n_decoded),
        format!("{} {:.2}ms",
            format_bar("Latency", stats.latency_ms_tok, 5.0, Color::Yellow, bar_empty),
            stats.latency_ms_tok),
        format!("{} {:.2}s",
            format_bar("Time", stats.time_seconds, 60.0, Color::Yellow, bar_empty),
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

fn main() {
    let config = Config::load();

    let _ = enable_raw_mode();

    let mut prev_gpus: Vec<GpuInfo> = Vec::new();
    let mut prev_llama: Option<String> = None;
    let mut prev_log_lines: Vec<String> = Vec::new();
    let mut prev_height: u16 = 0;

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
        println!();
        y += 1;

        if let Some(ref info) = llama_info {
            let model_name = info.model.rsplit('/').next().unwrap_or("unknown");
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(Color::Yellow, &format!("Model: {}", model_name)));
            println!();
            y += 1;

            if !info.params.is_empty() {
                let col_w = 14usize;
                let cols = 3usize;
                for chunk in info.params.chunks(cols) {
                    execute!(io::stdout(), MoveTo(0, y)).unwrap();
                    for (key, val) in chunk {
                        print!("{:width$} ", format!("{}={}", key, val), width = col_w);
                    }
                    println!();
                    y += 1;
                }
            }

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println!("─────────────────────────────────────");
            y += 1;
        } else {
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(Color::DarkGrey, "(no llama-server running)"));
            println!();
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println!("─────────────────────────────────────");
            y += 1;
        }

        execute!(io::stdout(), MoveTo(0, y)).unwrap();
        println!();
        y += 1;

        let bar_empty = parse_color_str(&config.bar_empty);

        for gpu in &gpus {
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(Color::Magenta, &format!("GPU {}", gpu.id)));
            println!();
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println!();
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Temp", gpu.temperature, 100.0, get_temp_color(gpu.temperature, &config), bar_empty));
            println!(" {}°C", gpu.temperature as u32);
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Fan", gpu.fan_speed, 100.0, Color::White, bar_empty));
            println!(" {:.0}%", gpu.fan_speed);
            y += 1;

            let power_color = parse_color_str(&config.power);
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Power", (gpu.power_usage as f64) / (gpu.power_cap as f64) * 100.0, 100.0, power_color, bar_empty));
            println!(" {}/{}W", gpu.power_usage, gpu.power_cap);
            y += 1;

            let power_left = if gpu.power_cap > gpu.power_usage { gpu.power_cap - gpu.power_usage } else { 0 };
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Pwr Left", power_left as f64, gpu.power_cap as f64, power_color, bar_empty));
            println!(" {}W", power_left);
            y += 1;

            let mem_color = parse_color_str(&config.memory);
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Memory", (gpu.memory_used as f64) / (gpu.memory_total as f64) * 100.0, 100.0, mem_color, bar_empty));
            println!(" {}/{}MiB", gpu.memory_used, gpu.memory_total);
            y += 1;

            let mem_free = if gpu.memory_total > gpu.memory_used { gpu.memory_total - gpu.memory_used } else { 0 };
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Mem Free", mem_free as f64, gpu.memory_total as f64, mem_color, bar_empty));
            println!(" {}MiB", mem_free);
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_bar("Util", gpu.gpu_util, 100.0, get_util_color(gpu.gpu_util, &config), bar_empty));
            println!(" {}%", gpu.gpu_util as u32);
            y += 1;

            if gpu.id + 1 < gpus.len() {
                execute!(io::stdout(), MoveTo(0, y)).unwrap();
                println!();
                y += 1;
                execute!(io::stdout(), MoveTo(0, y)).unwrap();
                println!("─────────────────────────────────────");
                y += 1;
            }
        }

        // Render log window if configured
        if let Some(ref log_file) = config.log_file {
            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println!();
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            print!("{}", format_colored(parse_color_str(&config.title), "LOG"));
            println!();
            y += 1;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println!("═══════════════════════════════════════════");
            y += 1;

            let log_lines_data = get_last_lines(log_file, config.log_lines);

            // Try to parse inference stats from the most recent lines
            let mut last_stats: Option<InferenceStats> = None;
            let mut last_n_decoded: u32 = 0;
            let mut last_gen_speed: f64 = 0.0;
            let mut last_latency: f64 = 0.0;

            for line in &log_lines_data {
                if let Some(stats) = parse_inference_stats(line) {
                    last_stats = Some(stats);
                }
                if let Some((nd, gs)) = parse_generation_stats(line) {
                    last_n_decoded = nd;
                    last_gen_speed = gs;
                }
                if let Some(lat) = parse_latency(line) {
                    last_latency = lat;
                }
            }

            // Always render inference bars (with zeros if no data yet)
            let mut stats_to_render = last_stats.unwrap_or(InferenceStats {
                n_tokens: 0,
                progress: 0.0,
                time_seconds: 0.0,
                tokens_per_second: 0.0,
                task_id: 0,
                slot_id: 0,
                n_decoded: 0,
                gen_speed_tps: 0.0,
                latency_ms_tok: 0.0,
            });
            stats_to_render.n_decoded = last_n_decoded;
            stats_to_render.gen_speed_tps = last_gen_speed;
            stats_to_render.latency_ms_tok = last_latency;

            let bar_lines = render_inference_bars(&stats_to_render);
            for (i, line) in bar_lines.iter().enumerate() {
               execute!(io::stdout(), MoveTo(0, y + i as u16)).unwrap();
                println!("{}", line);
            }
            y += bar_lines.len() as u16;

            execute!(io::stdout(), MoveTo(0, y)).unwrap();
            println!();
            y += 1;

            let log_display = render_log_window(&log_lines_data, config.log_height);
            for (i, line) in log_display.iter().enumerate() {
                execute!(io::stdout(), MoveTo(0, y + i as u16)).unwrap();
                println!("{}", line);
            }
            y += log_display.len() as u16;
        }

        // Clear leftover lines from a taller previous frame.
        for cy in y..prev_height {
            execute!(io::stdout(), MoveTo(0, cy)).unwrap();
            println!("{:120}", " ");
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
