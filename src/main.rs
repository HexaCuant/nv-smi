use std::env;
use std::fs;
use std::io;
use std::process::Command;
use std::thread;
use std::time::Duration;

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::execute;
use crossterm::style::{Color, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};
use yaml_rust::YamlLoader;

#[derive(Debug)]
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

fn parse_float(value: &str) -> f64 {
    value
        .replace("%", "")
        .replace("C", "")
        .parse()
        .unwrap_or(0.0)
}

fn parse_u32(value: &str) -> u32 {
    value
        .replace('W', "")
        .replace("MiB", "")
        .replace("GiB", "")
        .parse()
        .unwrap_or(0)
}

fn parse_gpus(output: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    let mut gpu_id: usize = 0;
    for line in output.lines() {
        if line.starts_with('|') {
            let inner = line.trim_start_matches('|').trim_end_matches('|');
            let tokens: Vec<&str> = inner.split_whitespace().collect();
            if !tokens.is_empty() && tokens[0].contains('%') {
                if tokens.len() >= 12 {
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

fn print_colored<S: AsRef<str>>(color: Color, text: S) {
    execute!(io::stdout(), SetForegroundColor(color)).unwrap();
    print!("{}", text.as_ref());
    print!("");
}

fn get_last_lines(path: &str, n: usize) -> Vec<String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    
    let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
    let start = if lines.len() > n { lines.len() - n } else { 0 };
    lines[start..].to_vec()
}

fn render_log_window(lines: &[String], y_start: usize, width: usize, height: usize) {
    for i in 0..height {
        let y = y_start + i;
        execute!(io::stdout(), MoveTo(0, y as u16), Clear(ClearType::CurrentLine)).unwrap();
        
        if i < lines.len() {
            print!("{}", lines[i]);
        }
    }
}

fn draw_bar<S: AsRef<str>>(label: S, value: f64, max: f64, color: Color, empty_color: Color) {
    let percent = if max > 0.0 {
        (value / max) * 100.0
    } else {
        0.0
    };
    let width = 25;
    let filled = ((percent / 100.0) * width as f64).round() as usize;

    execute!(io::stdout(), SetForegroundColor(Color::White)).unwrap();
    print!("{:7}", label.as_ref());
    print!("");

    for i in 0..width {
        if i < filled {
            execute!(io::stdout(), SetForegroundColor(color)).unwrap();
            print!("█");
            print!("");
        } else {
            execute!(io::stdout(), SetForegroundColor(empty_color)).unwrap();
            print!("░");
            print!("");
        }
    }
}

fn main() {
    let config = Config::load();

    loop {
        const MAX_GPUS: usize = 8;
        const LINES_PER_GPU: usize = 9;
        let max_lines = 3 + MAX_GPUS * LINES_PER_GPU;

        let output = get_nvidia_smi();
        let gpus = parse_gpus(&output);
        let actual_lines =
            2 + gpus.len() * (LINES_PER_GPU - 1) + if gpus.len() > 0 { 3 } else { 0 };

        print!("\x1b[2J");
        print!("\x1b[H");

        print_colored(parse_color_str(&config.title), "NV-SMI\n");
        println!("══════════════════════════════════════════=");
        println!();

        let bar_empty = parse_color_str(&config.bar_empty);

        // Calculate total lines: 2 (header) + per GPU
        let lines_per_gpu = 10; // GPU line, empty, temp+bar, power+bar, mem+bar, util+bar, 2 separator lines (except last)
        let total_lines = 3 + gpus.len() * lines_per_gpu;

        for gpu in &gpus {
            print_colored(Color::Magenta, &format!("GPU {}\n", gpu.id));
            println!();

            draw_bar(
                "Temp",
                gpu.temperature,
                100.0,
                get_temp_color(gpu.temperature, &config),
                bar_empty,
            );
            print!(" {}°C\n", gpu.temperature as u32);

            let power_color = parse_color_str(&config.power);
            draw_bar(
                "Power",
                (gpu.power_usage as f64) / (gpu.power_cap as f64) * 100.0,
                100.0,
                power_color,
                bar_empty,
            );
            print!(" {}/{}W\n", gpu.power_usage, gpu.power_cap);

            let mem_color = parse_color_str(&config.memory);
            draw_bar(
                "Memory",
                (gpu.memory_used as f64) / (gpu.memory_total as f64) * 100.0,
                100.0,
                mem_color,
                bar_empty,
            );
            print!(" {}/{}MiB\n", gpu.memory_used, gpu.memory_total);

            draw_bar(
                "Util",
                gpu.gpu_util,
                100.0,
                get_util_color(gpu.gpu_util, &config),
                bar_empty,
            );
            print!(" {}%\n", gpu.gpu_util as u32);

            if gpu.id + 1 < gpus.len() {
                println!();
                println!("─────────────────────────────────────");
            }
        }

        // Render log window if configured
        if let Some(ref log_file) = config.log_file {
            println!();
            print_colored(parse_color_str(&config.title), "LOG\n");
            println!("══════════════════════════════════════════=");
            
            let log_lines = get_last_lines(log_file, config.log_lines);
            let log_y_start = (3 + gpus.len() * 10 + 2) as u16;
            
            render_log_window(
                &log_lines,
                log_y_start as usize,
                80,
                config.log_height
            );
        }

        thread::sleep(Duration::from_secs(2));
    }
}
