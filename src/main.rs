mod agents;
mod docker;
mod gui;
mod hw;
mod hw_inventory;
mod icons;
mod ports;
mod procs;
mod strata;
mod tui;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "syskamling",
    about = "SysKamling: developer system monitor - hardware, ports, and Docker containers",
    version,
    subcommand_negates_reqs = false
)]
struct Cli {
    /// Path to the Docker daemon socket.
    #[arg(short, long, default_value = "/var/run/docker.sock", global = true)]
    socket: String,

    /// Refresh interval in seconds (0 disables polling).
    #[arg(short = 'i', long, default_value_t = 2.0, global = true)]
    refresh: f64,

    /// Screenshot harness: open directly on this tab (e.g. "processes"),
    /// auto-close after --shot-after seconds, optionally pre-expanding an
    /// app group in the Processes table.
    #[arg(long, value_name = "TAB")]
    shot: Option<String>,

    #[arg(long, default_value_t = 6.0)]
    shot_after: f64,

    #[arg(long)]
    expand: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the graphical desktop window (default).
    Gui,
    /// Run the terminal UI.
    Tui {
        /// Start directly on the logs screen.
        #[arg(long)]
        logs: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;

    // Connectivity check: if socket file doesn't exist, Docker is not installed/running.
    let socket_exists = std::path::Path::new(&cli.socket).exists();
    let has_docker = if socket_exists {
        let probe = docker::Docker::new(&cli.socket);
        match rt.block_on(probe.ping()) {
            Ok(v) => {
                eprintln!(
                    "connected to Docker daemon ({})",
                    if v.is_empty() { "OK".to_string() } else { v }
                );
                true
            }
            Err(e) => {
                eprintln!(
                    "warning: cannot reach Docker daemon at {}: {}",
                    cli.socket, e
                );
                false
            }
        }
    } else {
        false
    };

    match cli.command {
        None | Some(Command::Gui) => {
            let shot = cli.shot.map(|t| (t, cli.shot_after));
            gui::run_with(rt, &cli.socket, cli.refresh, shot, cli.expand, has_docker)
        }
        Some(Command::Tui { logs }) => tui::run(rt, &cli.socket, cli.refresh, logs, has_docker),
    }
}
