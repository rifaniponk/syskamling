mod docker;
mod gui;
mod tui;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "syskamling",
    about = "SysKamling: system watch for Linux - hardware, ports, and Docker containers",
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

    // Connectivity check up front so the user sees a real error on stderr.
    let probe = docker::Docker::new(&cli.socket);
    match rt.block_on(probe.ping()) {
        Ok(v) => eprintln!(
            "connected to Docker daemon ({})",
            if v.is_empty() { "OK".to_string() } else { v }
        ),
        Err(e) => {
            eprintln!("warning: cannot reach Docker daemon at {}: {}", cli.socket, e);
        }
    }

    match cli.command {
        None | Some(Command::Gui) => gui::run(rt, &cli.socket, cli.refresh),
        Some(Command::Tui { logs }) => tui::run(rt, &cli.socket, cli.refresh, logs),
    }
}
