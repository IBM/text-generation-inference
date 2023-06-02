use clap::Parser;
use std::env;
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::Arc;
use std::sync::{mpsc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, io};
use std::ffi::OsString;
use subprocess::{Popen, PopenConfig, PopenError, Redirection};
use tracing::info;

/// App Configuration
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "bigscience/bloom-560m", long, env)]
    model_name: String,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(default_value = "hf_accelerate", long, env)]
    deployment_framework: String,
    #[clap(default_value = "float16", long, env)]
    dtype_str: String,
    #[clap(long, env)]
    num_shard: Option<usize>,
    #[clap(default_value = "96", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2048", long, env)]
    max_sequence_length: usize,
    #[clap(default_value = "1024", long, env)]
    max_new_tokens: usize,
    #[clap(default_value = "12", long, env)]
    max_batch_size: usize,
    #[clap(default_value = None, long, env)]
    max_batch_weight: Option<usize>,
    #[clap(default_value = None, long, env)]
    max_prefill_weight: Option<usize>,
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "/tmp/text-generation-server", long, env)]
    shard_uds_path: String,
    #[clap(default_value = "localhost", long, env)]
    master_addr: String,
    #[clap(default_value = "29500", long, env)]
    master_port: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    tls_cert_path: Option<String>,
    #[clap(long, env)]
    tls_key_path: Option<String>,
    #[clap(long, env)]
    tls_client_ca_cert_path: Option<String>,
    #[clap(long, env)]
    output_special_tokens: bool,
}

fn main() -> ExitCode {
    // Pattern match configuration
    let args = Args::parse();

    if args.json_output {
        tracing_subscriber::fmt().json().with_current_span(false).init();
    } else {
        tracing_subscriber::fmt().compact().init();
    }

    info!("Launcher args: {:?}", args);

    // By default we only have one master shard
    let num_shard = args.num_shard.unwrap_or(1);

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Shared shutdown bool
    let shutdown = Arc::new(Mutex::new(false));
    // Shared shutdown channel
    // When shutting down, the main thread will wait for all senders to be dropped
    let (shutdown_sender, ref shutdown_receiver) = mpsc::channel();

    // Shared channel to track shard status
    let (status_sender, status_receiver) = mpsc::channel();

    // Start shard processes
    for rank in 0..num_shard {
        let args = args.clone();
        let status_sender = status_sender.clone();
        let shutdown = shutdown.clone();
        let shutdown_sender = shutdown_sender.clone();
        thread::spawn(move || {
            shard_manager(
                args.model_name,
                args.revision,
                args.deployment_framework,
                args.dtype_str,
                args.shard_uds_path,
                rank,
                num_shard,
                args.master_addr,
                args.master_port,
                status_sender,
                shutdown,
                shutdown_sender,
            )
        });
    }
    drop(shutdown_sender);

    // Wait for shard to start
    let mut shard_ready = 0;
    while running.load(Ordering::SeqCst) {
        match status_receiver.try_recv() {
            Ok(ShardStatus::Ready) => {
                shard_ready += 1;
                if shard_ready == num_shard {
                    break;
                }
            }
            Err(TryRecvError::Empty) => {
                sleep(Duration::from_millis(100));
            }
            Ok(ShardStatus::Failed((rank, err))) => {
                tracing::error!("Shard {rank} failed to start:\n{err}");
                shutdown_shards(shutdown, shutdown_receiver);
                return ExitCode::FAILURE;
            }
            Err(TryRecvError::Disconnected) => {
                tracing::error!("Shard status channel disconnected");
                shutdown_shards(shutdown, shutdown_receiver);
                return ExitCode::FAILURE;
            }
        }
    }

    // We might have received a termination signal
    if !running.load(Ordering::SeqCst) {
        shutdown_shards(shutdown, shutdown_receiver);
        return ExitCode::SUCCESS;
    }

    let tokenizer_path = resolve_tokenizer_path(args.model_name, args.revision)
        .expect("Could not find tokenizer for model");

    // All shard started
    // Start webserver
    tracing::info!("Starting Router");
    let mut argv = vec![
        "text-generation-router".to_string(),
        "--max-concurrent-requests".to_string(),
        args.max_concurrent_requests.to_string(),
        "--max-sequence-length".to_string(),
        args.max_sequence_length.to_string(),
        "--max-new-tokens".to_string(),
        args.max_new_tokens.to_string(),
        "--max-batch-size".to_string(),
        args.max_batch_size.to_string(),
        "--max-waiting-tokens".to_string(),
        args.max_waiting_tokens.to_string(),
        "--port".to_string(),
        args.port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{}-0", args.shard_uds_path),
        "--tokenizer-path".to_string(),
        tokenizer_path,
    ];

    if let Some(max_batch_weight) = args.max_batch_weight {
        argv.push("--max-batch-weight".to_string());
        argv.push(max_batch_weight.to_string());
    }
    if let Some(max_prefill_weight) = args.max_prefill_weight {
        argv.push("--max-prefill-weight".to_string());
        argv.push(max_prefill_weight.to_string());
    }

    if let Some(path) = args.tls_key_path {
        argv.push("--tls-key-path".to_string());
        argv.push(path);
    }
    if let Some(path) = args.tls_cert_path {
        argv.push("--tls-cert-path".to_string());
        argv.push(path);
    }
    if let Some(path) = args.tls_client_ca_cert_path {
        argv.push("--tls-client-ca-cert-path".to_string());
        argv.push(path);
    }

    if args.json_output {
        argv.push("--json-output".to_string());
    }

    if args.output_special_tokens {
        argv.push("--output-special-tokens".into());
    }

    let mut webserver = match Popen::create(
        &argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Pipe,
            // Needed for the shutdown procedure
            setpgid: true,
            // env: Some(vec![("RUST_BACKTRACE".into(), "1".into())]),
            ..Default::default()
        },
    ) {
        Ok(p) => p,
        Err(err) => {
            tracing::error!("Failed to start webserver: {err}");
            if let PopenError::IoError(err) = err {
                if err.kind() == io::ErrorKind::NotFound {
                    tracing::error!("text-generation-router not found in PATH");
                    tracing::error!("Please install it with `make install-router`")
                }
            } else {
                tracing::error!("{err}");
            }

            shutdown_shards(shutdown, &shutdown_receiver);
            return ExitCode::FAILURE;
        }
    };

    // Redirect STDOUT and STDERR to the console
    let webserver_stdout = webserver.stdout.take().unwrap();
    let webserver_stderr = webserver.stderr.take().unwrap();

    thread::spawn(move || {
        let stdout = BufReader::new(webserver_stdout);
        let stderr = BufReader::new(webserver_stderr);
        for line in stdout.lines() {
            println!("{}", line.unwrap());
        }
        for line in stderr.lines() {
            println!("{}", line.unwrap());
        }
    });

    // Default exit code
    let mut exit_code = ExitCode::SUCCESS;

    while running.load(Ordering::SeqCst) {
        if let Ok(ShardStatus::Failed((rank, err))) = status_receiver.try_recv() {
            tracing::error!("Shard {rank} failed:\n{err}");
            exit_code = ExitCode::FAILURE;
            break;
        };

        match webserver.poll() {
            Some(_) => {
                tracing::error!("Webserver Crashed");
                shutdown_shards(shutdown, &shutdown_receiver);
                return ExitCode::FAILURE;
            }
            None => {
                sleep(Duration::from_millis(100));
            }
        };
    }

    // Graceful termination
    webserver.terminate().unwrap();
    tracing::info!("Waiting for router to gracefully shutdown");
    webserver.wait_timeout(Duration::from_secs(120)).unwrap();
    tracing::info!("Router terminated");
    shutdown_shards(shutdown, &shutdown_receiver);

    exit_code
}

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed((usize, String)),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_name: String,
    revision: Option<String>,
    deployment_framework: String,
    dtype_str: String,
    uds_path: String,
    rank: usize,
    world_size: usize,
    master_addr: String,
    master_port: usize,
    status_sender: mpsc::Sender<ShardStatus>,
    shutdown: Arc<Mutex<bool>>,
    _shutdown_sender: mpsc::Sender<()>,
) {
    // Get UDS path
    let uds_string = format!("{uds_path}-{rank}");
    let uds = Path::new(&uds_string);
    // Clean previous runs
    fs::remove_file(uds).unwrap_or_default();

    // Process args
    let mut shard_argv = vec![
        "text-generation-server".to_string(),
        "serve".to_string(),
        model_name,
        deployment_framework,
        dtype_str,
        "--uds-path".to_string(),
        uds_path,
    ];

    // Activate tensor parallelism
    if world_size > 1 {
        shard_argv.push("--sharded".to_string());
    }

    // Model optional revision
    if let Some(revision) = revision {
        shard_argv.push("--revision".to_string());
        shard_argv.push(revision)
    }

    // Copy current process env
    let mut env: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Torch Distributed / DeepSpeed Env vars
    env.push(("RANK".into(), rank.to_string().into()));
    env.push(("LOCAL_RANK".into(), rank.to_string().into()));
    env.push(("LOCAL_SIZE".into(), world_size.to_string().into()));
    env.push(("CROSS_SIZE".into(), "1".into()));
    env.push(("CROSS_RANK".into(), "0".into()));
    env.push(("WORLD_SIZE".into(), world_size.to_string().into()));
    env.push(("MASTER_ADDR".into(), master_addr.into()));
    env.push(("MASTER_PORT".into(), master_port.to_string().into()));
    env.push(("NCCL_ASYNC_ERROR_HANDLING".into(), "1".into()));

    // Safetensors load fast
    env.push(("SAFETENSORS_FAST_GPU".into(), "1".into()));

    // Disable python stdout buffering
    env.push(("PYTHONUNBUFFERED".into(), "1".into()));

    // Start process
    tracing::info!("Starting shard {rank}");
    let mut p = match Popen::create(
        &shard_argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Pipe,
            // Needed for the shutdown procedure
            setpgid: true,
            // NCCL env vars
            env: Some(env),
            ..Default::default()
        },
    ) {
        Ok(p) => p,
        Err(err) => {
            if let PopenError::IoError(ref err) = err {
                if err.kind() == io::ErrorKind::NotFound {
                    tracing::error!("text-generation-server not found in PATH");
                    tracing::error!("Please install it with `make install-server`")
                }
            }
            status_sender
                .send(ShardStatus::Failed((rank, err.to_string())))
                .unwrap();
            return;
        }
    };

    // Redirect STDOUT and STDERR to the console
    let shard_stdout = p.stdout.take().unwrap();
    thread::spawn(move || BufReader::new(shard_stdout).lines().for_each(|line|
        println!("Shard {}: {}", rank, line.unwrap())
    ));
    let shard_stderr = p.stderr.take().unwrap();
    thread::spawn(move || BufReader::new(shard_stderr).lines().for_each(|line|
        eprintln!("Shard {}: {}", rank, line.unwrap())
    ));

    let mut ready = false;
    let start_time = Instant::now();
    let mut wait_time = Instant::now();
    loop {
        // Process exited
        if p.poll().is_some() {
            let mut err = String::new();
            //We don't need to do this now that we're logging
            //p.stderr.take().unwrap().read_to_string(&mut err).unwrap();
            status_sender
                .send(ShardStatus::Failed((rank, err)))
                .unwrap();
            return;
        }

        // We received a shutdown signal
        if *shutdown.lock().unwrap() {
            p.terminate().unwrap();
            let _ = p.wait_timeout(Duration::from_secs(90));
            tracing::info!("Shard {rank} terminated");
            return;
        }

        // Shard is ready
        if uds.exists() && !ready {
            tracing::info!("Shard {rank} ready in {:?}", start_time.elapsed());
            status_sender.send(ShardStatus::Ready).unwrap();
            ready = true;
        } else if !ready && wait_time.elapsed() > Duration::from_secs(10) {
            tracing::info!("Waiting for shard {rank} to be ready...");
            wait_time = Instant::now();
        }
        sleep(Duration::from_millis(100));
    }
}

fn shutdown_shards(shutdown: Arc<Mutex<bool>>, shutdown_receiver: &mpsc::Receiver<()>) {
    tracing::info!("Shutting down shards");
    // Update shutdown value to true
    // This will be picked up by the shard manager
    {
        let mut shutdown = shutdown.lock().unwrap();
        *shutdown = true;
    }

    // Wait for shards to shutdown
    // This will block till all shutdown_sender are dropped
    let _ = shutdown_receiver.recv();
}


fn resolve_tokenizer_path(model_name: String, revision: Option<String>) -> Result<String, io::Error> {
    let cache = env::var("TRANSFORMERS_CACHE")
        .map_err(|e| io::Error::new(ErrorKind::InvalidInput, e))?;
    let model_dir = Path::new(&cache).join(
        format!("models--{}", model_name.replace("/", "--"))
    );
    if !model_dir.try_exists()? {
        // Try treating model_name as explicit model path
        let try_path = Path::new(&model_name).join("tokenizer.json");
        if try_path.try_exists()? {
            return Ok(try_path.to_string_lossy().into())
        }

        return Err(io::Error::new(
            ErrorKind::NotFound,
            format!("Model {model_name} not found in local cache")
        ))
    }
    let main_ref_path = model_dir.join("refs")
        .join(revision.unwrap_or("main".into()));
    let main_ref = fs::read_to_string(main_ref_path)?;
    let tok_path = model_dir.join("snapshots")
        .join(main_ref).join("tokenizer.json");
    if tok_path.try_exists()? {
        Ok(tok_path.to_string_lossy().into())
    } else {
        Err(io::Error::new(
            ErrorKind::NotFound,
            format!("Tokenizer file not found in local cache for model {model_name}")
        ))
    }
}