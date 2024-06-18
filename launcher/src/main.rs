use std::{
    env,
    env::VarError,
    ffi::OsString,
    fs,
    fs::File,
    io,
    io::{BufRead, BufReader, ErrorKind, Write},
    os::unix::process::{CommandExt, ExitStatusExt},
    path::{Path, PathBuf},
    process::{Command, ExitCode, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
        mpsc::TryRecvError,
        Arc, Mutex,
    },
    thread,
    thread::sleep,
    time::{Duration, Instant},
};

use clap::Parser;
use nix::{
    sys::signal::{self, Signal},
    unistd::Pid,
};
use tracing::{error, info, warn};
use uuid::Uuid;

// In most cases this gives the best performance for inferencing
const DEFAULT_PYTORCH_CUDA_ALLOC_CONF: &str = "expandable_segments:True";
const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 2048;

/// App Configuration
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "bigscience/bloom-560m", long, env)]
    model_name: String,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(default_value = "hf_transformers", long, env)]
    deployment_framework: String,
    #[clap(default_value = None, long, env)]
    dtype: Option<String>,
    #[clap(default_value = None, long, env)]
    dtype_str: Option<String>,
    #[clap(default_value = None, long, env)]
    quantize: Option<String>,
    #[clap(long, env)]
    num_shard: Option<usize>,
    #[clap(default_value = "512", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = None, long, env)]
    max_sequence_length: Option<usize>,
    #[clap(default_value = "1024", long, env)]
    max_new_tokens: usize,
    #[clap(default_value = "12", long, env)]
    max_batch_size: usize,
    #[clap(default_value = "0.2", long, env)]
    max_prefill_padding: f32,
    #[clap(default_value = "20", long, env)]
    batch_safety_margin: usize,
    #[clap(default_value = "24", long, env)]
    max_waiting_tokens: usize,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "8033", long, short, env)]
    grpc_port: u16,
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
    #[clap(default_value = "1.0", long, short, env)]
    cuda_process_memory_fraction: f32,
    // Default for default_include_stop_seqs is true for now, for backwards compatibility
    #[clap(default_value = "true", long, env, action = clap::ArgAction::Set)]
    default_include_stop_seqs: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(long, env)]
    otlp_service_name: Option<String>,
}

fn main() -> ExitCode {
    // Register a panic handler up-front to write to /dev/termination-log
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        if let Some(&s) = panic_info.payload().downcast_ref::<&str>() {
            _ = write_termination_log(s);
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            _ = write_termination_log(s);
        }
        // No else case: If we cannot get good panic info, we won't write anything to the
        // termination log. The system logs should contain better information.
        default_hook(panic_info);
    }));

    // Pattern match configuration
    let args = Args::parse();
    if args.json_output {
        tracing_subscriber::fmt()
            .json()
            .with_current_span(false)
            .init();
    } else {
        tracing_subscriber::fmt().compact().init();
    }
    // log TGIS commit hash
    if let Some(commit_hash) = option_env!("GIT_COMMIT_HASH") {
        info!("TGIS Commit hash: {commit_hash}");
    }

    info!("Launcher args: {:?}", args);
    if args.cuda_process_memory_fraction <= 0.0 || args.cuda_process_memory_fraction > 1.0 {
        panic!("cuda_process_memory_fraction must be in range 0 < x <= 1");
    }

    match (args.dtype.as_ref(), args.dtype_str.as_ref()) {
        (Some(dt), Some(dt_s)) if dt != dt_s => {
            panic!("dtype and dtype_str args both provided with different values")
        }
        _ => (),
    }

    // Determine number of shards based on command line arg and env vars
    let num_shard = find_num_shards(args.num_shard);

    // Determine the model cache path and resolve from possible env vars:
    // - HF_HUB_CACHE
    // - TRANSFORMERS_CACHE (deprecated)
    // - HUGGINGFACE_HUB_CACHE (deprecated)
    //
    // We allow multiple to be set for compatibility, but then the values must match.

    let mut cache_env_var: String = "".to_string();
    let mut cache_env_value: String = "".to_string();

    if let Ok(t) = env::var("HF_HUB_CACHE") {
        cache_env_var = "HF_HUB_CACHE".into();
        cache_env_value = t.into();
    }

    for deprecated_env_var in vec!["TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        match (
            env::var(deprecated_env_var),
            !cache_env_var.is_empty(),
        ) {
            (Ok(t), false) => {
                cache_env_var = deprecated_env_var.into();
                cache_env_value = t.into();
            },
            (Ok(t), true) if t != cache_env_value => panic!(
                "{deprecated_env_var} and {cache_env_var} env vars can't be set to different values"
            ),
            (Ok(_), true) => warn!(
                "{deprecated_env_var} is deprecated and should not be used. Use HF_HUB_CACHE instead."
            ),
            _ => (),
        }
    }

    // ensure HF_HUB_CACHE is set for downstream usage
    // default value to match huggingface_hub
    // REF: https://github.com/huggingface/huggingface_hub/blob/5ff2d150d121d04799b78bc08f2343c21b8f07a9/docs/source/en/package_reference/environment_variables.md?plain=1#L32
    let cache_path = if !cache_env_value.is_empty() {
        PathBuf::from(cache_env_value)
    } else if let Ok(hf_home) = env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".cache").join("huggingface").join("hub")
    } else {
        PathBuf::new()
    };

    let config_path: PathBuf = resolve_config_path(cache_path.clone(), &args.model_name, args.revision.as_deref())
        .expect("Failed to resolve config path")
        .into();

    // Save fast tokenizer to /tmp
    let save_path = format!("/tmp/{}", Uuid::new_v4());
    let tokenizer_path =
        if save_fast_tokenizer(&args.model_name, args.revision.as_deref(), &save_path).is_ok() {
            format!("/{save_path}/tokenizer.json")
        } else {
            warn!("Failed to (re-)convert tokenizer, falling back to use existing fast tokenizer");
            let tokenizer_path = config_path.parent().unwrap().join("tokenizer.json");
            if tokenizer_path.is_file() {
                tokenizer_path.to_string_lossy().to_string()
            } else {
                panic!("No existing fast tokenizer (tokenizer.json) found")
            }
        };

    // Determine max sequence length based on command line arg and env vars
    let max_sequence_length = get_max_sequence_length(args.max_sequence_length, &config_path);

    match env::var("MAX_BATCH_WEIGHT") {
        Ok(max_batch_weight) if !max_batch_weight.trim().is_empty() => {
            warn!(
                "MAX_BATCH_WEIGHT is set to {max_batch_weight} but this parameter will be ignored."
            );
        }
        _ => {}
    }

    match env::var("MAX_PREFILL_WEIGHT") {
        Ok(max_prefill_weight) if !max_prefill_weight.trim().is_empty() => {
            warn!("MAX_PREFILL_WEIGHT is set to {max_prefill_weight} but this parameter will be ignored.");
        }
        _ => {}
    }

    // Set PYTORCH_CUDA_ALLOC_CONF to default value if it's not set in the environment
    let cuda_alloc_conf = match env::var("PYTORCH_CUDA_ALLOC_CONF") {
        Err(VarError::NotPresent) if DEFAULT_PYTORCH_CUDA_ALLOC_CONF.is_empty() => None,
        Err(VarError::NotPresent) => {
            info!("Setting PYTORCH_CUDA_ALLOC_CONF to default value: {DEFAULT_PYTORCH_CUDA_ALLOC_CONF}");
            Some(DEFAULT_PYTORCH_CUDA_ALLOC_CONF)
        }
        Ok(alloc_conf) if alloc_conf.trim().is_empty() => {
            info!("PYTORCH_CUDA_ALLOC_CONF is unset");
            Some("") // This means remove it from the env
        }
        Ok(alloc_conf) => {
            info!("PYTORCH_CUDA_ALLOC_CONF is set to: {alloc_conf}");
            None
        }
        Err(VarError::NotUnicode(_)) => panic!("PYTORCH_CUDA_ALLOC_CONF set to non-unicode value"),
    };

    // Backwards compatibility for "hf_custom_tp" deployment engine name
    let deployment_framework = if args.deployment_framework == "hf_custom_tp" {
        warn!(
            "The \"hf_custom_tp\" deployment engine name is deprecated, please use \"tgis_native\""
        );
        "tgis_native"
    } else {
        &args.deployment_framework
    };

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
    let cache_path_string = cache_path.into_os_string();
    for rank in 0..num_shard {
        let args = args.clone();
        let cache_path = cache_path_string.clone();
        let deployment_framework = deployment_framework.to_string();
        let status_sender = status_sender.clone();
        let shutdown = shutdown.clone();
        let shutdown_sender = shutdown_sender.clone();
        thread::spawn(move || {
            shard_manager(
                args.model_name,
                cache_path,
                args.revision,
                deployment_framework,
                args.dtype.or(args.dtype_str),
                args.quantize,
                max_sequence_length,
                args.max_new_tokens,
                args.max_batch_size,
                args.batch_safety_margin,
                args.shard_uds_path,
                args.cuda_process_memory_fraction,
                cuda_alloc_conf,
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
            Ok(ShardStatus::Failed(_status)) => {
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

    // All shard started
    // Start webserver
    info!("Starting Router");
    let mut argv = vec![
        "--max-concurrent-requests".to_string(),
        args.max_concurrent_requests.to_string(),
        "--max-sequence-length".to_string(),
        max_sequence_length.to_string(),
        "--max-new-tokens".to_string(),
        args.max_new_tokens.to_string(),
        "--max-batch-size".to_string(),
        args.max_batch_size.to_string(),
        "--max-prefill-padding".to_string(),
        args.max_prefill_padding.to_string(),
        "--max-waiting-tokens".to_string(),
        args.max_waiting_tokens.to_string(),
        "--port".to_string(),
        args.port.to_string(),
        "--grpc-port".to_string(),
        args.grpc_port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{}-0", args.shard_uds_path),
        "--tokenizer-path".to_string(),
        tokenizer_path,
    ];

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

    // OpenTelemetry
    if let Some(otlp_endpoint) = args.otlp_endpoint {
        argv.push("--otlp-endpoint".to_string());
        argv.push(otlp_endpoint);

        if let Some(otlp_service_name) = args.otlp_service_name {
            argv.push("--otlp-service-name".to_string());
            argv.push(otlp_service_name);
        }
    }

    if args.output_special_tokens {
        argv.push("--output-special-tokens".into());
    }

    if args.default_include_stop_seqs {
        argv.push("--default-include-stop-seqs".into());
    }

    let mut webserver = match Command::new("text-generation-router")
        .args(argv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == ErrorKind::NotFound {
                tracing::error!("text-generation-router not found in PATH");
                tracing::error!("Please install it with `make install-router`")
            } else {
                tracing::error!("Failed to start webserver: {err}");
            }

            shutdown_shards(shutdown, shutdown_receiver);
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
        if let Ok(ShardStatus::Failed(status)) = status_receiver.try_recv() {
            exit_code = ExitCode::FAILURE;
            terminate_gracefully(&mut webserver, shutdown.clone(), shutdown_receiver);
            if status.signal() == Some(7) && num_shard > 1 {
                panic!(
                    "Encountered SIGBUS error. This is usually caused by NCCL having insufficient shared memory. \
                    Ensure at least 1GB of shared memory is available. In case of OpenShift/K8s, \
                    mount a memory medium emptyDir volume to /dev/shm"
                )
            }
            return exit_code;
        };

        match webserver
            .try_wait()
            .expect("Error polling status of router process")
        {
            Some(_) => {
                tracing::error!("Webserver Crashed");
                shutdown_shards(shutdown, shutdown_receiver);
                return ExitCode::FAILURE;
            }
            None => sleep(Duration::from_millis(100)),
        };
    }

    // Graceful termination
    terminate_gracefully(&mut webserver, shutdown.clone(), shutdown_receiver);

    exit_code
}

/// Graceful termination
fn terminate_gracefully(
    webserver: &mut std::process::Child,
    shutdown: Arc<Mutex<bool>>,
    shutdown_receiver: &mpsc::Receiver<()>,
) {
    signal::kill(Pid::from_raw(webserver.id() as i32), Signal::SIGTERM).unwrap();
    info!("Waiting for router to gracefully shutdown");
    webserver.wait().unwrap();
    info!("Router terminated");
    shutdown_shards(shutdown, shutdown_receiver);
}

fn num_cuda_devices() -> Option<usize> {
    let devices = match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(devices) => devices,
        Err(_) => env::var("NVIDIA_VISIBLE_DEVICES").ok()?,
    };
    let n_devices = devices.split(',').count();
    Some(n_devices)
}
/// Finds a max sequence length for the model. In priority order:
/// 1. MAX_SEQUENCE_LENGTH set by user
/// 2. The sequence length specified in config.json
/// 3. A default of 2048
fn get_max_sequence_length(max_sequence_length: Option<usize>, config_path: &Path) -> usize {
    if let Some(max_sequence_length) = max_sequence_length {
        info!(
            "Using configured max_sequence_length: {}",
            max_sequence_length
        );
        return max_sequence_length;
    }
    if let Ok(model_config) = get_config_json(config_path) {
        if let Some(length) = get_max_sequence_length_from_config(&model_config) {
            info!(
                "Loaded max_sequence_length from model config.json: {}",
                length
            );
            return length;
        }
    }
    info!(
        "Using default max_sequence_length: {}",
        DEFAULT_MAX_SEQUENCE_LENGTH
    );
    DEFAULT_MAX_SEQUENCE_LENGTH
}

/// Opens the model's config.json file and reads into serde_json value
fn get_config_json(config_path: &Path) -> Result<serde_json::Value, std::io::Error> {
    let reader = BufReader::new(File::open(config_path)?);
    Ok(serde_json::from_reader(reader)?)
}

/// Attempts to find a max sequence length from the model's config.
/// There is no standard field for this, different model architectures name it differently.
/// This checks for some well-known field names, and returns nothing if none are found.
/// referenced from: https://github.com/vllm-project/vllm/blob/923797fea4d80a4dac4409ece3c450b84d5ba001/vllm/config.py#L557-L592
fn get_max_sequence_length_from_config(model_config: &serde_json::Value) -> Option<usize> {
    let possible_keys = [
        // OPT
        "max_position_embeddings",
        // GPT-2
        "n_positions",
        // MPT
        "max_seq_len",
        // ChatGLM2
        "seq_length",
        // Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ];

    for key in possible_keys {
        if let Some(value) = model_config.get(key) {
            if let Some(value) = value.as_u64() {
                return Some(value as usize);
            }
        }
    }
    None
}

fn find_num_shards(num_shard: Option<usize>) -> usize {
    // get the number of shards given `num_gpu` and `num_shard`
    let num_gpus = env::var("NUM_GPUS").ok().map(|s| {
        s.parse::<usize>()
            .expect("NUM_GPUS must be a positive integer")
    });
    let num_shard = match (num_gpus, num_shard) {
        (Some(num_gpu), None) => num_gpu,
        (None, Some(num_shard)) => num_shard,
        (Some(num_gpu), Some(num_shard)) => {
            if num_gpu != num_shard {
                panic!("NUM_GPUS and num_shard are set to different values ({num_gpu} and {num_shard})");
            }
            num_shard
        }
        // try to default to the number of available GPUs
        (None, None) => match num_cuda_devices() {
            Some(num_shard) => {
                info!("Inferring num_shard = {num_shard} from CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES");
                num_shard
            }
            None => {
                // By default we only have one master shard
                info!("Defaulting num_shard to 1");
                1
            }
        },
    };
    if num_shard < 1 {
        panic!("`num_shard` / NUM_GPUS cannot be < 1");
    }
    num_shard
}

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed(ExitStatus),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_name: String,
    cache_path: OsString,
    revision: Option<String>,
    deployment_framework: String,
    dtype: Option<String>,
    quantize: Option<String>,
    max_sequence_length: usize,
    max_new_tokens: usize,
    max_batch_size: usize,
    batch_safety_margin: usize,
    uds_path: String,
    cuda_process_memory_fraction: f32,
    cuda_alloc_conf: Option<&str>,
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
    if uds.exists() {
        fs::remove_file(uds).unwrap_or_default();
    }

    // Process args
    let mut shard_argv = vec![
        "serve".to_string(),
        model_name,
        deployment_framework,
        // Max seq length, new tokens, batch size
        // only used for PT2 compile warmup
        "--max-sequence-length".to_string(),
        max_sequence_length.to_string(),
        "--max-new-tokens".to_string(),
        max_new_tokens.to_string(),
        "--max-batch-size".to_string(),
        max_batch_size.to_string(),
        "--batch-safety-margin".to_string(),
        batch_safety_margin.to_string(),
        "--uds-path".to_string(),
        uds_path,
        "--cuda-process-memory-fraction".to_string(),
        cuda_process_memory_fraction.to_string(),
    ];

    if let Some(dtype) = dtype {
        shard_argv.push("--dtype".to_string());
        shard_argv.push(dtype);
    }

    if let Some(quantize) = quantize {
        shard_argv.push("--quantize".to_string());
        shard_argv.push(quantize);
    }

    // Activate tensor parallelism
    if world_size > 1 {
        shard_argv.push("--sharded".to_string());
    }

    // Model optional revision
    if let Some(revision) = revision {
        shard_argv.push("--revision".to_string());
        shard_argv.push(revision);
    }

    // Copy current process env
    let mut env: Vec<(OsString, OsString)> = env::vars_os().collect();

    if let Some(alloc_conf) = cuda_alloc_conf {
        if alloc_conf.is_empty() {
            // Remove it from env
            env.retain(|(k, _)| k != "PYTORCH_CUDA_ALLOC_CONF");
        } else {
            env.push(("PYTORCH_CUDA_ALLOC_CONF".into(), alloc_conf.into()));
        }
    }

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

    // See https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    env.push(("TORCH_NCCL_AVOID_RECORD_STREAMS".into(), "1".into()));

    // Safetensors load fast
    env.push(("SAFETENSORS_FAST_GPU".into(), "1".into()));

    // Disable python stdout buffering
    env.push(("PYTHONUNBUFFERED".into(), "1".into()));

    // Ensure offline-only
    env.push(("HF_HUB_OFFLINE".into(), "1".into()));

    // Ensure that we set the standard cache variable
    env.push(("HF_HUB_CACHE".into(), cache_path.into()));

    // Start process
    info!("Starting shard {rank}");
    let mut p = match Command::new("text-generation-server")
        .args(shard_argv)
        .envs(env)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == ErrorKind::NotFound {
                tracing::error!("text-generation-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            } else {
                tracing::error!("Shard {rank} failed to start:\n{err}");
            }
            status_sender
                .send(ShardStatus::Failed(ExitStatus::from_raw(0)))
                .unwrap();
            return;
        }
    };

    // Redirect STDOUT and STDERR to the console
    let shard_stdout = p.stdout.take().unwrap();
    let stdout_thread = thread::spawn(move || {
        BufReader::new(shard_stdout)
            .lines()
            .for_each(|line| println!("Shard {rank}: {}", line.unwrap()))
    });
    let shard_stderr = p.stderr.take().unwrap();
    let stderr_thread = thread::spawn(move || {
        BufReader::new(shard_stderr)
            .lines()
            .for_each(|line| eprintln!("Shard {rank}: {}", line.unwrap()))
    });

    let mut ready = false;
    let start_time = Instant::now();
    let mut wait_time = Instant::now();
    loop {
        // Process exited
        if let Some(status) = p
            .try_wait()
            .unwrap_or_else(|_| panic!("Error polling status of shard {rank}"))
        {
            if *shutdown.lock().unwrap() {
                info!("Shard {rank} terminated");
            } else {
                tracing::error!("Shard {rank} failed: {status:?}");
                // Ensure we finish propagating any final stdout/stderr from the shard
                stdout_thread.join().unwrap_or_default();
                io::stdout().flush().unwrap_or_default();
                stderr_thread.join().unwrap_or_default();
                io::stderr().flush().unwrap_or_default();
                status_sender.send(ShardStatus::Failed(status)).unwrap();
            }
            return;
        }

        // We received a shutdown signal
        if *shutdown.lock().unwrap() {
            p.kill().unwrap();
            let _ = p.wait().unwrap();
            info!("Shard {rank} terminated");
            return;
        }

        // Shard is ready
        if !ready {
            if uds.exists() {
                info!("Shard {rank} ready in {:?}", start_time.elapsed());
                status_sender.send(ShardStatus::Ready).unwrap();
                ready = true;
            } else if wait_time.elapsed() > Duration::from_secs(10) {
                info!("Waiting for shard {rank} to be ready...");
                wait_time = Instant::now();
            }
        }
        sleep(Duration::from_millis(100));
    }
}

fn shutdown_shards(shutdown: Arc<Mutex<bool>>, shutdown_receiver: &mpsc::Receiver<()>) {
    info!("Shutting down shards");
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

fn write_termination_log(msg: &str) -> Result<(), io::Error> {
    // Writes a message to the termination log.
    // Creates the logfile if it doesn't exist.
    let mut f = File::options()
        .write(true)
        .create(true)
        .truncate(true)
        .open("/dev/termination-log")?;
    writeln!(f, "{}", msg)?;
    Ok(())
}

fn resolve_config_path(cache_path: PathBuf, model_name: &str, revision: Option<&str>) -> Result<String, io::Error> {
    let model_hf_cache_dir = cache_path.join(format!("models--{}", model_name.replace('/', "--")));
    let model_dir = if model_hf_cache_dir.try_exists()? {
        Some(model_hf_cache_dir)
    } else {
        None
    };
    if let Some(dir) = model_dir {
        let revision = revision.unwrap_or("main");
        let ref_path = dir.join("refs").join(revision);
        let revision = match ref_path.try_exists()? {
            true => fs::read_to_string(ref_path)?,
            false => revision.to_string(),
        };
        let config_path = dir.join("snapshots").join(&revision).join("config.json");
        if config_path.try_exists()? {
            Ok(config_path.to_string_lossy().into())
        } else {
            let message = format!(
                "Config file not found in local cache for model {model_name}, revision {revision}"
            );
            error!(message);
            Err(io::Error::new(ErrorKind::NotFound, message))
        }
    } else {
        // Try treating model_name as explicit model path
        let try_path = Path::new(&model_name).join("config.json");
        if try_path.try_exists()? {
            Ok(try_path.to_string_lossy().into())
        } else {
            let message = format!("Model {model_name} not found");
            error!(message);
            Err(io::Error::new(ErrorKind::NotFound, message))
        }
    }
}

/// Convert and save fast tokenizer via transformers `AutoTokenizer.from_pretrained`.
fn save_fast_tokenizer(
    model_name: &str,
    revision: Option<&str>,
    save_path: &str,
) -> Result<(), io::Error> {
    info!("Saving fast tokenizer for `{model_name}` to `{save_path}`");
    let model_name = model_name.escape_default();
    let revision = revision.map(|v| v.escape_default());
    let code = if let Some(revision) = revision {
        format!(
            "from transformers import AutoTokenizer; \
            AutoTokenizer.from_pretrained(\"{model_name}\", \
            revision=\"{revision}\", local_files_only=True).save_pretrained(\"{save_path}\")"
        )
    } else {
        format!(
            "from transformers import AutoTokenizer; \
            AutoTokenizer.from_pretrained(\"{model_name}\").save_pretrained(\"{save_path}\")"
        )
    };
    match Command::new("python").args(["-c", &code]).status() {
        Ok(status) => {
            if status.success() {
                Ok(())
            } else {
                let message = "Python process for tokenizer conversion failed".to_string();
                error!(
                    exit_code = ?status.code(),
                    message
                );
                Err(io::Error::new(ErrorKind::Other, message))
            }
        }
        Err(e) => {
            error!("Failed to launch python process for tokenizer conversion");
            Err(e)
        }
    }
}
