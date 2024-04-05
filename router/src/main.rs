use std::{
    fs::File,
    io,
    io::Write,
    net::{IpAddr, Ipv4Addr, SocketAddr},
};

/// Text Generation Inference external gRPC server entrypoint
use clap::Parser;
use opentelemetry::{
    global,
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::{Resource, trace};
use opentelemetry_sdk::trace::Sampler;
use text_generation_client::ShardedClient;
use text_generation_router::{server, server::ServerRunArgs};
use tokenizers::Tokenizer;
use tracing::warn;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "512", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2048", long, env)]
    max_sequence_length: usize,
    #[clap(default_value = "1024", long, env)]
    max_new_tokens: usize,
    #[clap(default_value = "12", long, env)]
    max_batch_size: usize,
    #[clap(default_value = "0.2", long, env)]
    max_prefill_padding: f32,
    #[clap(default_value = "24", long, env)]
    max_waiting_tokens: usize,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "8033", long, short, env)]
    grpc_port: u16,
    #[clap(default_value = "/tmp/text-generation-0", long, env)]
    master_shard_uds_path: String,
    #[clap(long, env)]
    tokenizer_path: String,
    #[clap(default_value = None, long, env)]
    tokenization_workers: Option<usize>,
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
    #[clap(long, env)]
    default_include_stop_seqs: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
}

fn main() -> Result<(), std::io::Error> {
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

    // Get args
    let args = Args::parse();

    // Validate args
    validate_args(&args);

    // Instantiate tokenizer
    let mut tokenizer =
        Tokenizer::from_file(args.tokenizer_path).expect("Problem loading tokenizer for model");

    if let Some(tp) = tokenizer.get_truncation() {
        if tp.max_length < args.max_sequence_length {
            warn!(
                "Ignoring fast tokenizer truncation configuration with max_length {}, \
                max_sequence_length is set to {}",
                tp.max_length, args.max_sequence_length,
            );
        }
    }
    tokenizer.with_truncation(None).unwrap().with_padding(None);

    tracing::info!("Token decoder: {:?}", tokenizer.get_decoder());

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            init_logging(args.otlp_endpoint, args.json_output);
            // Instantiate sharded client from the master unix socket
            let mut sharded_client = ShardedClient::connect_uds(args.master_shard_uds_path)
                .await
                .expect("Could not connect to server");

            let tokenization_workers = args.tokenization_workers.unwrap_or_else(|| {
                // Determine number of threads to use for tokenization based on number of cores
                let num_cpus = num_cpus::get();
                let num_shards = sharded_client.shard_count();
                num_cpus.checked_sub(num_shards).unwrap_or(1).clamp(1, 8)
            });

            tracing::info!("Using pool of {tokenization_workers} threads for tokenization");

            // Clear the cache; useful if this process rebooted
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            let grpc_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), args.grpc_port);

            // Binds on localhost
            let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), args.port);

            // Run server
            server::run(ServerRunArgs {
                max_concurrent_requests: args.max_concurrent_requests,
                max_sequence_length: args.max_sequence_length,
                max_new_tokens: args.max_new_tokens,
                max_batch_size: args.max_batch_size,
                max_prefill_padding: args.max_prefill_padding,
                max_waiting_tokens: args.max_waiting_tokens,
                client: sharded_client,
                tokenizer,
                tokenization_workers,
                addr,
                grpc_addr,
                tls_key_pair: args
                    .tls_cert_path
                    .map(|cp| (cp, args.tls_key_path.unwrap())),
                tls_client_ca_cert: args.tls_client_ca_cert_path,
                output_special_tokens: args.output_special_tokens,
                default_include_stop_seqs: args.default_include_stop_seqs,
            })
            .await;
            Ok(())
        })
}

fn validate_args(args: &Args) {
    if args.tokenization_workers == Some(0) {
        panic!("tokenization_workers must be > 0");
    }

    if args.max_concurrent_requests == 0 {
        panic!("max_concurrent_requests must be > 0");
    }

    if args.tls_key_path.is_some() != args.tls_cert_path.is_some() {
        panic!("tls: must provide both cert and key")
    }

    if args.tls_client_ca_cert_path.is_some() && args.tls_cert_path.is_none() {
        panic!("tls: cannot provide client ca cert without keypair")
    }

    if args.max_prefill_padding < 0.0 || args.max_prefill_padding > 1.0 {
        panic!(
            "max_prefill_padding ({}) must be a percentage in the range [0.0, 1.0]",
            args.max_prefill_padding,
        )
    }

    if args.max_new_tokens < 1 {
        panic!("max_new_tokens ({}) at least 1", args.max_new_tokens)
    }

    if args.max_sequence_length < 2 {
        panic!(
            "max_sequence_length ({}) must be at least 2 (1 input + 1 output)",
            args.max_sequence_length,
        )
    }
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

fn init_logging(otlp_endpoint: Option<String>, json_output: bool) {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let fmt_layer = match json_output {
        true => fmt_layer.json().flatten_event(true).boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "text-generation-inference.router",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            axum_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
}
