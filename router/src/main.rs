/// Text Generation Inference external gRPC server entrypoint
use clap::Parser;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use text_generation_client::ShardedClient;
use text_generation_router::server;
use tokenizers::Tokenizer;
use tracing::warn;
use text_generation_router::server::ServerRunArgs;

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
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
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
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

fn main() -> Result<(), std::io::Error> {
    // Get args
    let args = Args::parse();

    if args.json_output {
        tracing_subscriber::fmt().json().with_current_span(false).init();
    } else {
        tracing_subscriber::fmt().compact().init();
    }

    if args.validation_workers == 0 {
        panic!("validation_workers must be > 0");
    }

    if args.tls_key_path.is_some() != args.tls_cert_path.is_some() {
        panic!("tls: must provide both cert and key")
    }

    if args.tls_client_ca_cert_path.is_some() && args.tls_cert_path.is_none() {
        panic!("tls: cannot provide client ca cert without keypair")
    }

    // Instantiate tokenizer
    let mut tokenizer = Tokenizer::from_file(args.tokenizer_path)
        .expect("Problem loading tokenizer for model");

    if let Some(tp) = tokenizer.get_truncation() {
        if tp.max_length < args.max_sequence_length {
            warn!(
                "Ignoring fast tokenizer truncation configuration with max_length {}, \
                max_sequence_length is set to {}",
                tp.max_length, args.max_sequence_length,
            );
        }
    }
    tokenizer.with_truncation(None).with_padding(None);

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // Instantiate sharded client from the master unix socket
            let mut sharded_client = ShardedClient::connect_uds(args.master_shard_uds_path)
                .await
                .expect("Could not connect to server");
            // Clear the cache; useful if the webserver rebooted
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            let grpc_addr = SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), args.grpc_port
            );

            // Binds on localhost
            let addr = SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), args.port
            );

            // Run server
            server::run(ServerRunArgs {
                max_concurrent_requests: args.max_concurrent_requests,
                max_sequence_length: args.max_sequence_length,
                max_new_tokens: args.max_new_tokens,
                max_batch_size: args.max_batch_size,
                max_batch_weight: args.max_batch_weight,
                max_prefill_weight: args.max_prefill_weight,
                max_waiting_tokens: args.max_waiting_tokens,
                client: sharded_client,
                tokenizer,
                validation_workers: args.validation_workers,
                addr,
                grpc_addr,
                tls_key_pair: args.tls_cert_path.map(|cp| (cp, args.tls_key_path.unwrap())),
                tls_client_ca_cert: args.tls_client_ca_cert_path,
                output_special_tokens: args.output_special_tokens,
            })
            .await;
            Ok(())
        })
}
