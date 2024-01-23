use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::time::Instant;
use text_generation_client::{Batch, NextTokenChooserParameters, Request, ShardedClient};

const TEST_INPUT: &str = "liveness";

#[derive(Clone, Debug)]
pub(crate) struct Health {
    client: ShardedClient,
    generation_health: Arc<AtomicBool>,
    test_input_tokens: u32,
}

impl Health {
    pub(crate) fn new(
        client: ShardedClient, generation_health: Arc<AtomicBool>, tokenizer: &Tokenizer
    ) -> Self {
        Self {
            client,
            generation_health,
            test_input_tokens: tokenizer.encode(TEST_INPUT, true)
                .expect("Tokenization error").len() as u32
        }
    }

    pub(crate) async fn check(&mut self) -> bool {
        let generation_healthy = self.generation_health.load(Ordering::SeqCst);

        let mut guard = Guard{ prefill: !generation_healthy, start_time: Some(Instant::now()) };

        let ok = if generation_healthy {
            // Generation is healthy, we only check that the shards are answering gRPC calls
            self.client.health().await
                .map_err(|err| tracing::error!("Basic shard healthcheck error: {err}"))
                .is_ok()
        } else {
            // Generation is unhealthy or have not sent any generation request yet

            // Dummy batch of 1 token and 1 generated token
            let liveness_request = Request {
                // Using this id will ensure the batch is not cached in the shards
                id: u64::MAX,
                prefix_id: String::new(),
                inputs: TEST_INPUT.to_string(),
                input_length: self.test_input_tokens,
                truncate: false,
                max_output_length: 1,
                parameters: Some(NextTokenChooserParameters {
                    ..Default::default()
                }),
                stream_response: false,
                details: None,
            };
            let batch = Batch {
                id: u64::MAX,
                requests: vec![liveness_request],
                total_tokens: 1,
            };
            // Skips the queue, but will still be serialized behind in-flight prefill/next_token requests
            let value = self.client.prefill(batch, vec![]).await
                .map_err(|err| tracing::error!("Prefill healthcheck error: {err}"))
                .is_ok();
            // Update generation health
            self.generation_health.store(value, Ordering::SeqCst);
            value
        };
        guard.start_time = None;
        ok
    }
}

struct Guard {
    prefill: bool,
    start_time: Option<Instant>, // None once completed
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(start_time) = self.start_time {
            tracing::warn!(
                "Healthcheck request cancelled during {} check after {}ms",
                if self.prefill { "prefill" } else { "basic shard" },
                start_time.elapsed().as_millis(),
            )
        }
    }
}
