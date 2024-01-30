/// Single shard Client
use std::time::Duration;
use crate::pb::generate::v1::text_generation_service_client::TextGenerationServiceClient;
use crate::pb::generate::v1::*;
use crate::{ClientError, Result};
use tonic::transport::{Channel, Uri};
use tracing::*;
use crate::pb::generate::v1::model_info_response::ModelType;

const PREFIX_LOOKUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Text Generation Inference gRPC client
#[derive(Debug, Clone)]
pub struct Client {
    stub: TextGenerationServiceClient<Channel>,
}

pub type GenerateTokenResponse = (Vec<Token>, Vec<InputTokens>, Vec<GenerateError>, u64, Duration);

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {});
        let response = self
            .stub
            .service_discovery(request)
            .instrument(info_span!("service_discovery"))
            .await?;
        let urls = response
            .into_inner()
            .urls
            .into_iter()
            // Remove unix socket prefix
            .map(|url| match url.strip_prefix("unix://") {
                None => url,
                Some(stripped_url) => stripped_url.to_string(),
            })
            .collect();
        Ok(urls)
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest {});
        self.stub
            .clear_cache(request)
            .instrument(info_span!("clear_cache"))
            .await?;
        Ok(())
    }

    /// Get shard model info
    #[instrument(skip(self))]
    pub async fn model_info(&mut self) -> Result<(ModelType, u32, bool, MemoryScalingModel)> {
        let request = tonic::Request::new(ModelInfoRequest {});
        let response = self.stub
            .model_info(request)
            .instrument(info_span!("model_info"))
            .await?
            .into_inner();
        ModelType::try_from(response.model_type)
            .map(|mt| (
                mt,
                response.eos_token,
                response.batch_padding,
                response.memory_scaling_model.unwrap(),
            ))
            .map_err(|_| ClientError::Generation("Unrecognized model type".to_string()))
    }

    /// Get model health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {});
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    /// Get shard model info
    #[instrument(skip(self))]
    pub async fn prefix_lookup(&mut self, prefix_id: String) -> Result<u32> {
        let mut request = tonic::Request::new(
            PrefixLookupRequest { prefix_id }
        );
        request.set_timeout(PREFIX_LOOKUP_TIMEOUT);
        let response = self.stub
            .prefix_lookup(request)
            .instrument(info_span!("prefix_lookup"))
            .await?
            .into_inner();
        Ok(response.prefix_length)
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns first generated token for each request in the batch, id of the next cached batch,
    /// and input token info if requested
    #[instrument(skip(self))]
    pub async fn prefill(
        &mut self, batch: Batch, to_prune: Vec<CachedBatch>,
    ) -> Result<GenerateTokenResponse> {
        let request = tonic::Request::new(PrefillRequest{
            batch: Some(batch), to_prune,
        });
        let response = self
            .stub
            .prefill(request)
            .instrument(info_span!("generate"))
            .await?
            .into_inner();
        let result = response
            .result
            .ok_or_else(|| ClientError::Generation("Unexpected empty response".into()))?;
        Ok((
            result.output_tokens,
            response.input_tokens,
            result.errors,
            result.batch_id,
            Duration::from_nanos(result.forward_time_ns),
        ))
    }

    /// Generate one token for each request in the given cached batch(es)
    ///
    /// Returns next generated token of each request in the batches and id of the next cached batch
    #[instrument(skip(self))]
    pub async fn next_token(
        &mut self, batches: Vec<CachedBatch>,
    ) -> Result<Option<GenerateTokenResponse>> {
        let request = tonic::Request::new(
            NextTokenRequest { batches }
        );
        let response = self
            .stub
            .next_token(request)
            .instrument(info_span!("generate_with_cache"))
            .await?
            .into_inner();
        Ok(response.result.map(|result| (
            result.output_tokens,
            vec![],
            result.errors,
            result.batch_id,
            Duration::from_nanos(result.forward_time_ns),
        )))
    }
}
