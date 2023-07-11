/// Multi shard Client
use crate::{ClientError, GenerateError, Result};
use crate::{Batch, Client, HealthResponse, Token};
use futures::future::join_all;
use tokio::runtime::Handle;
use tokio::sync::{broadcast, mpsc};
use tonic::transport::Uri;
use crate::pb::generate::v1::{CachedBatch, InputTokens};
use crate::pb::generate::v1::model_info_response::ModelType;
use crate::sharded_client::Request::{NextToken, Prefill};

#[derive(Clone, Debug)]
enum Request {
    Prefill(Batch, Vec<CachedBatch>),
    NextToken(Vec<CachedBatch>),
}

/// Text Generation Inference gRPC multi client
#[derive(Debug)]
pub struct ShardedClient {
    clients: Vec<Client>,
    sender: broadcast::Sender<(Request, mpsc::Sender<
        Result<Option<(Vec<Token>, Vec<InputTokens>, Vec<GenerateError>, u64)>>
    >)>,
    handle: Handle,
}

impl Clone for ShardedClient {
    fn clone(&self) -> Self {
        Self::new(self.clients.clone())
    }
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        let (sender, _) = broadcast::channel::<(Request, mpsc::Sender<_>)>(16);

        // Spawn a task for each shard
        for mut client in clients.clone() {
            let mut receiver: broadcast::Receiver<(Request, _)> = sender.subscribe();
            tokio::spawn(async move {
                while let Ok((request , response_chan)) = receiver.recv().await {
                    let result = match request {
                        Prefill(batch, to_prune) =>
                            client.prefill(batch, to_prune).await.map(|r| Some(r)),
                        NextToken(batches) =>
                            client.next_token(batches).await,
                    };
                    response_chan.try_send(result).unwrap_or_default();
                }
            });
        }

        Self { clients, sender, handle: Handle::current() }
    }

    /// Create a new ShardedClient from a master client. The master client will communicate with
    /// the other shards and returns all uris/unix sockets with the `service_discovery` gRPC method.
    async fn from_master_client(mut master_client: Client) -> Result<Self> {
        // Get all uris/unix sockets from the master client
        let uris = master_client.service_discovery().await.unwrap();
        let futures = uris.into_iter().map(Client::connect_uds);
        let clients: Result<Vec<Client>> = join_all(futures).await.into_iter().collect();
        Ok(Self::new(clients?))
    }

    /// Returns a client connected to the given uri
    pub async fn connect(uri: Uri) -> Result<Self> {
        let master_client = Client::connect(uri).await?;
        Self::from_master_client(master_client).await
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let master_client = Client::connect_uds(path).await?;
        Self::from_master_client(master_client).await
    }

    /// GRPC health check
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.health())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns first generated token for each request in the batch, id of the next cached batch,
    /// and input token info if requested.
    ///
    /// Optionally prunes existing batches first to maximize available memory
    pub async fn prefill(
        &mut self, batch: Batch, to_prune: Vec<CachedBatch>,
    ) -> Result<Option<(Vec<Token>, Vec<InputTokens>, Vec<GenerateError>, u64)>> {
        if batch.requests.is_empty() {
            return Ok(None);
        }
        let (tx, mut rx) = mpsc::channel(1);
        self.sender.send((Prefill(batch, to_prune), tx))
            .map_err(|e| ClientError::Generation(e.to_string()))?;
        rx.recv().await.ok_or_else(|| ClientError::Connection("client closed".to_string()))?
    }

    /// Generate one token for each request in the given cached batch
    ///
    /// Returns next generated token of each request in the batches and id of the next cached batch
    pub async fn next_token(
        &mut self,
        batches: Vec<CachedBatch>,
    ) -> Result<Option<(Vec<Token>, Vec<InputTokens>, Vec<GenerateError>, u64)>> {
        let (tx, mut rx) = mpsc::channel(1);
        self.sender.send((NextToken(batches), tx))
            .map_err(|e| ClientError::Generation(e.to_string()))?;
        rx.recv().await.ok_or_else(|| ClientError::Connection("client closed".to_string()))?
    }

    /// Clear the past generations cache
    pub async fn clear_cache(&mut self) -> Result<()> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.clear_cache())
            .collect();
        join_all(futures).await.into_iter().collect()
    }

    /// Get length of prompt prefix - verifies existence and populates cache
    pub fn prefix_lookup(&mut self, prefix_id: &String) -> Result<usize> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.prefix_lookup(prefix_id.clone()))
            .collect();
        let v: Vec<Result<u32>> = self.handle.block_on(join_all(futures));

        //let v: Vec<Result<u32>> = join_all(futures).await;
        v.first().unwrap().clone().map(|l| l as usize)
    }

    /// Get shard model info
    pub async fn model_info(&mut self) -> Result<(bool, u32, bool)> {
        self.clients[0].model_info().await
            .map(|(mt, eos, bpad)| (mt == ModelType::Seq2seqLm, eos, bpad))
    }
}
