//! Text Generation gRPC client library

mod client;
#[allow(clippy::derive_partial_eq_without_eq)]
mod pb;
mod sharded_client;

pub use client::Client;
pub use pb::generate::v1::{
    Batch, Token, InputTokens, NextTokenChooserParameters, RequestedDetails,
    Request, StopSequence, CachedBatch, RequestsStatus, GenerateError,
    HealthResponse,
};
pub use pb::generate::v1::next_token_chooser_parameters::LengthPenalty;
pub use sharded_client::ShardedClient;
pub use client::GenerateTokenResponse;
use thiserror::Error;
use tonic::{Code, transport};
use tonic::Status;

#[derive(Error, Debug, Clone)]
pub enum ClientError {
    #[error("Could not connect to Text Generation server: {0}")]
    Connection(String),
    #[error("{0}")]
    Generation(String),
    #[error("GPU out of memory")]
    OutOfMemory(),
}

impl From<Status> for ClientError {
    fn from(err: Status) -> Self {
        match err.code() {
            Code::ResourceExhausted => Self::OutOfMemory(),
            Code::Unavailable => Self::Connection(err.message().to_string()),
            _ => Self::Generation(err.message().to_string())
        }
    }
}

impl From<transport::Error> for ClientError {
    fn from(err: transport::Error) -> Self {
        Self::Connection(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ClientError>;
