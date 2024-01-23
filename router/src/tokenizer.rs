use std::fmt::{Debug, Formatter};
use flume::{Receiver, Sender};
use tokenizers::{Encoding, Tokenizer};
use tokio::sync::oneshot;

type TokenizationRequest = (
    String, bool, oneshot::Sender<Result<(String, usize, Option<Encoding>), tokenizers::Error>>
);


#[derive(Clone)]
pub(crate) struct AsyncTokenizer {
    sender: Sender<TokenizationRequest>,
}

impl Debug for AsyncTokenizer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("AsyncTokenizer")
    }
}

/// Uses pool of tokenizer threads to provide async tokenization methods
impl AsyncTokenizer {
    pub(crate) fn new(tokenizer: &Tokenizer, workers: usize) -> Self {
        let (sender, receiver) = flume::unbounded();
        for _ in 0..workers {
            let tokenizer = tokenizer.clone();
            let receiver = receiver.clone();
            tokio::task::spawn_blocking(
                move || tokenization_worker(tokenizer, receiver)
            );
        }
        Self { sender }
    }

    /// Tokenize the supplied string
    pub(crate) async fn tokenize(
        &self, input: String, include_encoding: bool,
    ) -> Result<(String, usize, Option<Encoding>), tokenizers::Error> {
        let (sender, receiver) = oneshot::channel();
        // unwrap is safe - receiver can only dropped after sender is dropped
        self.sender.send_async((input, include_encoding, sender)).await.unwrap();
        // unwrap is safe - sender is in scope
        receiver.await.unwrap()
    }
}

fn tokenization_worker(
    tokenizer: Tokenizer,
    receiver: Receiver<TokenizationRequest>,
) {
    while let Ok((input, with_encoding, sender)) = receiver.recv() {
        let result = tokenizer.encode(&input[..], true)
            .map(|encoding| (input, encoding.len(), with_encoding.then_some(encoding)));
        sender.send(result).unwrap_or_default();
    }
}