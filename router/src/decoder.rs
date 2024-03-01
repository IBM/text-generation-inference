use std::{collections::VecDeque, mem::take, slice::from_ref};

use tokenizers::{
    DecoderWrapper::{ByteLevel, Metaspace, Sequence, WordPiece, BPE, CTC},
    Error, Tokenizer,
};
use unicode_segmentation::UnicodeSegmentation;

use crate::batcher::{InferError, InferError::DetokenizationError};

// Note this currently includes a hack where if a Sequence-type decoder is encountered
// we assume that it's the Llama tokenizer and apply both byte-level and first-diff logic

pub(crate) struct Decoder {
    tokenizer: Tokenizer,
    single_tok_id: u32,
    single_tok: String,
    skip_special_toks: bool,
    pub(crate) seq2seq: bool,
    pub(crate) eos_token_id: u32,
}

impl Decoder {
    pub(crate) fn new(
        tokenizer: Tokenizer,
        seq2seq: bool,
        eos_token_id: u32,
        skip_special_toks: bool,
    ) -> Decoder {
        let prefix_id = *tokenizer
            .encode("A", false)
            .expect("Tokenizer setup error")
            .get_ids()
            .first()
            .unwrap();
        Decoder {
            single_tok_id: prefix_id,
            single_tok: tokenizer.decode(from_ref(&prefix_id), false).unwrap(),
            tokenizer,
            seq2seq,
            eos_token_id,
            skip_special_toks,
        }
    }

    fn decode_full(&self, mut ids: &[u32]) -> Result<String, InferError> {
        if !self.skip_special_toks && ids.last() == Some(&self.eos_token_id) {
            ids = &ids[..(ids.len() - 1)];
        }
        self.tokenizer
            .decode(ids, self.skip_special_toks)
            .map_err(Error::into)
    }

    pub(crate) fn id_to_token(&self, id: u32) -> String {
        self.tokenizer.id_to_token(id).unwrap_or_default()
    }

    pub(crate) fn decode(
        &self,
        mut ids: Vec<u32>,
        first: bool,
        last: bool,
    ) -> Result<String, InferError> {
        let decoder = self.tokenizer.get_decoder();
        if (first && self.seq2seq)
            || (last && matches![decoder, Some(BPE(_))])
            || matches![decoder, Some(ByteLevel(_) | CTC(_))]
        {
            // In these cases we don't need to do anything special for "continuation"
            let mut text = self.decode_full(&ids)?;
            text.truncate(text.trim_end_matches('�').len()); // Avoid add'l allocation
            return Ok(text);
        }
        // How we handle continuation depends on the specific decoder's behaviour,
        // see each one's implementation of decode_chain in the tokenizers library.
        match self.tokenizer.get_decoder() {
            Some(Metaspace(_) | WordPiece(_) | Sequence(_)) => {
                // For these, the first token in the sequence is treated differently,
                // so we add and then strip a placeholder token.
                ids.insert(0, self.single_tok_id);
                let result = self.decode_full(&ids)?;
                let mut text = result
                    .strip_prefix(&self.single_tok)
                    .ok_or_else(|| DetokenizationError("Unexpected".into()))?
                    .to_string();
                text.truncate(text.trim_end_matches('�').len()); // Avoid add'l allocation
                Ok(text)
            }
            Some(BPE(_)) => {
                ids.push(self.single_tok_id);
                let result = self.decode_full(&ids)?;
                Ok(result
                    .strip_suffix(&self.single_tok)
                    .ok_or_else(|| DetokenizationError("Unexpected".into()))?
                    .to_string())
            }
            None => {
                // Just prepend a space
                Ok(format!(" {}", self.decode_full(&ids)?))
            }
            Some(tok) => Err(DetokenizationError(format!(
                "Unsupported tokenizer type: {:?}",
                tok
            ))),
        }
    }

    pub(crate) fn decode_ref(
        &self,
        ids: &[u32],
        first: bool,
        last: bool,
    ) -> Result<String, InferError> {
        let decoder = self.tokenizer.get_decoder();
        if (first && self.seq2seq)
            || (last && matches![decoder, Some(BPE(_))])
            || matches![decoder, Some(ByteLevel(_) | CTC(_))]
        {
            // In these cases we don't need to do anything special for "continuation"
            let mut text = self.decode_full(ids)?;
            text.truncate(text.trim_end_matches('�').len()); // Avoid add'l allocation
            return Ok(text);
        }
        // How we handle continuation depends on the specific decoder's behaviour,
        // see each one's implementation of decode_chain in the tokenizers library.
        match self.tokenizer.get_decoder() {
            Some(Metaspace(_) | WordPiece(_) | Sequence(_)) => {
                // For these, the first token in the sequence is treated differently,
                // so we add and then strip a placeholder token.
                let ids = [from_ref(&self.single_tok_id), ids].concat();
                let result = self.decode_full(&ids)?;
                let mut text = result
                    .strip_prefix(&self.single_tok)
                    .ok_or_else(|| DetokenizationError("Unexpected".into()))?
                    .to_string();
                text.truncate(text.trim_end_matches('�').len()); // Avoid add'l allocation
                Ok(text)
            }
            Some(BPE(_)) => {
                let ids = [ids, from_ref(&self.single_tok_id)].concat();
                // ids.push(self.single_tok_id);
                let result = self.decode_full(&ids)?;
                Ok(result
                    .strip_suffix(&self.single_tok)
                    .ok_or_else(|| DetokenizationError("Unexpected".into()))?
                    .to_string())
            }
            None => {
                // Just prepend a space
                Ok(format!(" {}", self.decode_full(ids)?))
            }
            Some(tok) => Err(DetokenizationError(format!(
                "Unsupported tokenizer type: {:?}",
                tok
            ))),
        }
    }
}

#[derive(Debug)]
pub(crate) enum IncrementalDecoderWrapper {
    ByteLevel(IncrementalBLDecoder),        // For ByteLevel
    FirstDiff(IncrementalFirstDiffDecoder), // For Metaspace, WordPiece, None
    LastDiff(IncrementalLastDiffDecoder),   // For BPE
    DeDup(IncrementalDeDupDecoder),         // For CTE
    Buffered(Box<BufferedIncrementalDecoder>),
}

impl IncrementalDecoderWrapper {
    /// If hold_back_bytes is > 0, at least that number of bytes will be buffered
    /// in addition to the last token
    pub(crate) fn for_decoder(decoder: &Decoder, is_start: bool, hold_back_bytes: usize) -> Self {
        let idecoder = match decoder.tokenizer.get_decoder() {
            Some(ByteLevel(_)) => {
                Self::ByteLevel(IncrementalBLDecoder::new(false, false, hold_back_bytes))
            }
            Some(Sequence(_)) => {
                Self::ByteLevel(IncrementalBLDecoder::new(true, is_start, hold_back_bytes))
            }
            Some(BPE(_)) => Self::LastDiff(IncrementalLastDiffDecoder {
                output: String::new(),
                next_id: None,
            }),
            Some(CTC(_)) => Self::DeDup(IncrementalDeDupDecoder {
                output: String::new(),
                last_id: None,
            }),
            _ => Self::FirstDiff(IncrementalFirstDiffDecoder {
                output: String::new(),
                first: is_start,
            }),
        };

        match idecoder {
            // These incremental decoder types require extra buffering to hold back bytes
            Self::FirstDiff(_) | Self::LastDiff(_) | Self::DeDup(_) if hold_back_bytes != 0 => {
                Self::Buffered(Box::new(BufferedIncrementalDecoder {
                    delegate: idecoder,
                    hold_back_bytes,
                    offset_buffer: VecDeque::new(),
                    sent_up_to: 0,
                }))
            }
            // IncrementalBLDecoder handles the hold_back_bytes buffering itself
            _ => idecoder,
        }
    }
}

impl IncrementalDecoder for IncrementalDecoderWrapper {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError> {
        match self {
            Self::ByteLevel(d) => d.next(token, decoder),
            Self::FirstDiff(d) => d.next(token, decoder),
            Self::LastDiff(d) => d.next(token, decoder),
            Self::DeDup(d) => d.next(token, decoder),
            Self::Buffered(d) => d.next(token, decoder),
        }
    }

    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        decoder: &Decoder,
    ) -> Result<String, InferError> {
        match self {
            Self::ByteLevel(d) => d.flush(max_total_len, decoder),
            Self::FirstDiff(d) => d.flush(max_total_len, decoder),
            Self::LastDiff(d) => d.flush(max_total_len, decoder),
            Self::DeDup(d) => d.flush(max_total_len, decoder),
            Self::Buffered(d) => d.flush(max_total_len, decoder),
        }
    }

    fn output(&self) -> &str {
        match self {
            Self::ByteLevel(d) => d.output(),
            Self::FirstDiff(d) => d.output(),
            Self::LastDiff(d) => d.output(),
            Self::DeDup(d) => d.output(),
            Self::Buffered(d) => d.output(),
        }
    }

    fn into_string(self) -> String {
        match self {
            Self::ByteLevel(d) => d.into_string(),
            Self::FirstDiff(d) => d.into_string(),
            Self::LastDiff(d) => d.into_string(),
            Self::DeDup(d) => d.into_string(),
            Self::Buffered(d) => d.into_string(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalFirstDiffDecoder {
    output: String,
    first: bool,
}

impl IncrementalDecoder for IncrementalFirstDiffDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError> {
        let text = decoder.decode_ref(from_ref(&token), self.first, false)?;
        self.first = false;
        self.output += &text;
        let bytes_added = text.len();
        Ok((text, bytes_added))
    }

    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        _decoder: &Decoder,
    ) -> Result<String, InferError> {
        if let Some(max_len) = max_total_len {
            self.output.truncate(max_len);
        }
        Ok(String::new())
    }

    fn output(&self) -> &str {
        &self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalLastDiffDecoder {
    output: String,
    next_id: Option<u32>,
}

impl IncrementalDecoder for IncrementalLastDiffDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError> {
        let text = self.next_id.map_or_else(
            || Ok(String::new()),
            |ref id| decoder.decode_ref(from_ref(id), true, false),
        )?;
        self.next_id = Some(token);
        self.output += &text;
        let bytes_added = text.len();
        Ok((text, bytes_added))
    }

    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        decoder: &Decoder,
    ) -> Result<String, InferError> {
        let mut text = self.next_id.map_or_else(
            || Ok(String::new()),
            |ref id| decoder.decode_full(from_ref(id)),
        )?;
        self.next_id = None;
        self.output += &text;
        if let Some(max_len) = max_total_len {
            let diff = self.output.len().saturating_sub(max_len);
            if diff != 0 {
                text.truncate(text.len().saturating_sub(diff));
            }
            self.output.truncate(max_len);
        }
        Ok(text)
    }

    fn output(&self) -> &str {
        &self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalDeDupDecoder {
    output: String,
    last_id: Option<u32>,
}

impl IncrementalDecoder for IncrementalDeDupDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError> {
        if self.last_id.map(|id| id == token).unwrap_or(false) {
            return Ok((String::new(), 0));
        }
        self.last_id = Some(token);
        let text = decoder.decode_full(from_ref(&token))?;
        self.output += &text;
        let bytes_added = text.len();
        Ok((text, bytes_added))
    }

    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        _decoder: &Decoder,
    ) -> Result<String, InferError> {
        if let Some(max_len) = max_total_len {
            self.output.truncate(max_len);
        }
        Ok(String::new())
    }

    fn output(&self) -> &str {
        &self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

#[derive(Debug)]
pub(crate) struct BufferedIncrementalDecoder {
    delegate: IncrementalDecoderWrapper,
    hold_back_bytes: usize,
    offset_buffer: VecDeque<usize>,
    sent_up_to: usize,
}

impl IncrementalDecoder for BufferedIncrementalDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError> {
        let (text, added_bytes) = self.delegate.next(token, decoder)?;
        if text.is_empty() {
            return Ok((text, added_bytes));
        }
        let len = self.delegate.output().len();
        self.offset_buffer.push_back(len);

        let cutoff = len.saturating_sub(self.hold_back_bytes);
        if cutoff != 0 {
            let mut next_index = None;
            while self.offset_buffer.front().unwrap() <= &cutoff {
                next_index = Some(self.offset_buffer.pop_front().unwrap());
            }

            if let Some(next_index) = next_index {
                let from = self.sent_up_to;
                self.sent_up_to = next_index;
                return Ok((self.output()[from..next_index].to_string(), added_bytes));
            }
        }
        Ok((String::new(), added_bytes))
    }

    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        decoder: &Decoder,
    ) -> Result<String, InferError> {
        self.delegate.flush(max_total_len, decoder)?;
        if self.sent_up_to < self.output().len() {
            Ok(self.output()[self.sent_up_to..].to_string())
        } else {
            Ok(String::new())
        }
    }

    fn output(&self) -> &str {
        self.delegate.output()
    }

    fn into_string(self) -> String {
        self.delegate.into_string()
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalBLDecoder {
    id_buffer: Vec<u32>,
    str_buffer: String,
    output: String,
    first_diff: bool,
    first: bool,
    hold_back_bytes: usize,
}

impl IncrementalBLDecoder {
    fn new(first_diff: bool, first: bool, hold_back_bytes: usize) -> Self {
        Self {
            id_buffer: vec![],
            str_buffer: String::new(),
            output: String::new(),
            first_diff,
            first,
            hold_back_bytes,
        }
    }
}

impl IncrementalDecoder for IncrementalBLDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError> {
        self.id_buffer.push(token);
        let buffer = &*self.id_buffer;
        let text = if self.first_diff && !self.first {
            // Prepend placeholder token to avoid first-token differences
            let buffer = [from_ref(&decoder.single_tok_id), buffer].concat();
            let result = decoder.decode_full(&buffer)?;
            result
                .strip_prefix(&decoder.single_tok)
                .ok_or_else(|| DetokenizationError("Unexpected".into()))?
                .to_string()
        } else {
            self.first = false;
            decoder.decode_full(buffer)?
        };
        // Defer decoding until we have enough bytes for complete UTF-8
        let mut added_bytes = 0;
        if !text.ends_with('�') {
            self.output.push_str(&text);
            added_bytes = text.len();
            if self.str_buffer.is_empty() {
                self.str_buffer = text;
            } else {
                self.str_buffer.push_str(&text);
            }
            self.id_buffer.clear();

            // Keep at least hold_back_bytes in the str_buffer
            let cutoff = match self.hold_back_bytes {
                0 => self.str_buffer.len(),
                n => self.str_buffer.len().saturating_sub(n + added_bytes),
            };
            if cutoff != 0 {
                // Ensure that we return full grapheme clusters
                for (idx, _) in self.str_buffer.grapheme_indices(true).rev() {
                    if idx <= cutoff && idx > 0 {
                        return Ok((self.str_buffer.drain(..idx).collect(), added_bytes));
                    }
                }
            }
        }
        Ok((String::new(), added_bytes))
    }
    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        decoder: &Decoder,
    ) -> Result<String, InferError> {
        if !self.id_buffer.is_empty() {
            let last = decoder.decode_full(&self.id_buffer)?;
            let last = last.trim_end_matches('�');
            self.output += last;
            self.str_buffer.push_str(last);
            self.id_buffer.clear();
        }
        if let Some(max_len) = max_total_len {
            let diff = self.output.len().saturating_sub(max_len);
            if diff != 0 {
                self.output.truncate(max_len);
                self.str_buffer
                    .truncate(self.str_buffer.len().saturating_sub(diff));
            }
        }
        Ok(take(&mut self.str_buffer))
    }

    fn output(&self) -> &str {
        &self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

pub(crate) trait IncrementalDecoder {
    /// Consume next token id and return tuple of ready-to-be-returned text and number of bytes
    /// by which the length of output() has increased by (which is not necessarily text.len())
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<(String, usize), InferError>;
    /// Flush any remaining buffered tokens/text to the output() string, and return string of any
    /// not-yet-returned text suffix.
    /// The output will be truncated to max_total_len if specified, and this will also be reflected
    /// in the returned text. If text past max_total_len has already been returned by next(), an
    /// empty string will be returned.
    fn flush(
        &mut self,
        max_total_len: Option<usize>,
        decoder: &Decoder,
    ) -> Result<String, InferError>;
    /// A ref to the current accumulated output string
    fn output(&self) -> &str;
    /// Return the current accumulated output string, consuming this incremental decoder
    fn into_string(self) -> String;
}

impl From<Error> for InferError {
    fn from(err: Error) -> Self {
        DetokenizationError(err.to_string())
    }
}
