//! Inspired by: https://github.com/open-telemetry/opentelemetry-rust gRPC examples

use opentelemetry::{global, propagation::Extractor};
use tonic::Request;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

struct MetadataExtractor<'a>(&'a tonic::metadata::MetadataMap);

impl<'a> Extractor for MetadataExtractor<'a> {
    /// Get a value for a key from the MetadataMap.  If the value can't be converted to &str, returns None
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|metadata| metadata.to_str().ok())
    }

    /// Collect all the keys from the MetadataMap.
    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .map(|key| match key {
                tonic::metadata::KeyRef::Ascii(v) => v.as_str(),
                tonic::metadata::KeyRef::Binary(v) => v.as_str(),
            })
            .collect::<Vec<_>>()
    }
}

/// Extract context from metadata and set as current span's context
fn extract(metadata: &tonic::metadata::MetadataMap) {
    let parent_cx =
        global::get_text_map_propagator(|prop| prop.extract(&MetadataExtractor(metadata)));
    Span::current().set_parent(parent_cx);
}

pub trait ExtractTelemetryContext {
    fn extract_context(self) -> Self;
}

impl<T> ExtractTelemetryContext for Request<T> {
    fn extract_context(self) -> Self {
        extract(self.metadata());
        self
    }
}
