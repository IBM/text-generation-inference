// Small helpers for using the metrics crate.
// This aims to collect all usages of the metrics crate so that future api-breaking changes can be handled in one place.


// These counter helper methods will actually increment a second counter with `_total` appended to the name.
// This is for compatibility with other runtimes that use prometheus directly, which is very
// opinionated that all counters should end with the suffix _total.
// Cite: https://prometheus.github.io/client_python/instrumenting/counter/

pub fn increment_counter(name: &'static str, value: u64) {
    let counter1 = metrics::counter!(name);
    let counter2 = metrics::counter!(format!("{name}_total"));

    counter1.increment(value);
    counter2.increment(value);
}


pub fn increment_labeled_counter(name: &'static str, labels: Vec<(&'static str, &'static str)>, value: u64) {
    let counter1 = metrics::counter!(name, labels.as_slice());
    let counter2 = metrics::counter!(format!("{name}_total"));

    counter1.increment(value);
    counter2.increment(value);
}


pub fn set_gauge(name: &'static str, value: f64) {
    let gauge = metrics::gauge!(name);
    gauge.set(value);
}


pub fn observe_histogram(name: &'static str, value: f64) {
    let histogram = metrics::histogram!(name);
    histogram.record(value)
}


pub fn observe_labeled_histogram(name: &'static str, labels: &[(&'static str, &'static str)], value: f64) {
    let histogram = metrics::histogram!(name, labels);
    histogram.record(value)
}
