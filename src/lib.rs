pub mod anomaly;
pub mod correlation;
pub mod extraction;
pub mod format;
pub mod label;
pub mod scoring;
pub mod stats;
pub mod timestamp;
pub mod types;
#[cfg(feature = "wasm")]
pub mod wasm;

use std::collections::HashMap;

use crate::anomaly::detect_anomalies;
use crate::extraction::drain3::Config;
use crate::extraction::pipeline::ClpDrainPipeline;
use crate::scoring::{compute_scores, PatternScore};
use crate::stats::PatternStore;
use crate::timestamp::{extract_timestamp, strip_timestamp};
use crate::types::{FormatOptions, PatternID};

// ── CLI-only imports and types ──────────────────────────────────────────

#[cfg(feature = "cli")]
use std::fs::File;
#[cfg(feature = "cli")]
use std::io::{self, BufRead, BufReader};
#[cfg(feature = "cli")]
use anyhow::Result;
#[cfg(feature = "cli")]
use clap::Parser;
#[cfg(feature = "cli")]
use crate::format::format_output;
#[cfg(feature = "cli")]
use crate::types::OutputMode;

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(
    name = "ctrlb-decompose",
    version,
    about = "Compress raw log lines into structural patterns"
)]
pub struct Args {
    /// Log file to process (reads from stdin if omitted or "-")
    pub file: Option<String>,

    /// Output in human-readable format (default)
    #[arg(long)]
    pub human: bool,

    /// Output in LLM-optimized format (compact, token-efficient)
    #[arg(long)]
    pub llm: bool,

    /// Output in JSON format
    #[arg(long)]
    pub json: bool,

    /// Show top N patterns (default: 20)
    #[arg(long, default_value_t = 20)]
    pub top: usize,

    /// Include N example raw lines per pattern (default: 0)
    #[arg(long, default_value_t = 0)]
    pub context: usize,

    /// Disable terminal colors
    #[arg(long)]
    pub no_color: bool,

    /// Suppress the header/footer banners
    #[arg(long)]
    pub no_banner: bool,

    /// Suppress progress output on stderr
    #[arg(short, long)]
    pub quiet: bool,
}

#[cfg(feature = "cli")]
impl Args {
    pub fn output_mode(&self) -> OutputMode {
        if self.json {
            OutputMode::Json
        } else if self.llm {
            OutputMode::Llm
        } else {
            OutputMode::Human
        }
    }

    pub fn to_format_options(&self) -> FormatOptions {
        FormatOptions {
            top: self.top,
            context: if self.llm && self.context == 0 { 2 } else { self.context },
            no_color: self.no_color,
            no_banner: self.no_banner,
            output_mode: self.output_mode(),
        }
    }
}

#[cfg(feature = "cli")]
pub fn run(args: Args) -> Result<()> {
    use chrono::{DateTime, Utc};
    use rayon::prelude::*;
    use crate::extraction::clp::core::EightByteEncodedVariable;
    use crate::extraction::pipeline::{clp_encode_line, new_clp_context, merge_variables, DrainResult};

    let mut reader: Box<dyn BufRead> = match args.file.as_deref() {
        None | Some("-") => Box::new(BufReader::new(io::stdin())),
        Some(path) => Box::new(BufReader::new(File::open(path)?)),
    };

    let opts = args.to_format_options();

    let mut pipeline = ClpDrainPipeline::new(Config::default());
    let mut store = PatternStore::new(opts.context);
    let mut line_number: u64 = 0;

    const BATCH_SIZE: usize = 8192;
    let mut batch_lines: Vec<String> = Vec::with_capacity(BATCH_SIZE);

    struct PreProcessed {
        line: String,
        timestamp: Option<DateTime<Utc>>,
        logtype: String,
        encoded_vars: Vec<EightByteEncodedVariable>,
        dictionary_vars: Vec<String>,
    }

    let mut raw_buf = Vec::new();
    loop {
        // Phase 1: Read a batch of lines
        batch_lines.clear();
        loop {
            raw_buf.clear();
            let bytes_read = reader.read_until(b'\n', &mut raw_buf)?;
            if bytes_read == 0 {
                break;
            }
            let line = String::from_utf8_lossy(&raw_buf)
                .trim_end_matches('\n')
                .trim_end_matches('\r')
                .to_string();
            if !line.is_empty() {
                batch_lines.push(line);
            }
            if batch_lines.len() >= BATCH_SIZE {
                break;
            }
        }

        if batch_lines.is_empty() {
            break;
        }

        // Phase 2: Parallel timestamp extraction + CLP encoding
        let pre_processed: Vec<PreProcessed> = batch_lines
            .par_drain(..)
            .map_init(
                || new_clp_context(),
                |clp_ctx, line| {
                    let ts_match = extract_timestamp(&line);
                    let stripped;
                    let process_input = match &ts_match {
                        Some(ts) => {
                            stripped = strip_timestamp(&line, ts);
                            stripped.as_str()
                        }
                        None => line.as_str(),
                    };
                    let encoded = clp_encode_line(clp_ctx, process_input);
                    PreProcessed {
                        line,
                        timestamp: ts_match.map(|ts| ts.datetime),
                        logtype: encoded.logtype,
                        encoded_vars: encoded.encoded_vars,
                        dictionary_vars: encoded.dictionary_vars,
                    }
                },
            )
            .collect();

        // Phase 3a: Serial Drain3 only — get pattern_id + template for each line
        let drain_results: Vec<(DrainResult, &PreProcessed)> = pre_processed
            .iter()
            .map(|pre| {
                let dr = pipeline.drain_only(&pre.logtype);
                (dr, pre)
            })
            .collect();

        // Phase 3b: Parallel merge — expand CLP placeholders + classify variables
        let merged: Vec<_> = drain_results
            .par_iter()
            .map(|(dr, pre)| {
                merge_variables(
                    &dr.template,
                    &pre.logtype,
                    &pre.encoded_vars,
                    &pre.dictionary_vars,
                    dr.pattern_id,
                    dr.count,
                )
            })
            .collect();

        // Phase 3c: Serial accumulate
        for (parsed, pre) in merged.into_iter().zip(pre_processed.iter()) {
            line_number += 1;
            store.accumulate(
                parsed.pattern_id,
                &parsed.display_template,
                &parsed.variables,
                pre.timestamp,
                &pre.line,
                line_number,
            );
        }
    }

    if !args.quiet {
        eprintln!(
            "Processed {} lines -> {} patterns",
            store.global_line_count,
            store.patterns.len()
        );
    }

    store.finalize();

    let anomalies = detect_anomalies(&store);
    let scores = compute_scores(&store, &anomalies);

    let output = format_output(&store, &opts, &scores);
    print!("{}", output);

    if !args.quiet {
        eprintln!(
            "\nPowered by CtrlB \u{00b7} Search 5TB of logs in 614ms \u{2192} ctrlb.ai"
        );
    }

    Ok(())
}

// ── Core processing (no I/O, works in CLI and WASM) ─────────────────────

pub struct AnalysisOutput {
    pub store: PatternStore,
    pub scores: HashMap<PatternID, PatternScore>,
}

/// Process log text and return analysis results.
/// This is the WASM-friendly entry point — no filesystem, no stdin.
pub fn process_log_text(input: &str, opts: &FormatOptions) -> AnalysisOutput {
    let mut pipeline = ClpDrainPipeline::new(Config::default());
    let mut store = PatternStore::new(opts.context);
    let mut line_number: u64 = 0;

    for line in input.lines() {
        if line.is_empty() {
            continue;
        }
        line_number += 1;

        let ts_match = extract_timestamp(line);
        let stripped;
        let process_input = match &ts_match {
            Some(ts) => {
                stripped = strip_timestamp(line, ts);
                stripped.as_str()
            }
            None => line,
        };

        let parsed = pipeline.process_line(process_input);

        store.accumulate(
            parsed.pattern_id,
            &parsed.display_template,
            &parsed.variables,
            ts_match.map(|ts| ts.datetime),
            line,
            line_number,
        );
    }

    store.finalize();

    let anomalies = detect_anomalies(&store);
    let scores = compute_scores(&store, &anomalies);

    AnalysisOutput { store, scores }
}
