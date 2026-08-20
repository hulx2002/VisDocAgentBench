# Reproduction Guide

## Frozen Agent Protocol

Both routes operate over the same 2,375 rendered pages and return ten ranked opaque page handles. The visual route searches Qwen3-VL-Embedding-8B page-image representations. The OCR-text route searches Qwen3-Embedding-8B representations of PaddleOCR-VL-1.6 text. Both agents may inspect page images and crop regions; the OCR-text route additionally retrieves cached full-page OCR, while the visual route can invoke OCR on one page or crop at a time.

The full setting uses 12 agent actions, 3,200 output tokens per planner turn, and four query workers unless a model-specific rate limit requires fewer. Search returns at most 50 handles. `inspect_pages` and `crop_pages` each accept at most ten images per call. There are no separate cumulative tool quotas or automatic whole-query restarts.

Connection failures, read timeouts, and HTTP 408/429/500/502/503/504 are retried before an observation is added, for at most three physical attempts. Backoff is exponential and honors `Retry-After`. Other provider errors terminate the episode. Malformed planner text is not resampled at the transport layer: the raw response is logged, the current action is consumed, and the next planner turn receives concise format-error feedback. The tolerant parser executes the first complete JSON object and records any ignored trailing content.

Partial-episode checkpoints preserve the exact current history and resume the same unobserved planner step after transport exhaustion. Before resuming, the runner verifies that the query and answer, prompt and tool configuration, planner settings, and retrieval input contents match the current run. Page pixels and local model artifacts are included in this check. A mismatch is rejected before existing predictions can be rewritten. For a mutable embedding endpoint that retains the same URL and model name, set `EMBEDDING_SERVICE_REVISION` to an immutable revision label.

## Service Placement

The agent process, embedding service, OCR process, and planner endpoint are independent. They may run on one host or different hosts by changing the embedding base URL and API configuration. Keep the corpus, OCR cache, embeddings, handle seed, and prompts fixed.

Qwen3.5-397B-A17B was served in BF16 on two eight-GPU nodes, with tensor parallelism 8 within each node and pipeline parallelism 2 across nodes. The service used a 262,144-token context limit, four concurrent sequences, and model-default sampling. Reproduction requires a Chat Completions-compatible endpoint for this model; set its URL and served model name in the two `configs/qwen35.*.json` files. Thinking and no-thinking runs use the provided chat-template configurations. Thinking allows 32,768 output tokens per planner turn and a 1,200-second request window; no-thinking uses 3,200 tokens and 600 seconds. The serving backend and hardware topology may differ, while the model, precision, sampling, context capacity, and harness settings should remain fixed. The auxiliary embedding and OCR services can run on reserved GPU memory or separate hosts.

## Fixed Two-Stage Reranker

The non-iterative comparison retrieves 30 candidates with the route-specific Qwen retriever. The same VLM independently assesses ten fixed batches of three candidates, where every candidate supplies the complete page image and cached OCR text. One final call aggregates the assessments into a top-10 ranking. No intermediate observation can change the query, candidate set, batching, or subsequent retrieval.

## Output Contract

Every run directory contains `predictions.jsonl`, `report.json`, a log, and the deterministic opaque-handle mapping. Evaluation uses the first ten ranks, which must be distinct pages listed in `data/corpus/pages.jsonl`. Missing or invalid episodes are retained and score zero. Generated outputs are intentionally local and are ignored by Git.
