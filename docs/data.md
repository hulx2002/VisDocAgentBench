# Dataset Layout

The Hugging Face snapshot is downloaded into `data/`:

```text
data/
├── benchmark/
│   ├── queries.jsonl
│   ├── evaluator_annotations.jsonl
│   └── topics.json
├── corpus/
│   ├── documents.jsonl
│   ├── pages.jsonl
│   └── pages/<document_id>/<page_id>.png
└── dataset_info.json
```

`queries.jsonl` contains only `query_id`, natural-language `query`, and `topic_id`. `evaluator_annotations.jsonl` contains `answer_page_id`, evidence `level`, ordered `support_page_ids`, and `topic_id`. Standard agents receive the query text but never the evaluator-only answer or support fields.

`documents.jsonl` records the exact arXiv version, source URL, source license, topic, and whether its rendered pages are directly included. `pages.jsonl` records all 2,375 page IDs, document IDs, one-based page indices, dimensions, rendering resolution, and expected local paths.

## Source Licenses

The corpus is derived from arXiv papers, and each source document retains its own license. Page images are directly included only for sources licensed under CC0 or Creative Commons terms without a NoDerivatives clause. For other sources, the metadata and annotations remain available while `dataset_tools/download_and_render.py` reconstructs the pages from the exact source version for local evaluation.

Benchmark-authored query and evaluator annotations are licensed under CC BY 4.0. The composite terms are summarized in the dataset card and `LICENSES.md` in the data repository.
