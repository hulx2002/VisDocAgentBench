# Contributing

Bug reports and reproducibility questions are welcome through GitHub Issues. A useful report includes the command, environment, relevant configuration with credentials removed, and the shortest log excerpt that identifies the failure.

Code changes should preserve the released benchmark protocol unless they are explicitly presented as a new experimental setting. Before opening a pull request, run:

```bash
python -m pytest -q
python -m py_compile $(find baselines dataset_tools evaluation scripts -name '*.py')
```

Do not commit model weights, benchmark data, generated embeddings, OCR caches, predictions, traces, API credentials, or provider-specific endpoints.
