# Extraction implementation

The engine, multi-format and channel work from PR #4201 is included in PR #4218. The current implementation supports PDF/image, DOCX/HTML/Markdown and DCLX, with page/document scopes and model-compatible text/image channels.

Use `ExtractionVlmOptions` for model spec and engine configuration; pass schema/template/instructions through `ExtractionTarget`. NuExtract3's API route is verified by its GGUF run behind llama-server; Granite's API route is verified through LM Studio. Local MLX/vLLM engines remain outside this work; OpenAI-compatible services use the API engine.

See [the execution ledger](extraction-additional-vlm-models-execution.md), [the service-client contract](extraction-service-client-api.md) and [the finalization assessment](extraction-api-finalization.md). Old design details remain in Git history.
