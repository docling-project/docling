# PR #4218 model review: closed

Updated 2026-10-01. Review fixes landed in `0a1e58d7`: distinct schema/tagged-template representations, corrected NuExtract instructions, inference telemetry under `inference_metadata`, explicit validation states and structured `ErrorItem` failures.

`NuExtractTransformersModel` was removed. The user confirmed it was not a public-facing import and can be dropped. No shim or compatibility follow-up is needed.

NuExtract guidance uses the model prompt and caller instructions; required/nullability constraints are enforced by the output validator. Template dialect and output schema remain distinct contracts.

Current remaining work is in [the finalization assessment](extraction-api-finalization.md). The original review is preserved in Git history.
