# threaded_multistage_vlm_pipeline.py
"""Thread-safe, multi-stage VLM PDF pipeline with parallel task execution
=========================================================================
A specialized PDF conversion pipeline that uses layout analysis to route different
document elements to VLM models for specialized processing in parallel.

This implementation is based on the robust threading and queueing patterns
established in `threaded_standard_pdf_pipeline.py`.

Architecture:
* **Layout-first approach** - LayoutModel runs first to detect document structure.
* **Parallel VLM processing** - Document regions are routed to task-specific VLM models
  running in parallel.
* **Fan-out/fan-in pattern** - A layout stage fans out work to VLM stages, and a
  specialized collector stage safely fans in the results.
* **Robust Lifecycle Mgmt** - Pipeline shutdown is managed by closing queues,
  propagating the signal downstream to ensure deterministic termination without deadlocks.
* **Stateful Aggregation** - The collector stage manages the state of each page to
  ensure all VLM-processed components are re-assembled before final output.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from docling_core.types.doc import ImageRef
from docling_core.types.doc.labels import DocItemLabel
from pydantic import Field

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import (
    AssembledUnit,
    ConversionStatus,
    Page,
    VlmPrediction,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    PaginatedPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.settings import settings
from docling.datamodel.vlm_model_specs import (
    DOLPHIN_TRANSFORMERS,
    SMOLDOCLING_MLX,
    SMOLDOCLING_TRANSFORMERS,
)
from docling.models.layout_model import LayoutModel
from docling.models.page_assemble_model import PageAssembleModel, PageAssembleOptions
from docling.models.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
from docling.models.vlm_models_inline.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.models.vlm_models_inline.mlx_model import HuggingFaceMlxModel
from docling.models.vlm_task_interpreters import (
    HtmlTableInterpreter,
    OtslTableInterpreter,
    PlainTextInterpreter,
    VlmTaskInterpreter,
)
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.threaded_standard_pdf_pipeline import (
    ProcessingResult,
    RunContext,
    ThreadedItem,
    ThreadedPipelineStage,
    ThreadedQueue,
)
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures for multi-stage VLM pipeline
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VlmThreadedItem(ThreadedItem):
    """Envelope that travels between pipeline stages, can hold Page or ClusterItem.

    Note: payload field is inherited from ThreadedItem but can hold Any type (Page or ClusterItem)
    """


@dataclass
class ClusterItem:
    """A cluster extracted from a page for VLM processing."""

    page_no: int
    cluster_idx: int
    cluster: Any  # The actual cluster object from layout predictions
    task_name: str
    page_ref: Page  # Reference to source page (no copy)


@dataclass
class PageContainer:
    """Wraps a Page to add pipeline-specific metadata without polluting the core model."""

    page: Page
    expected_vlm_results: int


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Options
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VlmTaskConfig:
    """Consolidated configuration for a VLM task."""

    vlm_options: InlineVlmOptions
    labels: List[DocItemLabel]
    batch_size: int = 8


class ThreadedMultiStageVlmPipelineOptions(PaginatedPipelineOptions):
    """Options for the threaded multi-stage VLM pipeline."""

    # Layout stage
    layout_batch_size: int = 16
    layout_options: LayoutOptions

    # VLM tasks configuration - consolidated into single mapping
    vlm_tasks: Dict[str, VlmTaskConfig]

    # Standard threading options
    batch_timeout_seconds: float = 2.0
    queue_max_size: int = 100

    @classmethod
    def create_default(cls) -> ThreadedMultiStageVlmPipelineOptions:
        """Create default pipeline options with custom VLM configurations from example."""

        # Configure VLM options based on the custom pipeline example
        # base_model = SMOLVLM256_TRANSFORMERS
        # smoldocling_model = SMOLDOCLING_TRANSFORMERS

        base_model = SMOLDOCLING_TRANSFORMERS
        smoldocling_model = SMOLDOCLING_TRANSFORMERS

        text_opts = base_model.model_copy()
        text_opts.prompt = "Convert this page to docling."
        text_opts.response_format = ResponseFormat.DOCTAGS
        text_opts.max_new_tokens = 1024

        formula_opts = base_model.model_copy()
        formula_opts.prompt = "Convert formula to latex."
        formula_opts.response_format = ResponseFormat.DOCTAGS
        formula_opts.max_new_tokens = 512

        code_opts = smoldocling_model.model_copy()
        code_opts.prompt = "Convert code to text."
        code_opts.response_format = ResponseFormat.DOCTAGS

        table_opts = smoldocling_model.model_copy()
        table_opts.prompt = "Convert this table to OTSL."
        table_opts.response_format = ResponseFormat.OTSL

        return cls(
            layout_options=LayoutOptions(skip_cell_assignment=True),
            vlm_tasks={
                "table": VlmTaskConfig(
                    vlm_options=table_opts,
                    labels=[DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX],
                    batch_size=16,
                ),
                "formula": VlmTaskConfig(
                    vlm_options=formula_opts,
                    labels=[DocItemLabel.FORMULA],
                    batch_size=16,
                ),
                "code": VlmTaskConfig(
                    vlm_options=code_opts,
                    labels=[DocItemLabel.CODE],
                    batch_size=16,
                ),
                "text": VlmTaskConfig(
                    vlm_options=text_opts,
                    labels=[
                        DocItemLabel.TEXT,
                        DocItemLabel.TITLE,
                        DocItemLabel.SECTION_HEADER,
                        DocItemLabel.LIST_ITEM,
                        DocItemLabel.CAPTION,
                        DocItemLabel.FOOTNOTE,
                        DocItemLabel.PAGE_HEADER,
                        DocItemLabel.PAGE_FOOTER,
                        DocItemLabel.CHECKBOX_SELECTED,
                        DocItemLabel.CHECKBOX_UNSELECTED,
                        DocItemLabel.HANDWRITTEN_TEXT,
                        DocItemLabel.EMPTY_VALUE,
                    ],
                    batch_size=16,
                ),
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Specialized Pipeline Stages
# ──────────────────────────────────────────────────────────────────────────────


class LayoutFanOutStage(ThreadedPipelineStage):
    """Layout stage that analyzes pages and fans out clusters to VLM stages."""

    def __init__(self, task_routing: Dict[DocItemLabel, str], **kwargs):
        super().__init__(**kwargs)
        self.task_routing = task_routing
        # Selective routing: explicit outputs instead of broadcast
        self._collector_output: Optional[ThreadedQueue] = None
        self._task_outputs: Dict[str, ThreadedQueue] = {}

    # Explicit wiring API for clarity (avoid broadcast semantics)
    def set_collector_output(self, q: ThreadedQueue) -> None:
        self._collector_output = q

    def set_task_output(self, task_name: str, q: ThreadedQueue) -> None:
        self._task_outputs[task_name] = q

    def _process_batch(self, batch: Sequence[ThreadedItem]) -> list[ThreadedItem]:
        """Process pages through layout model and emit page and cluster items."""
        groups: dict[int, list[ThreadedItem]] = defaultdict(list)
        for itm in batch:
            groups[itm.run_id].append(itm)

        result: list[ThreadedItem] = []
        for rid, items in groups.items():
            good = [i for i in items if not i.is_failed]
            if not good:
                result.extend(items)
                continue

            try:
                pages = [i.payload for i in good if isinstance(i.payload, Page)]
                if len(pages) != len(good):
                    raise RuntimeError("Invalid payload types in layout stage")

                layout_pages = list(self.model(good[0].conv_res, pages))

                for page, orig_item in zip(layout_pages, good):
                    clusters_for_vlm = []
                    if page.predictions.layout and page.predictions.layout.clusters:
                        for cluster in page.predictions.layout.clusters:
                            if cluster.label in self.task_routing:
                                clusters_for_vlm.append(cluster)

                    # Wrap page in container with expected cluster count
                    page_container = PageContainer(
                        page=page, expected_vlm_results=len(clusters_for_vlm)
                    )

                    # Emit the page container for the collector
                    result.append(
                        VlmThreadedItem(
                            payload=page_container,  # type: ignore[arg-type]
                            run_id=rid,
                            page_no=page.page_no,
                            conv_res=orig_item.conv_res,
                        )
                    )

                    # Emit cluster items for VLM processing
                    for cluster_idx, cluster in enumerate(clusters_for_vlm):
                        task_name = self.task_routing[cluster.label]
                        result.append(
                            VlmThreadedItem(
                                payload=ClusterItem(  # type: ignore[arg-type]
                                    page_no=page.page_no,
                                    cluster_idx=cluster_idx,  # Note: idx is within clusters_for_vlm
                                    cluster=cluster,
                                    task_name=task_name,
                                    page_ref=page,
                                ),
                                run_id=rid,
                                page_no=page.page_no,
                                conv_res=orig_item.conv_res,
                            )
                        )

            except Exception as exc:
                _log.error(
                    "Layout fan-out stage failed for run %d: %s",
                    rid,
                    exc,
                    exc_info=True,
                )
                for it in items:
                    it.is_failed = True
                    it.error = exc
                result.extend(items)

        return result

    # Override emit to perform selective routing (no broadcast)
    def _emit(self, items: Iterable[VlmThreadedItem]) -> None:  # type: ignore[override]
        for item in items:
            payload = item.payload
            if isinstance(payload, PageContainer):
                if self._collector_output is not None:
                    if not self._collector_output.put(item):
                        _log.error(
                            "Collector queue closed while emitting from %s", self.name
                        )
                else:
                    _log.error("Collector output not wired for stage %s", self.name)
            elif isinstance(payload, ClusterItem):
                q = self._task_outputs.get(payload.task_name)
                if q is not None:
                    if not q.put(item):
                        _log.error(
                            "VLM queue for task %s closed while emitting from %s",
                            payload.task_name,
                            self.name,
                        )
                else:
                    _log.warning(
                        "No VLM output queue wired for task %s in stage %s",
                        payload.task_name,
                        self.name,
                    )
            else:
                # Unknown payload; drop with warning to avoid poisoning downstream
                _log.warning(
                    "Stage %s dropping unknown payload type: %s",
                    self.name,
                    type(payload).__name__,
                )


class VlmProcessingStage(ThreadedPipelineStage):
    """VLM stage that processes clusters for a specific task."""

    def __init__(
        self,
        task_name: str,
        task_prompt: Optional[str],
        task_options: InlineVlmOptions,
        labels_for_task: list[DocItemLabel],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.task_name = task_name
        # Store task-specific prompt to allow model instance sharing across tasks
        self.task_prompt = task_prompt
        self.interpreter: VlmTaskInterpreter = self._create_interpreter(
            task_options, labels_for_task
        )

    def _create_interpreter(
        self, task_options: InlineVlmOptions, labels_for_task: list[DocItemLabel]
    ) -> VlmTaskInterpreter:
        if DocItemLabel.TABLE in labels_for_task:
            if task_options.response_format == ResponseFormat.HTML:
                return HtmlTableInterpreter()
            elif task_options.response_format == ResponseFormat.OTSL:
                return OtslTableInterpreter()
        return PlainTextInterpreter()

    def _process_batch(self, batch: Sequence[ThreadedItem]) -> List[ThreadedItem]:
        """Process cluster items through VLM model.

        Contract: For every input ClusterItem belonging to this task, emit exactly
        one output item (success, empty, or failed). Non-cluster payloads are ignored.
        """

        groups: dict[int, list[ThreadedItem]] = defaultdict(list)
        for itm in batch:
            if (
                isinstance(itm.payload, ClusterItem)
                and itm.payload.task_name == self.task_name
            ):
                groups[itm.run_id].append(itm)

        result: List[ThreadedItem] = []
        for rid, items in groups.items():
            # Only consider non-failed inputs; failed inputs still produce a failed echo
            cluster_items = list(items)
            if not cluster_items:
                continue

            try:
                processed_items = self._process_clusters_through_vlm(cluster_items)
                result.extend(processed_items)
            except Exception as exc:
                _log.error(
                    "VLM stage %s failed for run %d: %s",
                    self.task_name,
                    rid,
                    exc,
                    exc_info=True,
                )
                for it in cluster_items:
                    it.is_failed = True
                    it.error = exc
                result.extend(cluster_items)

        return result

    def _process_clusters_through_vlm(
        self, cluster_items: List[ThreadedItem]
    ) -> List[ThreadedItem]:
        """Extract images, run VLM, and attach results to cluster objects.

        Guarantees: returns one output per input in the same order.
        """
        if not cluster_items:
            return []

        # 1. Prepare batch of images for VLM model
        image_batch: List[Any] = []
        item_mapping: List[ThreadedItem] = []
        outputs: List[ThreadedItem] = []

        for item in cluster_items:
            cluster_payload = item.payload
            assert isinstance(cluster_payload, ClusterItem)

            bbox = cluster_payload.cluster.bbox
            # Very small boxes produce low-quality crops; still emit an empty result deterministically
            if bbox.width < 5 or bbox.height < 5:
                _log.debug(
                    f"VLM stage {self.task_name}: Tiny bbox {bbox.width}x{bbox.height} -> emitting empty result"
                )
                outputs.append(item)
                continue

            cluster_image = cluster_payload.page_ref.get_image(
                scale=2.0, max_size=2048, cropbox=bbox
            )
            if cluster_image:
                image_batch.append(cluster_image)
                item_mapping.append(item)

            else:
                _log.warning(
                    f"VLM stage {self.task_name}: Failed to extract image for cluster on page {cluster_payload.page_no} -> emitting empty result"
                )
                outputs.append(item)

        if not image_batch:
            _log.debug(
                f"VLM stage {self.task_name}: No viable images; emitted {len(outputs)} empty results"
            )
            return outputs

        # 2. Run VLM model on the batch of images with task-specific prompt
        vlm_predictions = self.model.process_images(
            image_batch, prompt=self.task_prompt
        )

        # 3. Interpret predictions back into page/cluster structures
        predictions_list: List[VlmPrediction] = list(vlm_predictions)

        for i, (item, prediction) in enumerate(zip(item_mapping, predictions_list)):
            assert item.payload is not None
            assert isinstance(item.payload, ClusterItem)
            cluster = item.payload.cluster
            assert isinstance(prediction, VlmPrediction)
            try:
                self.interpreter.interpret(
                    page=item.payload.page_ref, cluster=cluster, prediction=prediction
                )
            except Exception as exc:
                page_no = item.payload.page_no if item.payload else "unknown"
                _log.debug(
                    f"VLM stage {self.task_name}: Interpreter failed for cluster {i} on page {page_no}: {exc}"
                )
            outputs.append(item)

        # If VLM returned fewer predictions than inputs (shouldn't), still honor one output per input
        if len(outputs) < len(cluster_items):
            _log.warning(
                f"VLM stage {self.task_name}: Produced {len(outputs)} results for {len(cluster_items)} inputs"
            )
            # Ensure ordering is preserved; append any missing echoes
            seen = {id(o) for o in outputs}
            for it in cluster_items:
                if id(it) not in seen:
                    outputs.append(it)

        return outputs


class FanInCollectorStage(ThreadedPipelineStage):
    """A stage that consumes from multiple input queues and terminates when all are closed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_queues: List[ThreadedQueue] = []

    def add_input_queue(self, q: ThreadedQueue) -> None:
        self.input_queues.append(q)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"Stage-{self.name}", daemon=False
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for q in self.input_queues:
            q.close()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                _log.warning("Stage %s did not terminate cleanly within 30s", self.name)

    def _run(self) -> None:
        """Polls all input queues and terminates when all are closed and empty."""
        try:
            while self._running:
                batch = []
                # Poll all input queues without blocking
                for q in self.input_queues:
                    items = q.get_batch(self.batch_size, timeout=0.0)
                    if items:
                        batch.extend(items)

                if batch:
                    processed = self._process_batch(batch)
                    self._emit(processed)

                # Check for termination condition
                all_inputs_closed = all(q.closed for q in self.input_queues)
                if all_inputs_closed:
                    # One final drain to ensure no items are left
                    final_drain = []
                    for q in self.input_queues:
                        final_drain.extend(q.get_batch(q._max, timeout=0.0))

                    if final_drain:
                        processed = self._process_batch(final_drain)
                        self._emit(processed)
                    else:
                        # All queues are closed and verifiably empty
                        break

                # If no items were found, sleep briefly to prevent busy-waiting
                if not batch:
                    time.sleep(0.01)

        except Exception:  # pragma: no cover
            _log.exception("Fatal error in stage %s", self.name)
        finally:
            for q in self._outputs:
                q.close()


class VlmCollectorStage(FanInCollectorStage):
    """Stateful collector that reassembles pages after VLM processing."""

    @dataclass
    class PageState:
        page: Page
        expected_clusters: int
        received_clusters: List[ClusterItem] = field(default_factory=list)

    def __init__(self, **kwargs):
        # The collector doesn't have its own model, it just aggregates
        kwargs["model"] = None
        super().__init__(**kwargs)
        self._runs: Dict[int, Dict[int, VlmCollectorStage.PageState]] = defaultdict(
            dict
        )

    def _process_batch(self, batch: Sequence[ThreadedItem]) -> List[ThreadedItem]:
        """Collects pages and processed clusters, emitting completed pages."""
        groups: dict[int, list[ThreadedItem]] = defaultdict(list)
        for itm in batch:
            groups[itm.run_id].append(itm)

        completed_pages: List[ThreadedItem] = []
        for rid, items in groups.items():
            run_state = self._runs[rid]
            for item in items:
                if item.is_failed:
                    # Forward failed items immediately
                    completed_pages.append(item)
                    continue

                payload = item.payload
                if isinstance(payload, PageContainer):
                    # Received a page container from layout stage, initialize its state
                    run_state[payload.page.page_no] = self.PageState(
                        page=payload.page,
                        expected_clusters=payload.expected_vlm_results,
                    )
                elif isinstance(payload, ClusterItem):
                    # Received a processed cluster from a VLM stage
                    if payload.page_no in run_state:
                        run_state[payload.page_no].received_clusters.append(payload)

                # Check if the page is now complete
                if isinstance(payload, (PageContainer, ClusterItem)):
                    page_no = (
                        payload.page.page_no
                        if isinstance(payload, PageContainer)
                        else payload.page_no
                    )
                    page_state = run_state.get(page_no)
                    if (
                        page_state
                        and len(page_state.received_clusters)
                        == page_state.expected_clusters
                    ):
                        # Page is complete, emit it
                        completed_pages.append(
                            VlmThreadedItem(
                                payload=page_state.page,
                                run_id=rid,
                                page_no=page_state.page.page_no,
                                conv_res=item.conv_res,
                            )
                        )
                        # Clean up state for this page
                        del run_state[page_no]

            # Clean up run if it's finished (all pages collected)
            if not run_state and all(q.closed for q in self.input_queues):
                del self._runs[rid]

        return completed_pages


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class ThreadedMultiStageVlmPipeline(BasePipeline):
    """High-performance PDF pipeline with parallel multi-stage VLM processing."""

    def __init__(self, pipeline_options: ThreadedMultiStageVlmPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: ThreadedMultiStageVlmPipelineOptions = pipeline_options
        self._run_seq = itertools.count(1)

        # Initialize models
        self._init_models()

    def _init_models(self) -> None:
        """Initialize all models used in the pipeline."""
        art_path = self._resolve_artifacts_path()
        self.keep_images = self.pipeline_options.generate_page_images

        # Standard models
        self.preprocessing_model = PagePreprocessingModel(
            options=PagePreprocessingOptions(
                images_scale=self.pipeline_options.images_scale,
                skip_cell_extraction=True,  # VLM provides the text content
            )
        )
        self.layout_model = LayoutModel(
            artifacts_path=art_path,
            accelerator_options=self.pipeline_options.accelerator_options,
            options=self.pipeline_options.layout_options,
        )
        self.assemble_model = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        # Initialize VLM models with deduplication across tasks via a single registry
        self.vlm_models = {}
        model_registry: dict[str, Any] = {}
        for task_name, task_config in self.pipeline_options.vlm_tasks.items():
            vlm_options = task_config.vlm_options
            key = self._vlm_model_key(vlm_options)
            model = model_registry.get(key)
            if model is None:
                if isinstance(vlm_options, InlineVlmOptions):
                    if vlm_options.inference_framework == InferenceFramework.MLX:
                        model = HuggingFaceMlxModel(
                            enabled=True,
                            artifacts_path=art_path,
                            accelerator_options=self.pipeline_options.accelerator_options,
                            vlm_options=vlm_options,
                        )
                    elif (
                        vlm_options.inference_framework
                        == InferenceFramework.TRANSFORMERS
                    ):
                        model = HuggingFaceTransformersVlmModel(
                            enabled=True,
                            artifacts_path=art_path,
                            accelerator_options=self.pipeline_options.accelerator_options,
                            vlm_options=vlm_options,
                        )
                    elif vlm_options.inference_framework == InferenceFramework.VLLM:
                        from docling.models.vlm_models_inline.vllm_model import (
                            VllmVlmModel,
                        )

                        model = VllmVlmModel(
                            enabled=True,
                            artifacts_path=art_path,
                            accelerator_options=self.pipeline_options.accelerator_options,
                            vlm_options=vlm_options,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported inference framework: {vlm_options.inference_framework}"
                        )
                else:
                    raise ValueError(
                        f"Unsupported VLM options type: {type(vlm_options)}"
                    )
                model_registry[key] = model

            self.vlm_models[task_name] = model

        # Build task routing map (label -> task_name)
        self.task_routing = {}
        for task_name, task_config in self.pipeline_options.vlm_tasks.items():
            for label in task_config.labels:
                self.task_routing[label] = task_name

    def _resolve_artifacts_path(self) -> Optional[Path]:
        """Resolve the artifacts path for model loading."""
        if self.pipeline_options.artifacts_path:
            p = Path(self.pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path:
            p = Path(settings.artifacts_path).expanduser()
        else:
            return None
        if not p.is_dir():
            raise RuntimeError(
                f"{p} does not exist or is not a directory containing the required models"
            )
        return p

    def _create_run_ctx(self) -> RunContext:
        """Create the pipeline DAG for a single run."""
        opts = self.pipeline_options

        # Create stages
        preprocess = ThreadedPipelineStage(
            name="preprocess",
            model=self.preprocessing_model,
            batch_size=1,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        layout = LayoutFanOutStage(
            name="layout",
            model=self.layout_model,
            task_routing=self.task_routing,
            batch_size=opts.layout_batch_size,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        vlm_stages = {}
        for task, model in self.vlm_models.items():
            task_config = self.pipeline_options.vlm_tasks[task]
            task_prompt_str = self._resolve_task_prompt(task_config.vlm_options)
            vlm_stages[task] = VlmProcessingStage(
                name=f"vlm_{task}",
                model=model,
                task_name=task,
                task_prompt=task_prompt_str,
                task_options=task_config.vlm_options,
                labels_for_task=task_config.labels,
                batch_size=task_config.batch_size,
                batch_timeout=opts.batch_timeout_seconds,
                queue_max_size=opts.queue_max_size,
            )

        collector = VlmCollectorStage(
            name="collector",
            batch_size=32,
            batch_timeout=opts.batch_timeout_seconds,
            queue_max_size=opts.queue_max_size,
        )

        # Wire the DAG
        output_q = ThreadedQueue(opts.queue_max_size)
        preprocess.add_output_queue(layout.input_queue)

        # Fan-out: layout -> VLM stages (for clusters) AND collector (for pages)
        q_layout_to_collector = ThreadedQueue(opts.queue_max_size)
        layout.set_collector_output(q_layout_to_collector)
        collector.add_input_queue(q_layout_to_collector)

        for task_name, vlm_stage in vlm_stages.items():
            q_layout_to_vlm = ThreadedQueue(opts.queue_max_size)
            layout.set_task_output(task_name, q_layout_to_vlm)
            vlm_stage.input_queue = q_layout_to_vlm

            q_vlm_to_collector = ThreadedQueue(opts.queue_max_size)
            vlm_stage.add_output_queue(q_vlm_to_collector)
            collector.add_input_queue(q_vlm_to_collector)

        # The collector now writes directly to the output queue
        collector.add_output_queue(output_q)

        all_stages = [preprocess, layout, *vlm_stages.values(), collector]
        return RunContext(
            stages=all_stages, first_stage=preprocess, output_queue=output_q
        )

    def _vlm_model_key(self, vlm_options: InlineVlmOptions) -> str:
        """Compute a stable deduplication key for a VLM options object.

        Inline models are keyed by (framework, repo_id). API models are keyed by URL.
        """
        if isinstance(vlm_options, InlineVlmOptions):
            return f"inline::{vlm_options.inference_framework.value}::{vlm_options.repo_id}"
        raise ValueError(
            f"Unsupported VLM options type for keying: {type(vlm_options)}"
        )

    def _resolve_task_prompt(self, vlm_options: InlineVlmOptions) -> Optional[str]:
        """Resolve a per-task prompt string for image-only processing.

        If the prompt is callable, invoke it with None to obtain a string, since
        the image-only path does not have a SegmentedPage context.
        """
        prompt = vlm_options.prompt
        if callable(prompt):
            return prompt(None)
        return prompt

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Build document using the multi-stage VLM pipeline."""
        run_id = next(self._run_seq)
        assert isinstance(conv_res.input._backend, PdfDocumentBackend)
        backend = conv_res.input._backend

        # Load pages
        start_page, end_page = conv_res.input.limits.page_range
        pages: List[Page] = []
        for i in range(conv_res.input.page_count):
            if start_page - 1 <= i <= end_page - 1:
                page = Page(page_no=i)
                page._backend = backend.load_page(i)
                if page._backend and page._backend.is_valid():
                    page.size = page._backend.get_size()
                    conv_res.pages.append(page)
                    pages.append(page)

        if not pages:
            conv_res.status = ConversionStatus.FAILURE
            return conv_res

        total_pages = len(pages)
        ctx = self._create_run_ctx()

        for stage in ctx.stages:
            stage.start()

        proc = ProcessingResult(total_expected=total_pages)
        fed_idx = 0
        feed_batch_size = 32

        try:
            while proc.success_count + proc.failure_count < total_pages:
                # 1. Feed the pipeline
                while fed_idx < total_pages:
                    ok = ctx.first_stage.input_queue.put(
                        VlmThreadedItem(
                            payload=pages[fed_idx],
                            run_id=run_id,
                            page_no=pages[fed_idx].page_no,
                            conv_res=conv_res,
                        ),
                        timeout=0.0,  # Non-blocking
                    )
                    if not ok:
                        break  # Input queue is full, switch to draining
                    fed_idx += 1

                if fed_idx == total_pages:
                    ctx.first_stage.input_queue.close()

                # 2. Drain results
                out_batch = ctx.output_queue.get_batch(feed_batch_size, timeout=0.1)
                for itm in out_batch:
                    if itm.run_id != run_id:
                        continue
                    if itm.is_failed or itm.error:
                        proc.failed_pages.append(
                            (itm.page_no, itm.error or RuntimeError("unknown error"))
                        )
                    elif itm.payload is not None:
                        proc.pages.append(itm.payload)

                # 3. Check for early termination
                if not out_batch and ctx.output_queue.closed:
                    missing = total_pages - (proc.success_count + proc.failure_count)
                    if missing > 0:
                        _log.warning(
                            "Pipeline terminated early, missing %d pages.", missing
                        )
                        proc.failed_pages.extend(
                            [(-1, RuntimeError("pipeline terminated early"))] * missing
                        )
                    break
        finally:
            for stage in ctx.stages:
                stage.stop()
            ctx.output_queue.close()

        self._integrate_results(conv_res, proc)
        return conv_res

    def _integrate_results(
        self, conv_res: ConversionResult, proc: ProcessingResult
    ) -> None:
        """Integrate processing results back into conversion result."""
        page_map = {p.page_no: p for p in proc.pages}

        # Rebuild pages list from successfully processed pages
        final_pages = []
        failed_page_nos = {no for no, err in proc.failed_pages}
        for p in conv_res.pages:
            if p.page_no in page_map:
                final_pages.append(page_map[p.page_no])
            elif p.page_no not in failed_page_nos:
                # This case should ideally not happen with the new collector
                _log.warning(
                    "Page %d was neither successful nor failed, discarding.", p.page_no
                )

        conv_res.pages = sorted(final_pages, key=lambda p: p.page_no)

        if proc.is_complete_failure or not conv_res.pages:
            conv_res.status = ConversionStatus.FAILURE
        elif proc.is_partial_success or len(conv_res.pages) < proc.total_expected:
            conv_res.status = ConversionStatus.PARTIAL_SUCCESS
        else:
            conv_res.status = ConversionStatus.SUCCESS

        if not self.keep_images:
            for p in conv_res.pages:
                p._image_cache = {}

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assemble the final document structure with VLM-enhanced content."""
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            # 1. Run PageAssembleModel on all collected pages to create page.assembled
            processed_pages = list(self.assemble_model(conv_res, conv_res.pages))
            conv_res.pages = processed_pages

            # 2. Collect all assembled elements from all pages
            all_elements, all_headers, all_body = [], [], []
            for page in conv_res.pages:
                if page.assembled:
                    all_elements.extend(page.assembled.elements)
                    all_headers.extend(page.assembled.headers)
                    all_body.extend(page.assembled.body)

            conv_res.assembled = AssembledUnit(
                elements=all_elements, headers=all_headers, body=all_body
            )

            # 3. Generate the final DoclingDocument using the reading order model
            conv_res.document = self.reading_order_model(conv_res)

            if self.pipeline_options.generate_page_images:
                for page in conv_res.pages:
                    if page.image:
                        page_no = page.page_no + 1
                        if page_no in conv_res.document.pages:
                            conv_res.document.pages[page_no].image = ImageRef.from_pil(
                                page.image,
                                dpi=int(72 * self.pipeline_options.images_scale),
                            )
        return conv_res

    @classmethod
    def get_default_options(cls) -> ThreadedMultiStageVlmPipelineOptions:
        """Get default pipeline options."""
        return ThreadedMultiStageVlmPipelineOptions.create_default()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
        """Check if backend is supported."""
        return isinstance(backend, PdfDocumentBackend)

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determine final conversion status."""
        return conv_res.status

    def _unload(self, conv_res: ConversionResult) -> None:
        """Clean up resources."""
        for p in conv_res.pages:
            if p._backend is not None:
                p._backend.unload()
        if conv_res.input._backend:
            conv_res.input._backend.unload()
