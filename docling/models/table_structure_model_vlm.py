import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional

from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.utils import parse_otsl_table_content
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    Page,
    Table,
    TableStructurePrediction,
    VlmPredictionToken,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    TableStructureOptions,
)
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TableStructureModelVlmMlx(BasePageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: TableStructureOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.options = options
        model_repo_id = "ds4sd/granite-docling-258m-2-9-2025-v2-mlx-bf16"

        self.max_tokens = 4096
        self.temperature = 0
        self.stop_strings = ["</doctag>", "<end_of_utterance>"]

        self.enabled = enabled
        if self.enabled:
            try:
                from mlx_vlm import generate, load, stream_generate  # type: ignore
                from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
                from mlx_vlm.utils import load_config  # type: ignore
            except ImportError:
                raise ImportError(
                    "mlx-vlm is not installed. Please install it via `pip install mlx-vlm` to use MLX VLM models."
                )

            repo_cache_folder = model_repo_id.replace("/", "--")

            self.apply_chat_template = apply_chat_template
            self.stream_generate = stream_generate

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(
                    model_repo_id,
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            ## Load the model
            self.vlm_model, self.processor = load(artifacts_path)
            self.config = load_config(artifacts_path)

            self.scale = 2.0  # Scale up table input images to 144 dpi

    def _predict_images(self, image_batch: Iterable[Image.Image]) -> Iterable[str]:
        user_prompt = "Convert table to OTSL."

        # Use the MLX chat template approach like in the __call__ method
        formatted_prompt = self.apply_chat_template(
            self.processor, self.config, user_prompt, num_images=1
        )

        for image in image_batch:
            # Stream generate with stop strings support
            start_time = time.time()
            _log.debug("start generating ...")

            tokens: list[VlmPredictionToken] = []
            output = ""

            # Use stream_generate for proper stop string handling
            for token in self.stream_generate(
                self.vlm_model,
                self.processor,
                formatted_prompt,
                [image],  # MLX stream_generate expects list of images
                max_tokens=self.max_tokens,
                verbose=False,
                temp=self.temperature,
            ):
                # Collect token information
                if len(token.logprobs.shape) == 1:
                    tokens.append(
                        VlmPredictionToken(
                            text=token.text,
                            token=token.token,
                            logprob=token.logprobs[token.token],
                        )
                    )
                elif len(token.logprobs.shape) == 2 and token.logprobs.shape[0] == 1:
                    tokens.append(
                        VlmPredictionToken(
                            text=token.text,
                            token=token.token,
                            logprob=token.logprobs[0, token.token],
                        )
                    )
                else:
                    _log.warning(
                        f"incompatible shape for logprobs: {token.logprobs.shape}"
                    )

                output += token.text

                # Check for any configured stop strings
                if self.stop_strings:
                    if any(stop_str in output for stop_str in self.stop_strings):
                        _log.debug("Stopping generation due to stop string match")
                        break

            generation_time = time.time() - start_time

            _log.debug(
                f"{generation_time:.2f} seconds for {len(tokens)} tokens ({len(tokens) / generation_time:.1f} tokens/sec)."
            )

            yield output

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        # Convert to list to allow multiple iterations
        pages = list(page_batch)

        # Separate valid and invalid pages
        table_images: List[Image.Image] = []
        table_clusters: List[Cluster] = []
        pages_to_tables: List[List[int]] = []

        tbl_ix = 0
        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                pages_to_tables.append([])
                continue

            table_indexes = []
            assert page.predictions.layout is not None
            for cluster in page.predictions.layout.clusters:
                if cluster.label not in {
                    DocItemLabel.TABLE,
                    DocItemLabel.DOCUMENT_INDEX,
                }:
                    continue

                table_image = page.get_image(scale=self.scale, cropbox=cluster.bbox)
                assert table_image is not None

                table_clusters.append(cluster)
                table_images.append(table_image)

                table_indexes.append(tbl_ix)
                tbl_ix += 1

            pages_to_tables.append(table_indexes)

        assert len(pages) == len(pages_to_tables)

        # Process all valid pages with batch prediction
        batch_predictions = []
        if table_images:
            with TimeRecorder(conv_res, "table_structure"):
                batch_predictions = list(self._predict_images(table_images))
        assert len(batch_predictions) == len(table_images)

        for page, page_tables_map in zip(pages, pages_to_tables):
            if not page_tables_map:
                yield page

            page.predictions.tablestructure = TableStructurePrediction()  # dummy

            for tbl_ix in page_tables_map:
                otsl_seq = batch_predictions[tbl_ix]
                table_cluster = table_clusters[tbl_ix]

                print(f"{otsl_seq=}")
                table_data = parse_otsl_table_content(otsl_seq)
                print(f"{table_data.num_rows=}")
                print(f"{table_data.num_cols=}")

                tbl = Table(
                    otsl_seq=[otsl_seq],
                    table_cells=table_data.table_cells,
                    num_rows=table_data.num_rows,
                    num_cols=table_data.num_cols,
                    id=table_cluster.id,
                    page_no=page.page_no,
                    cluster=table_cluster,
                    label=table_cluster.label,
                )

                page.predictions.tablestructure.table_map[table_cluster.id] = tbl

            yield page
