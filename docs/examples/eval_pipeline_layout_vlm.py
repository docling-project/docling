import argparse
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import (
    BenchMarkNames,
    EvaluationModality,
    PredictionFormats,
)
from docling_eval.prediction_providers.file_provider import FilePredictionProvider

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import DocumentStream, InputFormat, VlmStopReason
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.datamodel.vlm_model_specs import GRANITEDOCLING_MLX, GRANITEDOCLING_VLLM
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.experimental.datamodel.threaded_layout_vlm_pipeline_options import (
    ThreadedLayoutVlmPipelineOptions,
)
from docling.experimental.pipeline.threaded_layout_vlm_pipeline import (
    ThreadedLayoutVlmPipeline,
)
from docling.pipeline.vlm_pipeline import VlmPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate document conversion pipelines on DocLayNetV2 dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset directory (should contain a 'test' subdirectory with parquet files)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: 'test')",
    )
    (
        parser.add_argument(
            "--engine",
            type=str,
            required=False,
            default="vllm",
            choices=["vllm", "mlx"],
            help="VLM backend: 'vllm' (NVIDIA+vLLM), 'mlx' (Apple Silicon)",
        ),
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input directory
    gt_dataset_dir = Path(args.input).expanduser().resolve()
    if not gt_dataset_dir.exists():
        raise ValueError(f"Input directory does not exist: {gt_dataset_dir}")
    if not gt_dataset_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {gt_dataset_dir}")

    split = args.split
    split_dir = gt_dataset_dir / split
    if not split_dir.exists():
        raise ValueError(f"Split directory does not exist: {split_dir}")

    vlm_options = (
        vlm_model_specs.GRANITEDOCLING_VLLM.model_copy()
        if args.engine == "vllm"
        else vlm_model_specs.GRANITEDOCLING_MLX.model_copy()
    )
    vlm_options.max_new_tokens = 5000

    # Load parquet files from gt_dataset
    parquet_files = str(split_dir / "*.parquet")

    dataset = load_dataset(
        "parquet", data_files={split: parquet_files}, features=DatasetRecord.features()
    )
    split_dataset = dataset[split]

    # Create base output directory relative to input directory
    base_output_dir = gt_dataset_dir.parent
    json_output_dir_layout_aware = (
        base_output_dir / "predictions_json" / "layout_vlm_pipeline"
    )
    json_output_dir_layout_aware.mkdir(parents=True, exist_ok=True)

    json_output_dir_classic_vlm = (
        base_output_dir / "predictions_json" / "classic_vlm_pipeline"
    )
    json_output_dir_classic_vlm.mkdir(parents=True, exist_ok=True)

    report_path = base_output_dir / "vlm_length_stop_report.txt"
    with report_path.open("w") as report_file:
        report_file.write("VLM length stop report\n")

    # Create directories to save input prompt and output prompt
    input_prompt_dir = base_output_dir / "input_prompts"
    input_prompt_dir.mkdir(parents=True, exist_ok=True)

    # Create directories to save output prompt
    output_response_dir = base_output_dir / "output_responses"
    output_response_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options_layout_aware = ThreadedLayoutVlmPipelineOptions(
        # VLM configuration - defaults to GRANITEDOCLING_TRANSFORMERS
        vlm_options=vlm_options,
        # Layout configuration - defaults to DOCLING_LAYOUT_HERON
        # Batch sizes for parallel processing
        layout_batch_size=2,
        vlm_batch_size=1,
        # Queue configuration
        queue_max_size=10,
        batch_timeout_seconds=1.0,
        # Layout coordinate injection
        include_layout_coordinates=True,
        coordinate_precision=1,
        # Image processing
        images_scale=2.0,
        generate_page_images=True,
        input_prompt_dir=input_prompt_dir,
        save_input_prompt=True,
        output_response_dir=output_response_dir,
    )

    pipeline_options_classic_vlm = VlmPipelineOptions(vlm_otpions=vlm_options)

    # Create converter with the new pipeline
    print("Initializing DocumentConverter (this may take a while - loading models)...")
    doc_converter_layout_enhanced = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options_layout_aware,
                pipeline_cls=ThreadedLayoutVlmPipeline,
            ),
        }
    )
    doc_converter_classic_vlm = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options_classic_vlm,
                pipeline_cls=VlmPipeline,
            ),
        }
    )

    # Iterate through records, convert, and save as JSON
    records_classic_vlm_loop = 0
    records_layout_aware_loop = 0
    for record_dict in split_dataset:
        record = DatasetRecord.model_validate(record_dict)

        if record.original is None:
            print(f"Warning: No original document found for {record.doc_id}")
            continue

        # Get the original stream data before any conversion
        # This ensures we can create fresh streams for each conversion
        original_stream = record.original.stream
        original_stream.seek(0)  # Reset to beginning
        stream_data = original_stream.read()  # Read all data
        original_stream.seek(0)  # Reset again for potential future use

        # Create a fresh DocumentStream for the first conversion
        stream_copy_1 = BytesIO(stream_data)
        doc_stream_1 = DocumentStream(name=f"{record.doc_id}", stream=stream_copy_1)

        special_record = (
            "9f8ef69b897dde802f0f81b399b5a72baddbf2f3ff5f480cc5bdef5eea148237"
        )

        if record.doc_id != special_record:
            continue

        # Convert document with layout-aware pipeline
        conv_res_layout_aware = doc_converter_layout_enhanced.convert(
            source=doc_stream_1, raises_on_error=False
        )

        # Create a fresh DocumentStream for the second conversion
        stream_copy_2 = BytesIO(stream_data)
        doc_stream_2 = DocumentStream(name=f"{record.doc_id}", stream=stream_copy_2)

        # Convert document with classic VLM pipeline
        conv_res_classic_vlm = doc_converter_classic_vlm.convert(
            source=doc_stream_2, raises_on_error=False
        )

        if conv_res_layout_aware.document is not None:
            # Save converted document as JSON
            json_path = json_output_dir_layout_aware / f"{record.doc_id}.json"
            conv_res_layout_aware.document.save_as_json(json_path)
        else:
            print(f"Warning: Conversion failed for {record.doc_id}")

        if conv_res_classic_vlm.document is not None:
            # Save converted document as JSON
            json_path = json_output_dir_classic_vlm / f"{record.doc_id}.json"
            conv_res_classic_vlm.document.save_as_json(json_path)
        else:
            print(f"Warning: Conversion failed for {record.doc_id}")

        # Counut records where VLM stopped due to length
        layout_length_pages = []
        for page in conv_res_layout_aware.pages:
            response = page.predictions.vlm_response
            if response and response.stop_reason == VlmStopReason.LENGTH:
                layout_length_pages.append(page.page_no)

        if layout_length_pages:
            records_layout_aware_loop += len(layout_length_pages)
            with report_path.open("a") as report_file:
                for page_no in layout_length_pages:
                    report_file.write(
                        f"layout-aware: doc_id={record.doc_id}, page={page_no + 1}\n"
                    )

        classic_length_pages = []
        for page in conv_res_classic_vlm.pages:
            response = page.predictions.vlm_response
            print("/CLASSIC_VLM: Page", page.page_no + 1, "response:", response)
            if response and response.stop_reason == VlmStopReason.LENGTH:
                classic_length_pages.append(page.page_no)

        if classic_length_pages:
            records_classic_vlm_loop += len(classic_length_pages)
            with report_path.open("a") as report_file:
                for page_no in classic_length_pages:
                    report_file.write(
                        f"classic-vlm: doc_id={record.doc_id}, page={page_no + 1}\n"
                    )

    total_records = len(split_dataset) or 1
    percentage_loops_layout_aware = (records_layout_aware_loop / total_records) * 100
    percentage_loops_classic_vlm = (records_classic_vlm_loop / total_records) * 100

    with report_path.open("a") as report_file:
        report_file.write("\nSummary\n")
        report_file.write(
            f"Layout Aware Pipeline: VLM stopped due to length in {records_layout_aware_loop} out of {total_records} documents ({percentage_loops_layout_aware:.2f}%)\n"
        )
        report_file.write(
            f"Classic VLM Pipeline: VLM stopped due to length in {records_classic_vlm_loop} out of {total_records} documents ({percentage_loops_classic_vlm:.2f}%)\n"
        )

    # Create layout aware eval parquet dataset using FilePredictionProvider
    target_dataset_dir_layout_aware = base_output_dir / "layout_aware" / "eval_dataset"
    file_provider_layout_aware = FilePredictionProvider(
        prediction_format=PredictionFormats.JSON,
        source_path=json_output_dir_layout_aware,
        do_visualization=True,
        ignore_missing_files=True,
        use_ground_truth_page_images=True,
    )

    file_provider_layout_aware.create_prediction_dataset(
        name="DocLayNetV2_Predictions",
        gt_dataset_dir=gt_dataset_dir,
        target_dataset_dir=target_dataset_dir_layout_aware,
        split=split,
    )

    # Create classic vlm parquet dataset using FilePredictionProvider
    target_dataset_dir_classic_vlm = base_output_dir / "classic_vlm" / "eval_dataset"
    file_provider_classic_vlm = FilePredictionProvider(
        prediction_format=PredictionFormats.JSON,
        source_path=json_output_dir_classic_vlm,
        do_visualization=True,
        ignore_missing_files=True,
        use_ground_truth_page_images=True,
    )

    file_provider_classic_vlm.create_prediction_dataset(
        name="DocLayNetV2_Predictions",
        gt_dataset_dir=gt_dataset_dir,
        target_dataset_dir=target_dataset_dir_classic_vlm,
        split=split,
    )

    # Repeat from here to only rerun evals...

    print(f"\nClassic vlm eval datasets created at: {target_dataset_dir_classic_vlm}")
    print(f"\nLayout aware eval datasets created at: {target_dataset_dir_layout_aware}")

    # Evaluate layout - output consistent with input directory structure
    evaluation_output_dir_layout_aware = (
        base_output_dir
        / "layout_aware"
        / "evaluations"
        / EvaluationModality.MARKDOWN_TEXT.value
    )
    evaluation_output_dir_layout_aware.mkdir(parents=True, exist_ok=True)

    print("\nEvaluating layout aware...")
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,  # pick different modalities to evaluate
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_dataset_dir_layout_aware,
        odir=evaluation_output_dir_layout_aware,
        split=split,
    )

    print(
        f"Evaluation complete. Results saved to: {evaluation_output_dir_layout_aware}"
    )

    # Visualize results
    print("\nVisualizing results...")
    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_dataset_dir_layout_aware,
        odir=evaluation_output_dir_layout_aware,
        split=split,
    )

    # Evaluate classic vlm - output consistent with input directory structure
    evaluation_output_dir_classic_vlm = (
        base_output_dir
        / "classic_vlm"
        / "evaluations"
        / EvaluationModality.MARKDOWN_TEXT.value
    )
    evaluation_output_dir_classic_vlm.mkdir(parents=True, exist_ok=True)

    print("\nEvaluating classic vlm...")
    evaluate(
        modality=EvaluationModality.MARKDOWN_TEXT,  # pick different modalities to evaluate
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_dataset_dir_classic_vlm,
        odir=evaluation_output_dir_classic_vlm,
        split=split,
    )

    print(f"Evaluation complete. Results saved to: {evaluation_output_dir_classic_vlm}")

    # Visualize results
    print("\nVisualizing results...")
    visualize(
        modality=EvaluationModality.MARKDOWN_TEXT,
        benchmark=BenchMarkNames.DOCLAYNETV2,
        idir=target_dataset_dir_classic_vlm,
        odir=evaluation_output_dir_classic_vlm,
        split=split,
    )


# evaluation_output_dir = base_output_dir / "evaluations" / EvaluationModality.MARKDOWN_TEXT.value
# evaluation_output_dir.mkdir(parents=True, exist_ok=True)

# print(f"\nEvaluating markdown output...")
# evaluate(
#     modality=EvaluationModality.MARKDOWN_TEXT, # pick different modalities to evaluate
#     benchmark=BenchMarkNames.DOCLAYNETV2,
#     idir=target_dataset_dir,
#     odir=evaluation_output_dir,
#     split=split,
# )

# print(f"Evaluation complete. Results saved to: {evaluation_output_dir}")

# # Visualize results
# print(f"\nVisualizing results...")
# visualize(
#     modality=EvaluationModality.MARKDOWN_TEXT,
#     benchmark=BenchMarkNames.DOCLAYNETV2,
#     idir=target_dataset_dir,
#     odir=evaluation_output_dir,
#     split=split,
# )


# print(f"Visualization complete. Results saved to: {evaluation_output_dir}")

if __name__ == "__main__":
    main()
