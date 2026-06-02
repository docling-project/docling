from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fine-tune an object detection model with COCO annotations.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=script_dir / "_annotations.coco.json",
        help="Path to COCO annotation json file.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=script_dir / "images",
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="PekingU/rtdetr_r50vd",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "model_output",
        help="Directory to save training outputs.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Set >0 for smoke test.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0",
        help="GPU list for CUDA_VISIBLE_DEVICES. Use 'all' to keep current environment.",
    )
    return parser.parse_args()


def load_coco_dataset(annotations_path: Path, images_dir: Path) -> Dataset:
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    with annotations_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_image_id = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image_id[ann["image_id"]].append(ann)

    examples = []
    for image_id, image_info in images_by_id.items():
        file_name = image_info["file_name"]
        image_path = images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image referenced in COCO not found: {image_path}")

        objects = annotations_by_image_id.get(image_id, [])
        bbox_list = [obj["bbox"] for obj in objects]
        category_list = [obj["category_id"] for obj in objects]
        area_list = [obj.get("area", obj["bbox"][2] * obj["bbox"][3]) for obj in objects]

        examples.append(
            {
                "image_id": int(image_id),
                "image_path": str(image_path),
                "bbox": bbox_list,
                "category": category_list,
                "area": area_list,
            }
        )

    return Dataset.from_list(examples)


def build_collate_fn(processor):
    def collate_fn(batch):
        images = []
        targets = []

        for example in batch:
            with Image.open(example["image_path"]) as img:
                images.append(img.convert("RGB"))

            annotations = [
                {
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0,
                }
                for category_id, bbox, area in zip(example["category"], example["bbox"], example["area"])
            ]

            targets.append({"image_id": int(example["image_id"]), "annotations": annotations})

        encoded = processor(
            images=images,
            annotations=targets,
            return_tensors="pt",
            format="coco_detection",
        )
        return {"pixel_values": encoded["pixel_values"], "labels": encoded["labels"]}

    return collate_fn


def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Import transformers after CUDA_VISIBLE_DEVICES is fixed.
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments

    train_dataset = load_coco_dataset(args.annotations, args.images_dir)
    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForObjectDetection.from_pretrained(args.model_name_or_path)
    collate_fn = build_collate_fn(processor)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
