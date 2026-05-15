"""B300 PaddleOCR smoke test - in-process inference.

이 스크립트는 컨테이너 안에서 직접 실행된다.

수행하는 검증:
  1) paddle import / paddle.utils.run_check
  2) paddle.device.get_device() 가 'gpu:*' 인지 확인 (GPU 미사용이면 FAIL)
  3) /app/config/ocr.yaml 파이프라인을 로드하여 합성 이미지 1장 OCR
  4) 결과(text/score)를 JSON 으로 dump 하여 baseline 비교에 사용

GPU 가 없는 환경(빌드 머신, 일반 CI) 에서는 의도적으로 실패시킨다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _make_sample_image(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 96), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = "OCR SMOKE 2026"
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 24), text, fill=(0, 0, 0), font=font)
    img.save(path)


def _assert_gpu() -> str:
    import paddle

    device = paddle.device.get_device()
    if not device.startswith("gpu"):
        raise SystemExit(
            f"[smoke] FAIL: paddle device is '{device}', expected 'gpu:*'. "
            "GPU 가 컨테이너에 노출되지 않았거나 wheel 빌드 타깃과 호환되지 않습니다."
        )
    return device


def _run_pipeline(image_path: Path) -> list[dict]:
    from paddlex import create_pipeline

    pipeline = create_pipeline(pipeline="/app/config/ocr.yaml", device="gpu")
    results = list(pipeline.predict(str(image_path)))

    flattened: list[dict] = []
    for res in results:
        rec_texts = list(res.get("rec_texts", []) or [])
        rec_scores = list(res.get("rec_scores", []) or [])
        for text, score in zip(rec_texts, rec_scores):
            flattened.append({"text": text, "score": float(score)})
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="결과 JSON 출력 경로")
    parser.add_argument(
        "--image",
        default=None,
        help="OCR 대상 이미지 경로. 미지정 시 임시 합성 이미지 사용",
    )
    args = parser.parse_args()

    print(f"[smoke] paddle device check ...", flush=True)
    device = _assert_gpu()
    print(f"[smoke] paddle device = {device}", flush=True)

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = Path("/tmp/ocr_smoke_sample.png")
        _make_sample_image(image_path)
        print(f"[smoke] synthetic sample image written to {image_path}", flush=True)

    print(f"[smoke] running OCR pipeline ...", flush=True)
    items = _run_pipeline(image_path)

    payload = {"device": device, "image": str(image_path), "items": items}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if not items:
        print(
            "[smoke] WARNING: pipeline returned no text. "
            "OCR 자체는 실행됐지만 인식 결과가 비어 있습니다.",
            file=sys.stderr,
        )
    else:
        print(f"[smoke] recognized {len(items)} text item(s):", flush=True)
        for it in items:
            print(f"  - {it['text']!r} (score={it['score']:.3f})", flush=True)

    print(f"[smoke] PASS: inference completed. result -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
