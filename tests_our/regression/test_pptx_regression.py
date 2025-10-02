from pathlib import Path
import pytest
import json
import difflib
from collections import Counter

# sample_files에서 모든 PPTX 파일 자동 검색
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_files"
PPTX_FILES = sorted([f for f in SAMPLE_DIR.glob("*.pptx") if f.is_file()])

async def run_pptx_test(pptx_path, baseline_path, basic_processor):
    """PPTX 파일에 대한 regression test 실행"""
    dp = basic_processor()

    if not baseline_path.exists():
        pytest.fail(f"Baseline not found: {baseline_path}. Run with -m 'update_baseline' to create.")

    # 문서 처리 - __call__ 사용
    vectors = await dp(None, str(pptx_path))

    # 현재 결과 생성
    current_result = {
        "num_vectors": len(vectors),
        "vectors": [],
        "label_distribution": {},
        "total_characters": 0
    }

    label_counts = Counter()
    for vector in vectors:
        # vector를 dict로 변환
        if hasattr(vector, "model_dump"):
            vector_data = vector.model_dump()
        else:
            vector_data = vector if isinstance(vector, dict) else vars(vector)

        current_result["vectors"].append(vector_data)
        current_result["total_characters"] += vector_data.get("n_char", len(vector_data.get("text", "")))

        # Label 분포 수집 - chunk_bboxes에서 type 추출
        if "chunk_bboxes" in vector_data:
            try:
                bboxes = json.loads(vector_data["chunk_bboxes"])
                for bbox in bboxes:
                    if "type" in bbox:
                        label_counts[bbox["type"]] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    current_result["label_distribution"] = dict(label_counts)

    # Baseline 로드 및 비교
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # 체크 항목들
    assert current_result["num_vectors"] == baseline["num_vectors"], \
        f"[{pptx_path.name}] Vector count mismatch: {current_result['num_vectors']} != {baseline['num_vectors']}"

    assert current_result["label_distribution"] == baseline["label_distribution"], \
        f"[{pptx_path.name}] Label distribution mismatch:\nCurrent: {current_result['label_distribution']}\nBaseline: {baseline['label_distribution']}"

    char_diff = abs(current_result["total_characters"] - baseline["total_characters"])
    char_ratio = char_diff / max(baseline["total_characters"], 1)
    assert char_ratio < 0.05, \
        f"[{pptx_path.name}] Character count difference too large: {char_diff} chars ({char_ratio:.1%} change)"

    for i, (current_vector, baseline_vector) in enumerate(zip(current_result["vectors"], baseline["vectors"])):
        current_text = current_vector.get("text", "")
        baseline_text = baseline_vector.get("text", "")
        similarity = difflib.SequenceMatcher(
            None,
            current_text,
            baseline_text
        ).ratio()
        assert similarity > 0.85, \
            f"[{pptx_path.name}] Vector {i} text similarity too low: {similarity:.2%}"

async def create_pptx_baseline(pptx_path, baseline_path, basic_processor):
    """PPTX 파일에 대한 baseline 생성"""
    dp = basic_processor()

    try:
        # 문서 처리 - __call__ 사용
        vectors = await dp(None, str(pptx_path))

        # Baseline 생성
        result = {
            "num_vectors": len(vectors),
            "vectors": [],
            "label_distribution": {},
            "total_characters": 0
        }

        label_counts = Counter()
        for vector in vectors:
            # vector를 dict로 변환
            if hasattr(vector, "model_dump"):
                vector_data = vector.model_dump()
            else:
                vector_data = vector if isinstance(vector, dict) else vars(vector)

            result["vectors"].append(vector_data)
            result["total_characters"] += vector_data.get("n_char", len(vector_data.get("text", "")))

            # Label 분포 수집 - chunk_bboxes에서 type 추출
            if "chunk_bboxes" in vector_data:
                try:
                    bboxes = json.loads(vector_data["chunk_bboxes"])
                    for bbox in bboxes:
                        if "type" in bbox:
                            label_counts[bbox["type"]] += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        result["label_distribution"] = dict(label_counts)

        # 저장
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✓ Updated baseline: {baseline_path}")
    except Exception as e:
        print(f"⚠ Error processing {pptx_path.name}: {e}")
        # 처리 실패 시 빈 baseline 생성
        result = {
            "num_vectors": 0,
            "vectors": [],
            "label_distribution": {},
            "total_characters": 0,
            "note": f"Processing failed: {str(e)}"
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Created placeholder baseline: {baseline_path}")

# 각 PPTX 파일에 대해 자동으로 테스트 생성
@pytest.mark.regression
@pytest.mark.parametrize("pptx_file", PPTX_FILES, ids=lambda f: f.stem)
@pytest.mark.asyncio
async def test_pptx_regression(pptx_file, basic_processor):
    """PPTX 문서 처리 결과를 baseline과 비교합니다."""
    baseline_path = Path(__file__).parent / "baselines" / f"pptx_{pptx_file.stem}.json"
    await run_pptx_test(pptx_file, baseline_path, basic_processor)

@pytest.mark.update_baseline
@pytest.mark.asyncio
async def test_update_pptx_baselines(basic_processor):
    """모든 PPTX baseline 데이터를 업데이트합니다."""
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    for pptx_file in PPTX_FILES:
        baseline_path = baseline_dir / f"pptx_{pptx_file.stem}.json"
        await create_pptx_baseline(pptx_file, baseline_path, basic_processor)

    if not PPTX_FILES:
        print("⚠ No PPTX files found in sample_files directory")