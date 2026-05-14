"""
HWP regression test — baseline과 실제 비교 assert가 활성화된 상태.
비교 항목: 벡터 수, 전체 문자수 ±5%, 벡터별 텍스트 유사도 ≥ 0.85.
"""
from pathlib import Path
import pytest
import json
import difflib
from collections import Counter

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_files"
HWP_FILES = sorted([f for f in SAMPLE_DIR.glob("*.hwp") if f.is_file()])
BASELINE_DIR = Path(__file__).parent / "baselines"


def _summarize(vectors: list) -> dict:
    """벡터 리스트를 baseline 비교용 dict로 변환."""
    result = {
        "num_vectors": len(vectors),
        "vectors": [],
        "label_distribution": {},
        "total_characters": 0,
    }
    label_counts = Counter()
    for vector in vectors:
        if hasattr(vector, "model_dump"):
            vd = vector.model_dump()
        else:
            vd = vector if isinstance(vector, dict) else vars(vector)
        result["vectors"].append(vd)
        result["total_characters"] += vd.get("n_char", len(vd.get("text", "")))
        if "chunk_bboxes" in vd:
            try:
                for bbox in json.loads(vd["chunk_bboxes"]):
                    if "type" in bbox:
                        label_counts[bbox["type"]] += 1
            except (json.JSONDecodeError, TypeError):
                pass
    result["label_distribution"] = dict(label_counts)
    return result


# ---- Regression Test -------------------------------------------------------

@pytest.mark.regression
@pytest.mark.skipif(len(HWP_FILES) == 0, reason="no .hwp samples found")
@pytest.mark.parametrize("hwp_file", HWP_FILES, ids=lambda f: f.stem)
@pytest.mark.asyncio
async def test_hwp_regression(hwp_file, basic_processor):
    """HWP 문서 처리 결과를 baseline과 비교합니다."""
    baseline_path = BASELINE_DIR / f"hwp_{hwp_file.stem}.json"

    if not baseline_path.exists():
        pytest.fail(
            f"Baseline not found: {baseline_path}. "
            f"Run: pytest -m update_baseline -k test_update_hwp_baselines"
        )

    dp = basic_processor()
    vectors = await dp(None, str(hwp_file), chunker_type="hybrid")
    current = _summarize(vectors)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # 1) 벡터 수 일치
    assert current["num_vectors"] == baseline["num_vectors"], (
        f"[{hwp_file.name}] vector count: "
        f"{current['num_vectors']} != {baseline['num_vectors']}"
    )

    # 2) 전체 문자수 ±5% 이내
    base_chars = max(baseline["total_characters"], 1)
    char_ratio = abs(current["total_characters"] - base_chars) / base_chars
    assert char_ratio < 0.05, (
        f"[{hwp_file.name}] char count drift {char_ratio:.1%} "
        f"({current['total_characters']} vs {base_chars})"
    )

    # 3) 각 벡터 텍스트 유사도 ≥ 0.85
    for i, (cur_v, base_v) in enumerate(
        zip(current["vectors"], baseline["vectors"])
    ):
        sim = difflib.SequenceMatcher(
            None, cur_v.get("text", ""), base_v.get("text", "")
        ).ratio()
        assert sim >= 0.85, (
            f"[{hwp_file.name}] vector[{i}] text similarity {sim:.2%} < 85%"
        )


# ---- Baseline 업데이트 ------------------------------------------------------

@pytest.mark.update_baseline
@pytest.mark.asyncio
async def test_update_hwp_baselines(basic_processor):
    """모든 HWP baseline 데이터를 (재)생성합니다."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    for hwp_file in HWP_FILES:
        dp = basic_processor()
        vectors = await dp(None, str(hwp_file), chunker_type="hybrid")
        result = _summarize(vectors)

        out = BASELINE_DIR / f"hwp_{hwp_file.stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Updated baseline: {out}")

    if not HWP_FILES:
        print("⚠ No HWP files found in sample_files directory")


# ---- Recursive Chunker (이슈 #183 / #80) -----------------------------------
# 기본 HybridChunker 결과와 별도로, chunker_type='recursive' 결과를 baseline화한다.
# 구현 전까지는 baseline 파일이 없으므로 regression test는 자동 skip된다.

@pytest.mark.regression
@pytest.mark.skipif(len(HWP_FILES) == 0, reason="no .hwp samples found")
@pytest.mark.parametrize("hwp_file", HWP_FILES, ids=lambda f: f.stem)
@pytest.mark.asyncio
async def test_hwp_regression_recursive(hwp_file, basic_processor):
    """HWP RecursiveCharacterTextSplitter 결과를 baseline과 비교합니다."""
    baseline_path = BASELINE_DIR / f"hwp_recursive_{hwp_file.stem}.json"

    if not baseline_path.exists():
        pytest.skip(
            f"recursive baseline not yet generated for {hwp_file.name}. "
            f"Run: pytest -m update_baseline -k test_update_hwp_baselines_recursive"
        )

    dp = basic_processor()
    vectors = await dp(None, str(hwp_file), chunker_type="recursive")
    current = _summarize(vectors)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    # recursive는 RecursiveCharacterTextSplitter + 토크나이저 후처리 단계에서
    # 의존성(transformers, langchain-text-splitters 등) 미세 버전 차이에 따라
    # 분할 청크 수가 가변적이다. 정확한 vector 수 일치 검증 대신 total_characters
    # ±5%, 매칭되는 vector 텍스트 유사도 ≥85%로 검증한다.
    base_chars = max(baseline["total_characters"], 1)
    char_ratio = abs(current["total_characters"] - base_chars) / base_chars
    assert char_ratio < 0.05, (
        f"[{hwp_file.name}] char count drift {char_ratio:.1%} "
        f"({current['total_characters']} vs {base_chars})"
    )

    for i, (cur_v, base_v) in enumerate(
        zip(current["vectors"], baseline["vectors"])
    ):
        sim = difflib.SequenceMatcher(
            None, cur_v.get("text", ""), base_v.get("text", "")
        ).ratio()
        assert sim >= 0.85, (
            f"[{hwp_file.name}] vector[{i}] text similarity {sim:.2%} < 85%"
        )


@pytest.mark.update_baseline
@pytest.mark.asyncio
async def test_update_hwp_baselines_recursive(basic_processor):
    """RecursiveCharacterTextSplitter 결과의 HWP baseline 데이터를 (재)생성합니다."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    for hwp_file in HWP_FILES:
        dp = basic_processor()
        vectors = await dp(None, str(hwp_file), chunker_type="recursive")
        result = _summarize(vectors)

        out = BASELINE_DIR / f"hwp_recursive_{hwp_file.stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Updated baseline: {out}")

    if not HWP_FILES:
        print("⚠ No HWP files found in sample_files directory")
