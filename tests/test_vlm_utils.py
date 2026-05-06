"""Unit tests for VLM utility functions (stop-string stripping, image sizing)."""

from docling_core.types.doc import Size

from docling.utils.vlm_utils import compute_qwen2vl_image_size, strip_stop_strings


class TestStripStopStrings:
    """Tests for stop-string removal from decoded VLM outputs."""

    def test_removes_full_stop_string(self):
        texts = ["hello world<|im_end|>"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == ["hello world"]

    def test_removes_partial_trailing_prefix(self):
        texts = ["hello world<|im_"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == ["hello world"]

    def test_no_stop_string_present(self):
        texts = ["hello world"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == ["hello world"]

    def test_multiple_stop_strings(self):
        texts = ["output<|endoftext|>extra"]
        result = strip_stop_strings(texts, ["<|im_end|>", "<|endoftext|>"])
        assert result == ["output"]

    def test_first_occurrence_wins(self):
        texts = ["a<|im_end|>b<|im_end|>c"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == ["a"]

    def test_multiple_texts(self):
        texts = ["text1<|im_end|>", "text2", "text3<|im_"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == ["text1", "text2", "text3"]

    def test_empty_texts(self):
        result = strip_stop_strings([], ["<|im_end|>"])
        assert result == []

    def test_stop_at_beginning(self):
        texts = ["<|im_end|>trailing"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == [""]

    def test_partial_prefix_single_char(self):
        texts = ["output<"]
        result = strip_stop_strings(texts, ["<|im_end|>"])
        assert result == ["output"]

    def test_chandra_stop_tokens(self):
        texts = ["<div>content</div><|endoftext|>"]
        result = strip_stop_strings(texts, ["<|im_end|>", "<|endoftext|>"])
        assert result == ["<div>content</div>"]


class TestComputeQwen2vlImageSize:
    """Tests for Qwen2.5-VL smart_resize replication."""

    def test_basic_rounding_to_factor(self):
        result = compute_qwen2vl_image_size(width=500, height=700)
        assert result.width % 28 == 0
        assert result.height % 28 == 0

    def test_exact_factor_multiple(self):
        result = compute_qwen2vl_image_size(width=560, height=560)
        assert result.width == 560
        assert result.height == 560

    def test_scale_factor(self):
        result_1x = compute_qwen2vl_image_size(width=500, height=700, scale=1.0)
        result_2x = compute_qwen2vl_image_size(width=500, height=700, scale=2.0)
        assert result_2x.width > result_1x.width
        assert result_2x.height > result_1x.height

    def test_max_size_clamp(self):
        result = compute_qwen2vl_image_size(width=2000, height=3000, max_size=1000)
        assert result.width <= 1008  # 1000 rounded up to nearest factor
        assert result.height <= 1008

    def test_max_pixels_clamp(self):
        result = compute_qwen2vl_image_size(
            width=3000, height=3000, max_pixels=1_000_000
        )
        assert result.width * result.height <= 1_000_000

    def test_min_pixels_upscale(self):
        result = compute_qwen2vl_image_size(width=100, height=100, min_pixels=200704)
        assert result.width * result.height >= 200704

    def test_typical_document_page(self):
        result = compute_qwen2vl_image_size(width=612, height=792)
        assert result.width % 28 == 0
        assert result.height % 28 == 0
        assert result.width * result.height >= 200704
        assert result.width * result.height <= 2_500_000

    def test_returns_size_type(self):
        result = compute_qwen2vl_image_size(width=500, height=700)
        assert isinstance(result, Size)

    def test_very_large_image_clamps(self):
        result = compute_qwen2vl_image_size(width=5000, height=5000)
        assert result.width * result.height <= 2_500_000

    def test_small_image_scales_up(self):
        result = compute_qwen2vl_image_size(width=50, height=50)
        assert result.width * result.height >= 200704

    def test_custom_factor(self):
        result = compute_qwen2vl_image_size(width=500, height=700, factor=14)
        assert result.width % 14 == 0
        assert result.height % 14 == 0

    def test_max_size_no_effect_when_smaller(self):
        result_no_clamp = compute_qwen2vl_image_size(width=500, height=700)
        result_with_clamp = compute_qwen2vl_image_size(
            width=500, height=700, max_size=2000
        )
        assert result_no_clamp == result_with_clamp
