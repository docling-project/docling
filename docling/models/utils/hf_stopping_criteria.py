import logging

from transformers import StoppingCriteria

from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


class HFStoppingCriteriaWrapper(StoppingCriteria):
    """
    Adapts any GenerationStopper to HuggingFace Transformers.
    Decodes exactly min(seq_len, stopper.lookback_tokens()) tokens from the end.
    """

    def __init__(
        self,
        tokenizer,
        stopper: GenerationStopper,
        *,
        skip_special_tokens: bool = False,
    ):
        self.tokenizer = tokenizer
        self.stopper = stopper
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        lb = max(1, int(self.stopper.lookback_tokens()))
        for seq in input_ids:  # (batch, seq_len)
            window = seq[-lb:]  # slicing handles lb > len(seq)
            try:
                text = self.tokenizer.decode(
                    window, skip_special_tokens=self.skip_special_tokens
                )
            except Exception as e:
                _log.info(f"Decoding failed for stopping check: {e}")
                continue

            try:
                if self.stopper.should_stop(text):
                    _log.info(
                        "HF wrapper: stopping due to TextStopper.should_stop==True"
                    )
                    return True
            except Exception as e:
                _log.info(f"Error in TextStopper.should_stop: {e}")
                continue
        return False
