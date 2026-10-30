# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Normalisation of the stop reasons that VLM inference engines report."""

from docling.datamodel.base_models import VlmStopReason

# Finish reasons used by OpenAI compatible chat completion endpoints. The model
# ran out of tokens / hit the context window, refused to answer because of a
# content policy, or the request was aborted before it could finish. vLLM's
# repetition detection cut a repetition loop, as a Docling repetition stopper
# does. All of them leave the output incomplete.
_INCOMPLETE_FINISH_REASONS = {
    "length": VlmStopReason.LENGTH,
    "max_tokens": VlmStopReason.LENGTH,
    "content_filter": VlmStopReason.CONTENT_FILTERED,
    "abort": VlmStopReason.INFERENCE_ERROR,
    "error": VlmStopReason.INFERENCE_ERROR,
    "repetition": VlmStopReason.STOP_SEQUENCE,
}

# Finish reasons that mean the model emitted its end-of-text token on its own,
# i.e. the output is complete.
_COMPLETE_FINISH_REASONS = {
    "stop": VlmStopReason.END_OF_SEQUENCE,
    "end": VlmStopReason.END_OF_SEQUENCE,
    "end_of_sequence": VlmStopReason.END_OF_SEQUENCE,
}

# A configured stop string is the proper end of the output, so a generation that
# stopped on one is complete.
_STOPPED_BY_REQUEST = {
    "stop_string": VlmStopReason.END_OF_SEQUENCE,
}


def normalize_stop_reason(stop_reason: str | VlmStopReason | None) -> VlmStopReason:
    """Map an engine specific stop reason onto :class:`VlmStopReason`.

    Engines do not agree on the vocabulary: OpenAI compatible endpoints report
    ``finish_reason`` values such as ``length`` or ``stop``, vLLM reports
    ``stop`` or ``abort``, while the local engines report the Docling names.
    A value that is not recognised is reported as
    :attr:`VlmStopReason.UNSPECIFIED` rather than guessed, so that an unknown
    reason never gets mistaken for a truncated output.
    """
    if isinstance(stop_reason, VlmStopReason):
        return stop_reason
    if not stop_reason:
        return VlmStopReason.UNSPECIFIED

    value = str(stop_reason).strip().lower()
    try:
        return VlmStopReason(value)
    except ValueError:
        pass
    if value in _INCOMPLETE_FINISH_REASONS:
        return _INCOMPLETE_FINISH_REASONS[value]
    if value in _COMPLETE_FINISH_REASONS:
        return _COMPLETE_FINISH_REASONS[value]
    if value in _STOPPED_BY_REQUEST:
        return _STOPPED_BY_REQUEST[value]
    return VlmStopReason.UNSPECIFIED
