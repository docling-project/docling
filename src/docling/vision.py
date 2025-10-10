from typing import Optional, List

class PictureDescriptionVlmOptions:
    def __init__(
        self,
        prompt: Optional[str] = None,
        min_word_count: int = 10,
        retry_on_failure: bool = True,
        example_descriptions: Optional[List[str]] = None,
    ):
        self.prompt = prompt
        self.min_word_count = min_word_count
        self.retry_on_failure = retry_on_failure
        self.example_descriptions = example_descriptions or []


def is_insufficient(description: str, min_word_count: int = 10) -> bool:
    desc = description.strip().lower()
    if len(desc.split()) < min_word_count:
        return True
    if desc in {"in", "this", "the", "", "a", "an"}:
        return True
    return False


def build_retry_prompt(previous_desc: str, examples: List[str]) -> str:
    example_text = ""
    if examples:
        example_text = "Examples of good descriptions:\n" + "\n".join(
            f"- {ex}" for ex in examples
        ) + "\n"
    return (
        f"{example_text}"
        f"Your previous description was: \"{previous_desc.strip()}\"\n"
        "That description is too short or vague.\n"
        "Please describe this image clearly in a few sentences, "
        "including important labels, structures, or key features."
    )


def do_picture_description(
    vlm,
    image,
    options: PictureDescriptionVlmOptions = PictureDescriptionVlmOptions(),
) -> str:
    base_prompt = options.prompt or "Describe this image in a few sentences."

    description = vlm.describe(image, prompt=base_prompt)
    print(f"First description attempt:\n{description}\n")

    if options.retry_on_failure and is_insufficient(description, options.min_word_count):
        retry_prompt = build_retry_prompt(description, options.example_descriptions)
        description = vlm.describe(image, prompt=retry_prompt)
        print(f"Retry description attempt:\n{description}\n")

    return description


# ----- Dummy VLM to simulate describe() -----
class DummyVLM:
    def describe(self, image, prompt=None):
        if prompt and "previous description" in prompt:
            # Simulate better retry description
            return (
                "This image is a detailed diagram showing sensor response curves "
                "with labeled axes and multiple plotted lines representing different sensors."
            )
        # Simulate poor initial description
        return "This"


# ------ Run example --------
if __name__ == "__main__":
    vlm = DummyVLM()
    dummy_image = "fake_image_data"

    options = PictureDescriptionVlmOptions(
        min_word_count=10,
        retry_on_failure=True,
        example_descriptions=[
            "A graph showing sensor response curves with labels for each sensor.",
            "A diagram depicting the flow of data between components with annotations."
        ],
    )

    final_description = do_picture_description(vlm, dummy_image, options)
    print("Final description returned:\n", final_description)
