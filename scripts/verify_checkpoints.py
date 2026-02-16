from apt.models import (
    APTLanguageModel,
    APTTokenizer,
    default_language_model_checkpoint,
    default_tokenizer_checkpoint,
)


def main() -> None:
    tokenizer_ckpt = default_tokenizer_checkpoint()
    lm_ckpt = default_language_model_checkpoint()

    if not tokenizer_ckpt.exists():
        raise FileNotFoundError(f"Missing tokenizer checkpoint: {tokenizer_ckpt}")
    if not lm_ckpt.exists():
        raise FileNotFoundError(f"Missing language model checkpoint: {lm_ckpt}")

    print(f"Loading tokenizer checkpoint: {tokenizer_ckpt}")
    tokenizer = APTTokenizer.from_pretrained(tokenizer_ckpt)
    print(f"Tokenizer params: {tokenizer.num_params():,}")

    print(f"Loading language model checkpoint: {lm_ckpt}")
    model = APTLanguageModel.from_pretrained(lm_ckpt, tokenizer_ckpt)
    print(f"Language model params (trainable): {model.num_params():,}")


if __name__ == "__main__":
    main()
