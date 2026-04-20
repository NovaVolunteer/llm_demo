#!/usr/bin/env python3
"""Demonstrate how temperature affects LLM output repetition."""

from __future__ import annotations

import argparse
import re


def repetition_ratio(text: str) -> tuple[int, int, float]:
    """Return repeated trigram count, total trigrams, and repetition ratio."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 3:
        return 0, 0, 0.0

    trigrams = list(zip(words, words[1:], words[2:]))
    if not trigrams:
        return 0, 0, 0.0
    repeated = len(trigrams) - len(set(trigrams))
    return repeated, len(trigrams), repeated / len(trigrams)


def generate(
    model,
    tokenizer,
    torch_module,
    seed_setter,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> str:
    seed_setter(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch_module.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show how low temperature can increase repetitive generation."
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="Small free model from Hugging Face Hub (default: sshleifer/tiny-gpt2)",
    )
    parser.add_argument(
        "--prompt",
        default="A volunteer coordinator gives this pep talk to the team:",
        help="Prompt to continue",
    )
    parser.add_argument("--low-temperature", type=float, default=0.1)
    parser.add_argument("--high-temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "Missing dependency. Install with: pip install transformers torch"
        ) from exc

    print(f"Loading model '{args.model}' from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except OSError as exc:
        raise SystemExit(
            "Could not download model files from Hugging Face. "
            "Check network access or pass a local model path with --model."
        ) from exc

    low_text = generate(
        model,
        tokenizer,
        torch,
        set_seed,
        args.prompt,
        args.low_temperature,
        args.max_new_tokens,
        args.seed,
    )
    high_text = generate(
        model,
        tokenizer,
        torch,
        set_seed,
        args.prompt,
        args.high_temperature,
        args.max_new_tokens,
        args.seed,
    )

    low_repeat, low_total, low_ratio = repetition_ratio(low_text)
    high_repeat, high_total, high_ratio = repetition_ratio(high_text)

    print("\nPrompt:")
    print(args.prompt)

    print(f"\n--- Low temperature ({args.low_temperature}) ---")
    print(low_text)
    print(
        f"Repetition score: {low_repeat}/{low_total} repeated trigrams "
        f"({low_ratio:.1%})"
    )

    print(f"\n--- Higher temperature ({args.high_temperature}) ---")
    print(high_text)
    print(
        f"Repetition score: {high_repeat}/{high_total} repeated trigrams "
        f"({high_ratio:.1%})"
    )

    if low_ratio <= high_ratio:
        print(
            "\nTip: If repetition is not obvious, try a lower --low-temperature "
            "(e.g. 0.05) or larger --max-new-tokens."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
