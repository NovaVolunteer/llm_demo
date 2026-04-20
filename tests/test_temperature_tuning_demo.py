import unittest

from temperature_tuning_demo import display_text, generate, repetition_ratio


class FakeTokenIds(list):
    @property
    def shape(self):
        return (1, len(self))


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.last_decoded_tokens = None

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": FakeTokenIds([10, 11, 12])}

    def decode(self, tokens, skip_special_tokens=True):
        self.last_decoded_tokens = list(tokens)
        return "generated continuation"


class FakeModel:
    def __init__(self, output_tokens):
        self.output_tokens = output_tokens

    def generate(self, **kwargs):
        return [self.output_tokens]


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    @staticmethod
    def no_grad():
        return FakeNoGrad()


class RepetitionRatioTests(unittest.TestCase):
    def test_detects_repeated_trigrams(self):
        repeated, total, ratio = repetition_ratio("go go go go")
        self.assertEqual(repeated, 1)
        self.assertEqual(total, 2)
        self.assertAlmostEqual(ratio, 0.5)

    def test_no_repetition_returns_zero_ratio(self):
        repeated, total, ratio = repetition_ratio("we help our community together")
        self.assertEqual(repeated, 0)
        self.assertEqual(total, 3)
        self.assertEqual(ratio, 0.0)

    def test_short_or_non_word_text_returns_zeroes(self):
        for text in ("", "help others", "!!! ###"):
            with self.subTest(text=text):
                repeated, total, ratio = repetition_ratio(text)
                self.assertEqual(repeated, 0)
                self.assertEqual(total, 0)
                self.assertEqual(ratio, 0.0)


class GenerateTests(unittest.TestCase):
    def test_decodes_only_new_tokens(self):
        tokenizer = FakeTokenizer()
        model = FakeModel([10, 11, 12, 21, 22])
        seeded = []

        def seed_setter(value):
            seeded.append(value)

        text = generate(
            model=model,
            tokenizer=tokenizer,
            torch_module=FakeTorch,
            seed_setter=seed_setter,
            prompt="Prompt text",
            temperature=0.7,
            max_new_tokens=10,
            seed=99,
        )

        self.assertEqual(text, "generated continuation")
        self.assertEqual(tokenizer.last_decoded_tokens, [21, 22])
        self.assertEqual(seeded, [99])

    def test_returns_empty_when_no_new_tokens(self):
        tokenizer = FakeTokenizer()
        model = FakeModel([10, 11, 12])

        text = generate(
            model=model,
            tokenizer=tokenizer,
            torch_module=FakeTorch,
            seed_setter=lambda _: None,
            prompt="Prompt text",
            temperature=0.7,
            max_new_tokens=10,
            seed=99,
        )

        self.assertEqual(text, "")
        self.assertIsNone(tokenizer.last_decoded_tokens)


class DisplayTextTests(unittest.TestCase):
    def test_returns_clean_visible_text(self):
        self.assertEqual(display_text("  hello world  "), "hello world")

    def test_returns_placeholder_for_empty_or_whitespace(self):
        placeholder = "[No visible text generated at this temperature/seed.]"
        self.assertEqual(display_text(""), placeholder)
        self.assertEqual(display_text("   \n\t  "), placeholder)


if __name__ == "__main__":
    unittest.main()
