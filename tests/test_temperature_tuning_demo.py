import unittest

from temperature_tuning_demo import repetition_ratio


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


if __name__ == "__main__":
    unittest.main()
