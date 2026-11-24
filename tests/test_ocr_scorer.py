import unittest
from unittest.mock import Mock

from scorers.ocr import OcrScorer


class OCRScorerTest(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.scorer = OcrScorer(self.backend)
        self.image = object()

    def score(self, prompt='A sign that reads "OPEN"'):
        return self.scorer([self.image], [prompt])[0]

    def test_exact_match(self):
        self.backend.ocr.return_value = [[([], ("Op en", 0.99))]]
        self.assertEqual(self.score(), 1.0)
        self.backend.ocr.assert_called_once_with(self.image, cls=False)

    def test_partial_match(self):
        for recognized in ("OPAL", "OPE", "OAPEN", "PEN O"):
            with self.subTest(recognized=recognized):
                self.backend.ocr.return_value = [[([], (recognized, 0.99))]]
                expected = 0.75 if recognized in ("OPE", "OAPEN") else 0.5
                self.assertEqual(self.score(), expected)

    def test_empty_detections(self):
        for result in (None, [], [None], [[]]):
            with self.subTest(result=result):
                self.backend.ocr.return_value = result
                self.assertEqual(self.score(), 0.0)

    def test_backend_error_propagates(self):
        error = RuntimeError("backend failed")
        self.backend.ocr.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.score()
        self.assertIs(raised.exception, error)

    def test_invalid_targets(self):
        for prompt in ('A sign', 'A sign that reads "OPEN', 'A sign reading ""', 'A sign reading "  "'):
            with self.subTest(prompt=prompt):
                with self.assertRaisesRegex(ValueError, "non-empty quoted text"):
                    self.score(prompt)
        self.backend.ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
