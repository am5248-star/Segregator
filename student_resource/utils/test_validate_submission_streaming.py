"""Unit tests for the validator's low-memory candidate subset diagnostic."""
import tempfile
import unittest
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from validate_submission import validate_candidate_subset_streaming


class StreamingSubsetTests(unittest.TestCase):
    def run_check(self, matching_rows, candidate_rows):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            matching = Path(tmp) / "matching.tsv"
            candidate = Path(tmp) / "candidate.tsv"
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\n" + matching_rows,
                encoding="utf-8",
            )
            candidate.write_text(
                "source1_entity_id\tcandidate_entity_ids\n" + candidate_rows,
                encoding="utf-8",
            )
            warnings = []
            validate_candidate_subset_streaming(matching, candidate, warnings)
            return warnings

    def test_aligned_subset_passes(self):
        warnings = self.run_check(
            "S1-1\tS2-3\nS1-2\t\n",
            "S1-1\tS2-3,S3-4\nS1-2\t\n",
        )
        self.assertEqual(warnings, [])

    def test_missing_candidate_is_reported(self):
        warnings = self.run_check("S1-1\tS2-9\n", "S1-1\tS2-3\n")
        self.assertEqual(len(warnings), 1)
        self.assertIn("absent from their candidate lists", warnings[0])

    def test_misaligned_rows_skip_subset_check_with_warning(self):
        warnings = self.run_check("S1-1\tS2-3\n", "S1-2\tS2-3\n")
        self.assertEqual(len(warnings), 1)
        self.assertIn("not in the same Source 1 row order", warnings[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
