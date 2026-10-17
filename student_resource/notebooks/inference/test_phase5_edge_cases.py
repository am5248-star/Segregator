"""Small deterministic edge tests for the Phase 3/5 helper behavior.

Run from the project root with:
    python notebooks/inference/test_phase5_edge_cases.py
"""
import ast
import math
from pathlib import Path
import unittest

import numpy as np


def load_pipeline_helpers():
    here = Path(__file__).resolve().parent
    phase3_path = here / "phase3_blocking_benchmark.py"
    if not phase3_path.is_file():
        phase3_path = here.parent / "validation" / "phase3_blocking_benchmark.py"
    phase5_path = here / "phase5_test_inference.py"
    namespace = {}
    phase3_source = phase3_path.read_text(encoding="utf-8")
    helper_source = phase3_source.split("\nroot = locate_root()\n", 1)[0]
    exec(compile(helper_source, str(phase3_path), "exec"), namespace)

    phase5_tree = ast.parse(phase5_path.read_text(encoding="utf-8"))
    wanted = {"build_route_candidates", "feature_matrix"}
    functions = [node for node in phase5_tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace.update({"math": math, "FEATURE_NAMES": [f"feature_{i}" for i in range(14)]})
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(phase5_path), "exec"), namespace)
    return namespace


PIPELINE = load_pipeline_helpers()


class Phase5EdgeCaseTests(unittest.TestCase):
    def test_unicode_and_punctuation_normalize(self):
        self.assertEqual(PIPELINE["normalize"]("  Ａcme, Inc.  "), "acme inc")

    def test_legal_suffix_only_name_has_empty_core(self):
        self.assertEqual(PIPELINE["canonical_name"]("Private Inc"), "")

    def test_street_and_unit_aliases(self):
        self.assertEqual(PIPELINE["canonical_address"]("10 Main Street, Apt. 2"),
                         "10 main st apt 2")

    def test_malformed_target_ids_are_rejected(self):
        self.assertEqual(PIPELINE["make_target_id"]("S2-7"), (2 << 32) | 7)
        for value in ("X2-7", "S2-4294967296", "S2-0007", "S1-7", "S3-x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PIPELINE["make_target_id"](value)

    def test_posting_limit_boundary(self):
        dtype = PIPELINE["DTYPE"]
        key = PIPELINE["hash64"]("common")
        accepted = np.array([(key, (2 << 32) | i) for i in range(200)], dtype=dtype)
        rejected = np.array([(key, (2 << 32) | i) for i in range(201)], dtype=dtype)
        self.assertEqual(len(PIPELINE["find_candidates"](accepted, key)), 200)
        self.assertEqual(len(PIPELINE["find_candidates"](rejected, key)), 0)

    def test_empty_name_does_not_match_empty_name_block(self):
        dtype = PIPELINE["DTYPE"]
        target_id = (2 << 32) | 9
        indexes = {name: np.array([], dtype=dtype) for name in (
            "address", "core", "unordered", "canonical_address", "name_token",
            "address_token", "char", "name_pair", "address_pair")}
        key = PIPELINE["index_key"]("", "US")
        indexes["name"] = np.array([(key, target_id)], dtype=dtype)
        row = {"country": "US", "business_name": "", "business_address": "A nonempty address"}
        _, candidates = PIPELINE["build_route_candidates"](row, indexes)
        self.assertEqual(candidates, set())

    def test_empty_address_does_not_match_empty_address_block(self):
        dtype = PIPELINE["DTYPE"]
        target_id = (3 << 32) | 12
        indexes = {name: np.array([], dtype=dtype) for name in (
            "name", "core", "unordered", "canonical_address", "name_token",
            "address_token", "char", "name_pair", "address_pair")}
        key = PIPELINE["index_key"]("", "France")
        indexes["address"] = np.array([(key, target_id)], dtype=dtype)
        row = {"country": "France", "business_name": "Valid Business", "business_address": ""}
        _, candidates = PIPELINE["build_route_candidates"](row, indexes)
        self.assertEqual(candidates, set())

    def test_empty_candidate_feature_matrix(self):
        ids, features = PIPELINE["feature_matrix"]([], set())
        self.assertEqual(ids.shape, (0,))
        self.assertEqual(features.shape, (0, 14))


if __name__ == "__main__":
    unittest.main(verbosity=2)
