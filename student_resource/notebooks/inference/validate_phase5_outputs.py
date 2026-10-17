"""Streaming validation of Phase 5 outputs with disk-backed target-ID lookup."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "dataset" / "test" / "test_source1.tsv").is_file())
test_dir = root / "dataset" / "test"
parser = argparse.ArgumentParser(description="Stream-validate Phase 5 inference outputs.")
parser.add_argument("--output-dir", type=Path, default=root / "output",
                    help="Directory containing candidate_pairs.tsv and matching_results.tsv")
args = parser.parse_args()
output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
source1_path = test_dir / "test_source1.tsv"
candidate_path = output_dir / "candidate_pairs.tsv"
matching_path = output_dir / "matching_results.tsv"


def count_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f, delimiter="\t"))


def parse_row(line, expected_sid, header, label):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2 or fields[0] != expected_sid:
        raise ValueError(f"{label}: malformed or misordered row for {expected_sid}")
    return [] if not fields[1] else fields[1].split(",")


target_paths = [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]
target_rows = sum(count_rows(path) for path in target_paths)
with tempfile.TemporaryDirectory(prefix="phase5_validate_", dir=root / "dataset" / "splits") as tmp:
    target_keys = np.memmap(Path(tmp) / "target_ids.bin", mode="w+", dtype="<u8", shape=(target_rows,))
    pos = 0
    for path in target_paths:
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                eid = row["entity_id"]
                source = int(eid[1])
                suffix = eid[3:]
                if source not in (2, 3) or eid != f"S{source}-{int(suffix)}":
                    raise ValueError(f"Target ID is not losslessly packable: {eid!r}")
                target_keys[pos] = (source << 32) | int(suffix)
                pos += 1
    if pos != target_rows:
        raise ValueError(f"Indexed {pos} targets but expected {target_rows}")
    target_keys.sort(kind="quicksort")
    target_keys.flush()

    candidate_ids_seen = 0
    match_ids_seen = 0
    empty_candidates = 0
    empty_matches = 0
    source1_seen = set()
    with source1_path.open(encoding="utf-8", newline="") as s1_file, \
            candidate_path.open(encoding="utf-8", newline="") as candidate_file, \
            matching_path.open(encoding="utf-8", newline="") as matching_file:
        s1_reader = csv.DictReader(s1_file, delimiter="\t")
        if candidate_file.readline().rstrip("\r\n") != "source1_entity_id\tcandidate_entity_ids":
            raise ValueError("Unexpected candidate file header")
        if matching_file.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("Unexpected matching file header")
        for row_no, s1_row in enumerate(s1_reader, start=1):
            sid = s1_row["entity_id"]
            if sid in source1_seen:
                raise ValueError(f"Duplicate Source 1 ID in test input: {sid}")
            source1_seen.add(sid)
            candidate_line = candidate_file.readline()
            matching_line = matching_file.readline()
            if not candidate_line or not matching_line:
                raise ValueError(f"Output ended before Source 1 row {row_no}: {sid}")
            candidates = parse_row(candidate_line, sid, None, "candidate file")
            matches = parse_row(matching_line, sid, None, "matching file")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"Duplicate candidate IDs for {sid}")
            if len(matches) != len(set(matches)):
                raise ValueError(f"Duplicate match IDs for {sid}")
            if not set(matches).issubset(candidates):
                raise ValueError(f"Predicted match absent from candidate set for {sid}")
            for values, label in ((candidates, "candidate"), (matches, "match")):
                packed = np.empty(len(values), dtype=np.uint64)
                for i, value in enumerate(values):
                    if not value.startswith(("S2-", "S3-")):
                        raise ValueError(f"Invalid {label} ID prefix: {value}")
                    source = int(value[1])
                    suffix = value[3:]
                    if not suffix.isdigit() or value != f"S{source}-{int(suffix)}":
                        raise ValueError(f"Invalid {label} ID format: {value}")
                    packed[i] = (source << 32) | int(suffix)
                if len(packed):
                    locations = np.searchsorted(target_keys, packed)
                    in_bounds = locations < target_rows
                    found = np.zeros(len(packed), dtype=bool)
                    found[in_bounds] = target_keys[locations[in_bounds]] == packed[in_bounds]
                    if not np.all(found):
                        bad = values[int(np.flatnonzero(~found)[0])]
                        raise ValueError(f"Unknown {label} ID: {bad}")
            candidate_ids_seen += len(candidates)
            match_ids_seen += len(matches)
            empty_candidates += not candidates
            empty_matches += not matches
        if candidate_file.readline() or matching_file.readline():
            raise ValueError("Output has more rows than test Source 1")

    print(json.dumps({
        "result": "PASS",
        "source1_rows": len(source1_seen),
        "candidate_pairs": candidate_ids_seen,
        "predicted_matches": match_ids_seen,
        "empty_candidate_rows": empty_candidates,
        "empty_match_rows": empty_matches,
        "targets_checked": target_rows,
        "checks": ["one ordered row per test Source 1 ID", "valid existing S2/S3 IDs",
                   "no duplicate IDs", "matches are a subset of candidates"],
    }, indent=2))
    target_keys.flush()
    target_keys._mmap.close()
