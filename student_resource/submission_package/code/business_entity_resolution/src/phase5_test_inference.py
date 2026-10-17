"""Run the frozen Phase 3 blocking + Phase 4 matcher on every test S1 row.

The candidate TSV records exactly the packed target IDs passed to the Phase 4
feature builder/scorer. The matcher weights and threshold are read unchanged
from the completed Phase 4 training run.
"""
import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np


root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "dataset" / "test" / "test_source1.tsv").is_file())
phase3_path = Path(__file__).with_name("phase3_blocking_benchmark.py")
phase3_source = phase3_path.read_text(encoding="utf-8")
helper_source = phase3_source.split("\nroot = locate_root()\n", 1)[0]
exec(compile(helper_source, str(phase3_path), "exec"), globals())

metrics_path = Path(__file__).with_name("phase4_matcher_metrics.json")
phase4 = json.loads(metrics_path.read_text(encoding="utf-8"))
weights = np.asarray(phase4["model"]["weight_intercept_and_features"], dtype=np.float32)
threshold = float(phase4["probability_threshold"])
if len(weights) != 1 + 14:
    raise ValueError(f"Unexpected Phase 4 model feature count: {len(weights) - 1}")

FEATURE_NAMES = phase4["feature_names"]
MAX_POSTING = 200
MAX_CHAR_HITS = 3
STOPWORDS = {"the", "and", "for", "with", "from", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "plc", "pvt", "private", "public", "group", "holdings", "services"}
ADDRESS_STOPWORDS = STOPWORDS | {"street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "suite", "unit", "apt", "apartment", "floor", "near", "plot", "number", "building", "phase", "sector"}
DTYPE = np.dtype([("key", "<u8"), ("target", "<u8")])
LEGAL_ALIASES = {"incorporation": "inc", "incorporated": "inc", "corporation": "corp",
                 "limited": "ltd", "company": "co", "private": "pvt", "public": "pub"}
LEGAL_SUFFIXES = {"inc", "corp", "ltd", "co", "llc", "plc", "pvt", "pub", "gmbh", "sa", "sarl"}
STREET_ALIASES = {"street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
                  "lane": "ln", "drive": "dr", "highway": "hwy", "apartment": "apt"}


def build_route_candidates(row, indexes):
    country = row["country"].strip()
    business_name = row["business_name"]
    business_address = row["business_address"]
    name_tokens = token_list(business_name)
    address_tokens = token_list(business_address, ADDRESS_STOPWORDS)
    core = canonical_name(business_name)
    routes = [
        exact_candidates(indexes["name"], business_name, country),
        exact_candidates(indexes["address"], business_address, country),
        exact_candidates(indexes["core"], core, country),
        exact_candidates(indexes["unordered"], " ".join(sorted(core.split())), country),
        exact_candidates(indexes["canonical_address"], canonical_address(business_address), country),
        rare_token_candidates(indexes["name_token"], name_tokens),
        rare_token_candidates(indexes["address_token"], address_tokens),
        rare_token_candidates(indexes["name_token"], name_tokens, 4),
        rare_token_candidates(indexes["address_token"], address_tokens, 3),
        rare_token_candidates(indexes["char"], char_ngrams(business_name), MAX_CHAR_HITS),
        rare_token_candidates(indexes["name_pair"], token_pairs(top_tokens(business_name, STOPWORDS, 5)), 3),
        rare_token_candidates(indexes["address_pair"], token_pairs(top_tokens(business_address, ADDRESS_STOPWORDS, 4)), 3),
    ]
    return routes, set().union(*routes)


def feature_matrix(routes, candidate_ids):
    ids = np.asarray(sorted(candidate_ids), dtype=np.uint64)
    masks = {int(target_id): 0 for target_id in ids}
    for bit, route_ids in enumerate(routes):
        for target_id in route_ids:
            if target_id in masks:
                masks[target_id] |= 1 << bit
    mask_values = np.fromiter((masks[int(target_id)] for target_id in ids), dtype=np.uint16, count=len(ids))
    features = np.empty((len(ids), len(FEATURE_NAMES)), dtype=np.float32)
    for bit in range(12):
        features[:, bit] = (mask_values >> bit) & 1
    features[:, 12] = (ids >> 32) == 3
    features[:, 13] = math.log1p(len(ids)) / 10.0
    return ids, features


def predict_scores(features):
    logits = np.clip(weights[0] + features @ weights[1:], -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


def count_rows(path):
    source = int(path.stem.removeprefix("test_source")) if path.stem != "test_source1" else 1
    count = 0
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if source in (2, 3):
                suffix = row["entity_id"][3:]
                if row["entity_id"] != f"S{source}-{int(suffix)}":
                    raise ValueError(f"Target ID cannot be losslessly packed: {row['entity_id']!r}")
            count += 1
    return count


test_dir = root / "dataset" / "test"
source1_path = test_dir / "test_source1.tsv"
target_paths = {source: test_dir / f"test_source{source}.tsv" for source in (2, 3)}
source1_rows = count_rows(source1_path)
target_counts = {source: count_rows(path) for source, path in target_paths.items()}
target_rows = sum(target_counts.values())
print(f"Test Source 1 rows: {source1_rows:,}; targets: {target_counts}; total={target_rows:,}", flush=True)

parser = argparse.ArgumentParser(description="Run the frozen Phase 5 test inference pipeline.")
parser.add_argument("--output-dir", type=Path, default=root / "output",
                    help="Directory for candidate_pairs.tsv and matching_results.tsv")
args = parser.parse_args()
output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
candidate_path = output_dir / "candidate_pairs.tsv"
matching_path = output_dir / "matching_results.tsv"

with tempfile.TemporaryDirectory(prefix="er_phase5_test_", dir=root / "dataset" / "splits") as temp_name:
    temp_dir = Path(temp_name)
    arrays = {
        "name": np.memmap(temp_dir / "name.bin", mode="w+", dtype=DTYPE, shape=(target_rows,)),
        "address": np.memmap(temp_dir / "address.bin", mode="w+", dtype=DTYPE, shape=(target_rows,)),
        "core": np.memmap(temp_dir / "core.bin", mode="w+", dtype=DTYPE, shape=(target_rows,)),
        "unordered": np.memmap(temp_dir / "unordered.bin", mode="w+", dtype=DTYPE, shape=(target_rows,)),
        "canonical_address": np.memmap(temp_dir / "canonical_address.bin", mode="w+", dtype=DTYPE, shape=(target_rows,)),
        "name_token": np.memmap(temp_dir / "name_token.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 5,)),
        "address_token": np.memmap(temp_dir / "address_token.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 4,)),
        "name_pair": np.memmap(temp_dir / "name_pair.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 3,)),
        "address_pair": np.memmap(temp_dir / "address_pair.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 3,)),
        "char": np.memmap(temp_dir / "char.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 12,)),
    }
    counts = {key: 0 for key in arrays}
    for source, path in target_paths.items():
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                target_id = make_target_id(row["entity_id"])
                if row["entity_id"] != f"S{source}-{target_id & 0xffffffff}":
                    raise ValueError(f"Non-canonical test target ID: {row['entity_id']!r}")
                country = row["country"].strip()
                name = row["business_name"]
                address = row["business_address"]
                core = canonical_name(name)
                arrays["name"][counts["name"]] = (index_key(name, country), target_id)
                arrays["address"][counts["address"]] = (index_key(address, country), target_id)
                arrays["core"][counts["core"]] = (index_key(core, country), target_id)
                arrays["unordered"][counts["unordered"]] = (index_key(" ".join(sorted(core.split())), country), target_id)
                arrays["canonical_address"][counts["canonical_address"]] = (index_key(canonical_address(address), country), target_id)
                for token in top_tokens(name, STOPWORDS, 5):
                    arrays["name_token"][counts["name_token"]] = (hash64(token), target_id)
                    counts["name_token"] += 1
                for token in top_tokens(address, ADDRESS_STOPWORDS, 4):
                    arrays["address_token"][counts["address_token"]] = (hash64(token), target_id)
                    counts["address_token"] += 1
                for pair in token_pairs(top_tokens(name, STOPWORDS, 3)):
                    arrays["name_pair"][counts["name_pair"]] = (hash64(pair), target_id)
                    counts["name_pair"] += 1
                for pair in token_pairs(top_tokens(address, ADDRESS_STOPWORDS, 3)):
                    arrays["address_pair"][counts["address_pair"]] = (hash64(pair), target_id)
                    counts["address_pair"] += 1
                for gram in char_ngrams(name):
                    arrays["char"][counts["char"]] = (hash64(gram), target_id)
                    counts["char"] += 1
                for key in ("name", "address", "core", "unordered", "canonical_address"):
                    counts[key] += 1
        print(f"Indexed test targets for source {source}: {counts['name']:,}/{target_rows:,}", flush=True)
    if counts["name"] != target_rows:
        raise ValueError(f"Target count mismatch: expected {target_rows}, saw {counts['name']}")
    for key in ("name_token", "address_token", "name_pair", "address_pair", "char"):
        arrays[key] = arrays[key][:counts[key]]
    for key, array in arrays.items():
        print(f"Sorting {key} index ({len(array):,} postings)...", flush=True)
        array.sort(order="key", kind="quicksort")
        array.flush()

    indexes = arrays
    candidate_count = 0
    match_count = 0
    processed = 0
    with source1_path.open(encoding="utf-8", newline="") as source1_file, \
            candidate_path.open("w", encoding="utf-8", newline="") as candidate_file, \
            matching_path.open("w", encoding="utf-8", newline="") as matching_file:
        source1_reader = csv.DictReader(source1_file, delimiter="\t")
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")
        for row in source1_reader:
            sid = row["entity_id"]
            routes, candidate_ids = build_route_candidates(row, indexes)
            ids, features = feature_matrix(routes, candidate_ids)

            # Persist exactly the sorted candidate IDs presented to feature_matrix and scoring.
            candidate_names = [f"S{int(target_id) >> 32}-{int(target_id) & 0xffffffff}" for target_id in ids]
            candidate_file.write(sid + "\t" + ",".join(candidate_names) + "\n")

            scores = predict_scores(features)
            matched_names = [candidate_names[i] for i in np.flatnonzero(scores >= threshold)]
            matching_file.write(sid + "\t" + ",".join(matched_names) + "\n")
            candidate_count += len(ids)
            match_count += len(matched_names)
            processed += 1
            if processed % 100_000 == 0:
                print(f"Processed test Source 1: {processed:,}/{source1_rows:,}; candidates={candidate_count:,}; matches={match_count:,}", flush=True)
    if processed != source1_rows:
        raise ValueError(f"Source 1 row mismatch: expected {source1_rows}, wrote {processed}")
    print(json.dumps({
        "model": "frozen Phase 4 pairwise matcher",
        "blocking": "frozen Phase 3 union_token_pairs candidate routes",
        "probability_threshold": threshold,
        "test_source1_rows": processed,
        "candidate_pairs_passed_to_matcher": candidate_count,
        "predicted_matches": match_count,
        "candidate_file": str(candidate_path),
        "matching_file": str(matching_path),
    }, indent=2), flush=True)
    for array in arrays.values():
        array.flush()
        if isinstance(array, np.memmap):
            array._mmap.close()
