"""Full validation-set benchmark for scalable, label-blind blocking routes."""
import csv, hashlib, heapq, json, re, tempfile, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

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

def locate_root():
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "dataset" / "train" / "train_source1.tsv").is_file():
            return parent
    raise FileNotFoundError("Could not locate dataset/train")

def normalize(value):
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in value).split())

def hash64(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=b"ER-block-v1").digest(), "little")

def token_list(value, stopwords=STOPWORDS):
    return [t for t in normalize(value).split() if len(t) >= 3 and t not in stopwords]

def canonical_name(value):
    tokens = [LEGAL_ALIASES.get(t, t) for t in normalize((value or "").replace("&", " and ")).split()]
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)

def canonical_address(value):
    tokens = [STREET_ALIASES.get(t, t) for t in normalize(value).split()]
    return " ".join(tokens)

def char_ngrams(value):
    tokens = sorted(set(token_list(value)), key=lambda t: (-len(t), t))[:2]
    grams = set()
    for token in tokens:
        if len(token) >= 4:
            windows = [token[i:i+3] for i in range(len(token)-2)]
            if len(windows) > 6:
                picks = np.linspace(0, len(windows)-1, 6, dtype=int)
                windows = [windows[i] for i in picks]
            grams.update(windows)
    return grams

def make_target_id(entity_id):
    if not isinstance(entity_id, str) or not entity_id.startswith(("S2-", "S3-")):
        raise ValueError(f"Malformed target entity_id: {entity_id!r}")
    source = int(entity_id[1])
    suffix = entity_id[3:]
    if not suffix.isdigit():
        raise ValueError(f"Malformed target entity_id: {entity_id!r}")
    suffix_value = int(suffix)
    if suffix_value > 0xffffffff or entity_id != f"S{source}-{suffix_value}":
        raise ValueError(f"Target entity_id cannot be packed losslessly: {entity_id!r}")
    return (source << 32) | suffix_value

def index_key(value, country):
    return hash64(country.strip().casefold() + "\x1f" + normalize(value))

def posting_bounds(index, key):
    keys = index["key"]
    left = int(np.searchsorted(keys, np.uint64(key), side="left"))
    right = int(np.searchsorted(keys, np.uint64(key), side="right"))
    return left, right

def find_candidates(index, key):
    left, right = posting_bounds(index, key)
    if right - left == 0 or right - left > MAX_POSTING:
        return ()
    return index["target"][left:right]

def exact_candidates(index, value, country):
    """Do not let an empty or punctuation-only field form a blocking key."""
    if not normalize(value):
        return set()
    return set(map(int, find_candidates(index, index_key(value, country))))

def rare_token_candidates(index, values, max_hits=2):
    eligible = []
    for token in set(values):
        key = hash64(token)
        left, right = posting_bounds(index, key)
        count = right - left
        if 0 < count <= MAX_POSTING:
            eligible.append((count, token, key))
    candidates = set()
    for _, _, key in sorted(eligible)[:max_hits]:
        candidates.update(map(int, find_candidates(index, key)))
    return candidates

def top_tokens(value, stopwords, limit):
    return sorted(set(token_list(value, stopwords)), key=lambda token: (-len(token), token))[:limit]

def token_pairs(values):
    ordered = sorted(set(values))
    return [left + "\x1f" + right for i, left in enumerate(ordered) for right in ordered[i+1:]]

def new_metrics():
    empty = lambda: {"candidate_pairs": 0, "true_pairs_recalled": 0, "true_pairs_total": 0,
                     "positive_s1_with_any_hit": 0, "positive_s1_total": 0}
    return {"all": empty(), "by_source": {"S2": empty(), "S3": empty()},
            "candidate_count_histogram": Counter()}

def add_route(metrics, candidate_ids, truth_ids):
    metrics["candidate_count_histogram"][str(len(candidate_ids))] += 1
    for label, source_code in (("S2", 2), ("S3", 3)):
        target_candidates = {target for target in candidate_ids if target >> 32 == source_code}
        target_truth = {target for target in truth_ids if target >> 32 == source_code}
        bucket = metrics["by_source"][label]
        bucket["candidate_pairs"] += len(target_candidates)
        bucket["true_pairs_total"] += len(target_truth)
        hits = len(target_candidates.intersection(target_truth))
        bucket["true_pairs_recalled"] += hits
        bucket["positive_s1_total"] += bool(target_truth)
        bucket["positive_s1_with_any_hit"] += bool(target_truth) and hits > 0
    bucket = metrics["all"]
    bucket["candidate_pairs"] += len(candidate_ids)
    bucket["true_pairs_total"] += len(truth_ids)
    hits = len(candidate_ids.intersection(truth_ids))
    bucket["true_pairs_recalled"] += hits
    bucket["positive_s1_total"] += bool(truth_ids)
    bucket["positive_s1_with_any_hit"] += bool(truth_ids) and hits > 0

def summarize_bucket(bucket, total_possible):
    pairs = bucket["candidate_pairs"]
    total_true = bucket["true_pairs_total"]
    positive_entities = bucket["positive_s1_total"]
    hit_entities = bucket["positive_s1_with_any_hit"]
    recalled = bucket["true_pairs_recalled"]
    return {"candidate_pairs": pairs,
            "candidate_pair_density": pairs / total_possible,
            "reduction_ratio": 1.0 - pairs / total_possible,
            "true_target_recall": recalled / total_true if total_true else 0.0,
            "positive_source1_coverage": hit_entities / positive_entities if positive_entities else 0.0,
            "candidate_precision_diagnostic": recalled / pairs if pairs else 0.0,
            "true_target_count": total_true,
            "positive_source1_count": positive_entities}

def finish_metrics(metrics, total_possible, source_possible):
    return {"all_sources": summarize_bucket(metrics["all"], total_possible),
            "by_source": {source: summarize_bucket(metrics["by_source"][source], source_possible[source]) for source in ("S2", "S3")},
            "candidate_count_histogram": dict(sorted(metrics["candidate_count_histogram"].items(), key=lambda x: int(x[0])))}

root = locate_root()
train_dir = root / "dataset" / "train"
split_file = root / "dataset" / "splits" / "validation_source1_ids.txt"
validation_ids = set(split_file.read_text(encoding="utf-8").splitlines()[1:])
# Ground truth is loaded for validation IDs only; it is never used to construct blocks.
truth = {}
with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sid = row["source1_entity_id"]
        if sid in validation_ids:
            truth[sid] = {make_target_id(v) for v in row["matched_entity_ids"].split(",") if v}
if truth.keys() != validation_ids:
    raise ValueError("Validation split and ground-truth labels do not align")

# Preallocate compact memory-mapped indexes.
# Row counts verified during Phase 1; avoid an extra full scan before indexing.
# The index build below checks that the actual files still match these counts.
target_rows = 5_034_616 + 5_285_603

with tempfile.TemporaryDirectory(prefix="er_phase3_", dir=root / "dataset" / "splits") as temp:
    temp = Path(temp)
    name_idx = np.memmap(temp / "name.bin", mode="w+", dtype=DTYPE, shape=(target_rows,))
    address_idx = np.memmap(temp / "address.bin", mode="w+", dtype=DTYPE, shape=(target_rows,))
    token_idx = np.memmap(temp / "token.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 5,))
    address_token_idx = np.memmap(temp / "address_token.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 4,))
    core_name_idx = np.memmap(temp / "core_name.bin", mode="w+", dtype=DTYPE, shape=(target_rows,))
    unordered_name_idx = np.memmap(temp / "unordered_name.bin", mode="w+", dtype=DTYPE, shape=(target_rows,))
    canonical_address_idx = np.memmap(temp / "canonical_address.bin", mode="w+", dtype=DTYPE, shape=(target_rows,))
    name_pair_idx = np.memmap(temp / "name_pair.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 3,))
    address_pair_idx = np.memmap(temp / "address_pair.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 3,))
    char_idx = np.memmap(temp / "char.bin", mode="w+", dtype=DTYPE, shape=(target_rows * 12,))
    i = t = a_t = c_t = n_p = a_p = 0
    leading_zero_ids = 0
    for source in (2, 3):
        path = train_dir / f"train_source{source}.tsv"
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                entity_id = row["entity_id"]
                packed = make_target_id(entity_id)
                suffix = entity_id[3:]
                leading_zero_ids += len(suffix) > 1 and suffix.startswith("0")
                country = row["country"].strip()
                name_idx[i] = (index_key(row["business_name"], country), packed)
                address_idx[i] = (index_key(row["business_address"], country), packed)
                core = canonical_name(row["business_name"])
                core_name_idx[i] = (index_key(core, country), packed)
                unordered_name_idx[i] = (index_key(" ".join(sorted(core.split())), country), packed)
                canonical_address_idx[i] = (index_key(canonical_address(row["business_address"]), country), packed)
                name_tokens = top_tokens(row["business_name"], STOPWORDS, 5)
                address_tokens = top_tokens(row["business_address"], ADDRESS_STOPWORDS, 4)
                for token in name_tokens:
                    token_idx[t] = (hash64(token), packed)
                    t += 1
                for token in address_tokens:
                    address_token_idx[a_t] = (hash64(token), packed)
                    a_t += 1
                for pair in token_pairs(name_tokens[:3]):
                    name_pair_idx[n_p] = (hash64(pair), packed)
                    n_p += 1
                for pair in token_pairs(address_tokens[:3]):
                    address_pair_idx[a_p] = (hash64(pair), packed)
                    a_p += 1
                for gram in char_ngrams(row["business_name"]):
                    char_idx[c_t] = (hash64(gram), packed)
                    c_t += 1
                i += 1
                if i % 1_000_000 == 0:
                    print(f"Indexed target records: {i:,}/{target_rows:,}", flush=True)
    if i != target_rows:
        raise ValueError(f"Target index count mismatch: expected {target_rows}, saw {i}")
    token_idx = token_idx[:t]
    address_token_idx = address_token_idx[:a_t]
    char_idx = char_idx[:c_t]
    name_pair_idx = name_pair_idx[:n_p]
    address_pair_idx = address_pair_idx[:a_p]
    for label, index in (("name", name_idx), ("address", address_idx), ("core-name", core_name_idx), ("unordered-name", unordered_name_idx), ("canonical-address", canonical_address_idx), ("name-token", token_idx), ("address-token", address_token_idx), ("name-token-pair", name_pair_idx), ("address-token-pair", address_pair_idx), ("char-trigram", char_idx)):
        print(f"Sorting {label} index ({len(index):,} postings)...", flush=True)
        index.sort(order="key", kind="quicksort")
        index.flush()

    route_names = ("exact_name_country", "exact_address_country", "core_name_country", "unordered_core_name_country", "canonical_address_country", "rare_name_tokens_global", "rare_address_tokens_global", "name_char_trigrams_global", "expanded_name_tokens_global", "expanded_address_tokens_global", "name_token_pairs_global", "address_token_pairs_global", "union", "union_expanded", "union_token_pairs")
    route_stats = {name: new_metrics() for name in route_names}
    total_true_by_source = Counter()
    missed_sample_heaps = {"S2": [], "S3": []}
    rows = 0
    total_possible = len(validation_ids) * target_rows
    with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["entity_id"]
            if sid not in validation_ids:
                continue
            rows += 1
            true_ids = truth[sid]
            for target in true_ids:
                total_true_by_source[target >> 32] += 1
            country = row["country"].strip()
            core = canonical_name(row["business_name"])
            unordered_core = " ".join(sorted(core.split()))
            canonical_addr = canonical_address(row["business_address"])
            name_candidates = exact_candidates(name_idx, row["business_name"], country)
            address_candidates = exact_candidates(address_idx, row["business_address"], country)
            core_name_candidates = exact_candidates(core_name_idx, core, country)
            unordered_name_candidates = exact_candidates(unordered_name_idx, unordered_core, country)
            canonical_address_candidates = exact_candidates(canonical_address_idx, canonical_addr, country)
            name_token_candidates = rare_token_candidates(token_idx, token_list(row["business_name"]))
            address_token_candidates = rare_token_candidates(address_token_idx, token_list(row["business_address"], ADDRESS_STOPWORDS))
            expanded_name_candidates = rare_token_candidates(token_idx, token_list(row["business_name"]), 4)
            expanded_address_candidates = rare_token_candidates(address_token_idx, token_list(row["business_address"], ADDRESS_STOPWORDS), 3)
            name_pair_candidates = rare_token_candidates(name_pair_idx, token_pairs(top_tokens(row["business_name"], STOPWORDS, 5)), 3)
            address_pair_candidates = rare_token_candidates(address_pair_idx, token_pairs(top_tokens(row["business_address"], ADDRESS_STOPWORDS, 4)), 3)
            char_candidates = rare_token_candidates(char_idx, char_ngrams(row["business_name"]), MAX_CHAR_HITS)
            union = name_candidates | address_candidates | core_name_candidates | unordered_name_candidates | canonical_address_candidates | name_token_candidates | address_token_candidates | char_candidates
            union_expanded = union | expanded_name_candidates | expanded_address_candidates
            union_token_pairs = union_expanded | name_pair_candidates | address_pair_candidates
            for route, candidates in (("exact_name_country", name_candidates),
                                      ("exact_address_country", address_candidates),
                                      ("core_name_country", core_name_candidates),
                                      ("unordered_core_name_country", unordered_name_candidates),
                                      ("canonical_address_country", canonical_address_candidates),
                                      ("rare_name_tokens_global", name_token_candidates),
                                      ("rare_address_tokens_global", address_token_candidates),
                                      ("name_char_trigrams_global", char_candidates),
                                      ("expanded_name_tokens_global", expanded_name_candidates),
                                      ("expanded_address_tokens_global", expanded_address_candidates),
                                      ("name_token_pairs_global", name_pair_candidates),
                                      ("address_token_pairs_global", address_pair_candidates),
                                      ("union", union),
                                      ("union_expanded", union_expanded),
                                      ("union_token_pairs", union_token_pairs)):
                add_route(route_stats[route], candidates, true_ids)
            for label, source_code in (("S2", 2), ("S3", 3)):
                missed = sorted(target for target in true_ids - union_token_pairs if target >> 32 == source_code)
                if missed:
                    rank = hash64(f"{sid}\x1f{label}")
                    heap = missed_sample_heaps[label]
                    sample = (-rank, sid, missed[0], row["business_name"], row["business_address"], country)
                    if len(heap) < 100:
                        heapq.heappush(heap, sample)
                    elif sample > heap[0]:
                        heapq.heapreplace(heap, sample)
            if rows % 100000 == 0:
                print(f"Scored validation queries: {rows:,}/{len(validation_ids):,}", flush=True)

    if rows != len(validation_ids):
        raise ValueError(f"Validation query count mismatch: expected {len(validation_ids)}, saw {rows}")
    # Candidate universe compares each validation entity with each combined S2+S3 target.
    total_possible = len(validation_ids) * target_rows
    results = {"validation_source1_rows": rows,
               "target_rows_source2_source3": target_rows,
               "target_rows_by_source": {"S2": 5_034_616, "S3": 5_285_603},
               "max_possible_pairs": total_possible,
               "max_posting_list": MAX_POSTING,
               "normalization": "Unicode NFKC + casefold + alphanumeric tokens; no transliteration or external lookup",
               "country_policy": "exact country in name/address blocks; global name-token fallback; country labels are never hard-coded",
               "target_ids_with_leading_zero_suffix": leading_zero_ids,
               "true_matches_by_source": {f"S{source}": total for source, total in sorted(total_true_by_source.items())},
               "routes": {route: finish_metrics(metrics, total_possible, {
                   "S2": len(validation_ids) * 5_034_616,
                   "S3": len(validation_ids) * 5_285_603}) for route, metrics in route_stats.items()}}
    output = root / "dataset" / "splits" / "phase3_blocking_metrics.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    selected_targets = {sample[2] for heap in missed_sample_heaps.values() for sample in heap}
    target_details = {}
    for source in (2, 3):
        with (train_dir / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                packed = make_target_id(row["entity_id"])
                if packed in selected_targets:
                    target_details[packed] = row
                    if len(target_details) == len(selected_targets):
                        break
    examples = []
    for label, heap in missed_sample_heaps.items():
        for _, sid, target_id, source1_name, source1_address, source1_country in sorted(heap, reverse=True):
            target = target_details.get(target_id)
            if target is None:
                continue
            examples.append({"target_source": label, "source1_entity_id": sid,
                             "target_entity_id": target["entity_id"],
                             "source1": {"business_name": source1_name, "business_address": source1_address, "country": source1_country},
                             "target": {"business_name": target["business_name"], "business_address": target["business_address"], "country": target["country"]},
                             "diagnostics": {"same_country": source1_country.strip().casefold() == target["country"].strip().casefold(),
                                             "exact_normalized_name": normalize(source1_name) == normalize(target["business_name"]),
                                             "canonical_name": canonical_name(source1_name) == canonical_name(target["business_name"]),
                                             "exact_normalized_address": normalize(source1_address) == normalize(target["business_address"]),
                                             "canonical_address": canonical_address(source1_address) == canonical_address(target["business_address"]),
                                             "source1_address_blank": not normalize(source1_address),
                                             "target_address_blank": not normalize(target["business_address"])}})
    examples_output = root / "dataset" / "splits" / "phase3_missed_examples.json"
    examples_output.write_text(json.dumps({"sample_size": len(examples), "sampling": "deterministic hash-ranked sample of validation entities missed by union_token_pairs", "examples": examples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console_results = json.loads(json.dumps(results))
    for route in console_results["routes"].values():
        route.pop("candidate_count_histogram", None)
    print(json.dumps(console_results, indent=2), flush=True)
    print("Metrics saved:", output, flush=True)
    print("Missed-link examples saved:", examples_output, flush=True)
    for index in (name_idx, address_idx, core_name_idx, unordered_name_idx, canonical_address_idx, token_idx, address_token_idx, name_pair_idx, address_pair_idx, char_idx):
        index.flush()
        if isinstance(index, np.memmap):
            index._mmap.close()
