"""Train and evaluate a candidate-conditioned logistic pair scorer.

Features are label-blind blocking-route evidence, target source, and per-query
candidate count. Candidate pairs for training come only from non-validation
Source 1 rows. A deterministic query-level calibration subset selects the score
threshold; the fixed Phase 2 validation split is evaluated once at that threshold.
"""
from pathlib import Path
import heapq, json, math

project_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / "dataset" / "train" / "train_source1.tsv").is_file())
phase3_path = Path(__file__).with_name("phase3_blocking_benchmark.py")
phase3_source = phase3_path.read_text(encoding="utf-8")
helper_source = phase3_source.split("\nroot = locate_root()\n", 1)[0]
exec(compile(helper_source, str(phase3_path), "exec"), globals())

MAX_TRAIN_QUERIES = 20_000
NEGATIVES_PER_QUERY = 30
TRAIN_SPLIT = "ML-CHALLENGE-2026-PHASE4-TRAIN-1"
FEATURE_NAMES = ["exact_name", "exact_address", "core_name", "unordered_name",
                 "canonical_address", "rare_name_token", "rare_address_token",
                 "expanded_name_token", "expanded_address_token", "char_trigram",
                 "name_token_pair", "address_token_pair", "target_is_s3",
                 "log1p_candidate_count_div10"]

def entity_f05(true_ids, predicted_ids):
    true_ids, predicted_ids = set(true_ids), set(predicted_ids)
    if not true_ids and not predicted_ids:
        return 1.0
    tp = len(true_ids & predicted_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)
    denominator = 1.25 * tp + fp + 0.25 * fn
    return 1.25 * tp / denominator if denominator else 0.0

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
    candidates = set().union(*routes)
    return routes, candidates

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

def fit_logistic(features, labels, iterations=80, learning_rate=0.35, l2=0.01):
    X = np.column_stack([np.ones(len(features), dtype=np.float32), features]).astype(np.float32, copy=False)
    y = labels.astype(np.float32, copy=False)
    weights = np.zeros(X.shape[1], dtype=np.float32)
    for _ in range(iterations):
        logits = np.clip(X @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (X.T @ (probabilities - y)) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights

def predict_scores(features, weights):
    logits = np.clip(weights[0] + features @ weights[1:], -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))

def macro_scores_at_threshold(records, thresholds):
    totals = np.zeros(len(thresholds), dtype=np.float64)
    for scores, labels, true_count in records:
        order = np.argsort(-scores)
        sorted_scores = scores[order]
        cumulative_tp = np.cumsum(labels[order], dtype=np.int64)
        predicted_count = np.searchsorted(-sorted_scores, -thresholds, side="right")
        tp = np.zeros(len(thresholds), dtype=np.int64)
        active = predicted_count > 0
        tp[active] = cumulative_tp[predicted_count[active] - 1]
        fp = predicted_count - tp
        fn = true_count - tp
        denominator = 1.25 * tp + fp + 0.25 * fn
        scores_by_threshold = np.divide(1.25 * tp, denominator, out=np.ones_like(denominator, dtype=float), where=denominator != 0)
        totals += scores_by_threshold
    return totals / len(records)

root = project_root
train_dir = root / "dataset" / "train"
split_dir = root / "dataset" / "splits"
validation_ids = set((split_dir / "validation_source1_ids.txt").read_text(encoding="utf-8").splitlines()[1:])

# Select a deterministic training-query sample from the complement of validation.
sample_heap = []
with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sid = row["entity_id"]
        if sid in validation_ids:
            continue
        priority = hash64(TRAIN_SPLIT + "|" + sid)
        item = (-priority, sid, row["business_name"], row["business_address"], row["country"])
        if len(sample_heap) < MAX_TRAIN_QUERIES:
            heapq.heappush(sample_heap, item)
        elif item > sample_heap[0]:
            heapq.heapreplace(sample_heap, item)
train_queries = {sid: {"entity_id": sid, "business_name": name, "business_address": address, "country": country}
                 for _, sid, name, address, country in sample_heap}
calibration_ids = {sid for sid in train_queries if hash64("cal|" + sid) % 5 == 0}
fit_ids = set(train_queries) - calibration_ids
train_truth = {}
with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sid = row["source1_entity_id"]
        if sid in train_queries:
            train_truth[sid] = {make_target_id(value) for value in (row["matched_entity_ids"] or "").split(",") if value}

validation_truth = {}
with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sid = row["source1_entity_id"]
        if sid in validation_ids:
            validation_truth[sid] = {make_target_id(value) for value in (row["matched_entity_ids"] or "").split(",") if value}
if validation_truth.keys() != validation_ids or train_truth.keys() != train_queries.keys():
    raise ValueError("Training/validation labels do not align with selected Source 1 IDs")

target_rows = 5_034_616 + 5_285_603
with tempfile.TemporaryDirectory(prefix="er_phase4_", dir=split_dir) as temp_dir:
    temp_dir = Path(temp_dir)
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
    n = {key: 0 for key in arrays}
    for source in (2, 3):
        with (train_dir / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                target_id = make_target_id(row["entity_id"])
                country = row["country"].strip()
                name = row["business_name"]
                address = row["business_address"]
                core = canonical_name(name)
                arrays["name"][n["name"]] = (index_key(name, country), target_id)
                arrays["address"][n["address"]] = (index_key(address, country), target_id)
                arrays["core"][n["core"]] = (index_key(core, country), target_id)
                arrays["unordered"][n["unordered"]] = (index_key(" ".join(sorted(core.split())), country), target_id)
                arrays["canonical_address"][n["canonical_address"]] = (index_key(canonical_address(address), country), target_id)
                for token in top_tokens(name, STOPWORDS, 5):
                    arrays["name_token"][n["name_token"]] = (hash64(token), target_id)
                    n["name_token"] += 1
                for token in top_tokens(address, ADDRESS_STOPWORDS, 4):
                    arrays["address_token"][n["address_token"]] = (hash64(token), target_id)
                    n["address_token"] += 1
                for pair in token_pairs(top_tokens(name, STOPWORDS, 3)):
                    arrays["name_pair"][n["name_pair"]] = (hash64(pair), target_id)
                    n["name_pair"] += 1
                for pair in token_pairs(top_tokens(address, ADDRESS_STOPWORDS, 3)):
                    arrays["address_pair"][n["address_pair"]] = (hash64(pair), target_id)
                    n["address_pair"] += 1
                for gram in char_ngrams(name):
                    arrays["char"][n["char"]] = (hash64(gram), target_id)
                    n["char"] += 1
                for key in ("name", "address", "core", "unordered", "canonical_address"):
                    n[key] += 1
    if n["name"] != target_rows:
        raise ValueError(f"Target index count mismatch: {n['name']} != {target_rows}")
    for key in ("name_token", "address_token", "name_pair", "address_pair", "char"):
        arrays[key] = arrays[key][:n[key]]
    for key, array in arrays.items():
        print(f"Sorting {key} index ({len(array):,} postings)...", flush=True)
        array.sort(order="key", kind="quicksort")
        array.flush()

    indexes = arrays
    training_features, training_labels = [], []
    calibration_records = []
    for q_index, (sid, row) in enumerate(sorted(train_queries.items())):
        routes, candidate_ids = build_route_candidates(row, indexes)
        ids, X = feature_matrix(routes, candidate_ids)
        truth_ids = train_truth[sid]
        labels = np.fromiter((int(int(target_id) in truth_ids) for target_id in ids), dtype=np.uint8, count=len(ids))
        if sid in fit_ids:
            positive_positions = np.flatnonzero(labels)
            negative_positions = np.flatnonzero(labels == 0)
            negatives = sorted(negative_positions, key=lambda pos: hash64(f"neg|{sid}|{int(ids[pos])}"))[:NEGATIVES_PER_QUERY]
            chosen = np.concatenate([positive_positions, np.asarray(negatives, dtype=np.int64)])
            if len(chosen):
                training_features.append(X[chosen])
                training_labels.append(labels[chosen])
        else:
            calibration_records.append((sid, X, labels, len(truth_ids)))
        if (q_index + 1) % 5_000 == 0:
            print(f"Prepared training queries: {q_index + 1:,}/{len(train_queries):,}", flush=True)
    X_train = np.concatenate(training_features, axis=0)
    y_train = np.concatenate(training_labels, axis=0)
    training_pair_count = int(len(y_train))
    training_positive_count = int(y_train.sum())
    training_negative_count = int((y_train == 0).sum())
    if not np.any(y_train) or np.all(y_train):
        raise ValueError("Training sample does not contain both pair classes")
    print(f"Training pair rows: {len(y_train):,}; positives: {int(y_train.sum()):,}; negatives: {int((y_train == 0).sum()):,}", flush=True)
    weights = fit_logistic(X_train, y_train)
    del X_train, y_train, training_features, training_labels

    calibration_scored = []
    all_calibration_scores = []
    for sid, X, labels, true_count in calibration_records:
        scores = predict_scores(X, weights)
        calibration_scored.append((scores, labels, true_count))
        if len(scores):
            all_calibration_scores.append(scores)
    if not all_calibration_scores:
        raise ValueError("Calibration sample produced no candidates")
    score_pool = np.concatenate(all_calibration_scores)
    thresholds = np.unique(np.concatenate([np.quantile(score_pool, np.linspace(0, 1, 151)), [np.nextafter(score_pool.max(), 1.0)]]))
    calibration_f05 = macro_scores_at_threshold(calibration_scored, thresholds)
    threshold = float(thresholds[int(np.argmax(calibration_f05))])
    calibration_best = float(np.max(calibration_f05))
    print(f"Calibrated threshold: {threshold:.8f}; calibration macro F0.5: {calibration_best:.6f}", flush=True)

    validation_source1_rows = 0
    total_candidate_pairs = total_true = total_hits = positive_entities = covered_entities = 0
    entity_f05_sum = 0.0
    source_f05_sum = {"S2": 0.0, "S3": 0.0}
    source_tp = {"S2": 0, "S3": 0}
    source_pred = {"S2": 0, "S3": 0}
    source_true = {"S2": 0, "S3": 0}
    source_hits = {"S2": 0, "S3": 0}
    source_positive_entities = {"S2": 0, "S3": 0}
    source_covered_entities = {"S2": 0, "S3": 0}
    no_match_sum = 0.0
    with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["entity_id"]
            if sid not in validation_ids:
                continue
            validation_source1_rows += 1
            routes, candidates = build_route_candidates(row, indexes)
            ids, X = feature_matrix(routes, candidates)
            scores = predict_scores(X, weights)
            predictions = {int(ids[i]) for i in np.flatnonzero(scores >= threshold)}
            truth_ids = validation_truth[sid]
            entity_f05_sum += entity_f05(truth_ids, predictions)
            no_match_sum += entity_f05(truth_ids, ())
            total_candidate_pairs += len(candidates)
            total_true += len(truth_ids)
            total_hits += len(truth_ids & candidates)
            positive_entities += bool(truth_ids)
            covered_entities += bool(truth_ids) and bool(truth_ids & candidates)
            for label, code in (("S2", 2), ("S3", 3)):
                truth_source = {target for target in truth_ids if target >> 32 == code}
                pred_source = {target for target in predictions if target >> 32 == code}
                candidate_source = {target for target in candidates if target >> 32 == code}
                hit_source = truth_source & pred_source
                source_f05_sum[label] += entity_f05(truth_source, pred_source)
                source_tp[label] += len(hit_source)
                source_pred[label] += len(pred_source)
                source_true[label] += len(truth_source)
                source_hits[label] += len(truth_source & candidate_source)
                source_positive_entities[label] += bool(truth_source)
                source_covered_entities[label] += bool(truth_source) and bool(truth_source & candidate_source)
            if validation_source1_rows % 100_000 == 0:
                print(f"Scored validation queries: {validation_source1_rows:,}/{len(validation_ids):,}", flush=True)
    if validation_source1_rows != len(validation_ids):
        raise ValueError("Validation query count mismatch")

    max_pairs = len(validation_ids) * target_rows
    predictions_total = sum(source_pred.values())
    tp_total = sum(source_tp.values())
    fn_total = total_true - tp_total
    fp_total = predictions_total - tp_total
    results = {
        "validation_source1_rows": validation_source1_rows,
        "training_queries_sampled": len(train_queries),
        "fit_queries": len(fit_ids),
        "calibration_queries": len(calibration_ids),
        "training_pairs": {"rows": training_pair_count, "positive": training_positive_count, "negative": training_negative_count},
        "feature_names": FEATURE_NAMES,
        "threshold_selection": "macro F0.5 on deterministic training-only calibration queries",
        "probability_threshold": threshold,
        "calibration_macro_f05": calibration_best,
        "validation": {
            "candidate_pairs": total_candidate_pairs,
            "average_candidates_per_source1": total_candidate_pairs / validation_source1_rows,
            "candidate_recall": total_hits / total_true,
            "candidate_positive_entity_coverage": covered_entities / positive_entities,
            "candidate_reduction_ratio": 1.0 - total_candidate_pairs / max_pairs,
            "macro_f05": entity_f05_sum / validation_source1_rows,
            "no_match_macro_f05": no_match_sum / validation_source1_rows,
            "micro_precision": tp_total / predictions_total if predictions_total else 0.0,
            "micro_recall": tp_total / total_true if total_true else 0.0,
            "micro_f05": 1.25 * tp_total / (1.25 * tp_total + fp_total + 0.25 * fn_total) if tp_total or fp_total or fn_total else 0.0,
            "by_target_source": {
                label: {
                    "macro_f05": source_f05_sum[label] / validation_source1_rows,
                    "candidate_recall": source_hits[label] / source_true[label] if source_true[label] else 0.0,
                    "candidate_positive_entity_coverage": source_covered_entities[label] / source_positive_entities[label] if source_positive_entities[label] else 0.0,
                    "micro_precision": source_tp[label] / source_pred[label] if source_pred[label] else 0.0,
                    "micro_recall": source_tp[label] / source_true[label] if source_true[label] else 0.0,
                } for label in ("S2", "S3")
            }
        },
        "model": {"type": "L2-regularized logistic regression", "optimizer": "batch gradient descent", "iterations": 80,
                  "negative_sampling": f"up to {NEGATIVES_PER_QUERY} deterministic hard negatives per fit query; all generated positives retained",
                  "weight_intercept_and_features": [float(value) for value in weights]},
    }
    output = split_dir / "phase4_matcher_metrics.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print("Metrics saved:", output, flush=True)
    for array in arrays.values():
        array.flush()
        if isinstance(array, np.memmap):
            array._mmap.close()
