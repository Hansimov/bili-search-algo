import argparse
import json
import math
import random
import re
import sys

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from sedb import MongoOperator
from tclogger import logger, dict_to_str, logstr, str_to_ts

from configs.envs import DATA_ROOT, MONGO_ENVS
from data_utils.videos.filter import REGION_MONGO_FILTERS

OWNER_DOMAIN_ROOT = DATA_ROOT / "owners"
OWNER_DOMAIN_SAMPLES_PATH = OWNER_DOMAIN_ROOT / "owner_domain_samples.jsonl"
OWNER_DOMAIN_METRICS_PATH = OWNER_DOMAIN_ROOT / "owner_domain_metrics.json"
OWNER_DOMAIN_PREDICTIONS_PATH = OWNER_DOMAIN_ROOT / "owner_domain_predictions.jsonl"

VIDEO_PROJECTION = {
    "_id": 0,
    "bvid": 1,
    "title": 1,
    "desc": 1,
    "tags": 1,
    "pubdate": 1,
    "tid": 1,
    "ptid": 1,
    "stat.view": 1,
    "owner.mid": 1,
    "owner.name": 1,
}

DEFAULT_LABEL_GROUPS = [
    group for group in REGION_MONGO_FILTERS.keys() if group not in {"recent", "test"}
]
LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]{1,}")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")


def dict_get(data: dict, key: str, default=None):
    value = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def match_filter_spec(doc: dict, spec: dict) -> bool:
    for key, value in spec.items():
        if key == "$or":
            if not any(match_filter_spec(doc, item) for item in value):
                return False
            continue

        doc_value = dict_get(doc, key)
        if isinstance(value, dict):
            for op, op_value in value.items():
                if op == "$in":
                    if doc_value not in op_value:
                        return False
                elif op == "$nin":
                    if doc_value in op_value:
                        return False
                elif op == "$ne":
                    if doc_value == op_value:
                        return False
                elif op == "$gte":
                    if doc_value is None or doc_value < op_value:
                        return False
                elif op == "$lte":
                    if doc_value is None or doc_value > op_value:
                        return False
                else:
                    raise ValueError(f"Unsupported filter op: {op}")
        elif doc_value != value:
            return False

    return True


def select_owner_label(
    videos: list[dict],
    label_groups: list[str] = None,
    dominant_ratio: float = 0.6,
    min_label_videos: int = 3,
) -> Optional[dict]:
    label_groups = label_groups or DEFAULT_LABEL_GROUPS
    counts = Counter()
    for video in videos:
        for group in label_groups:
            if match_filter_spec(video, REGION_MONGO_FILTERS[group]):
                counts[group] += 1

    if not counts:
        return None

    label, label_count = counts.most_common(1)[0]
    total_labeled = sum(counts.values())
    label_ratio = label_count / max(total_labeled, 1)
    if label_count < min_label_videos or label_ratio < dominant_ratio:
        return None

    return {
        "label": label,
        "label_count": label_count,
        "label_ratio": round(label_ratio, 4),
        "group_counts": dict(counts.most_common()),
    }


def tokenize_text(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = LATIN_TOKEN_RE.findall(lowered)

    for span in CJK_SPAN_RE.findall(lowered):
        span = span.strip()
        if not span:
            continue
        if len(span) <= 8:
            tokens.append(span)
        if len(span) == 1:
            tokens.append(span)
            continue
        for idx in range(len(span) - 1):
            tokens.append(span[idx : idx + 2])

    return tokens


class OwnerDomainSampleBuilder:
    def __init__(
        self,
        mongo_collection: str = "videos",
        min_videos: int = 8,
        dominant_ratio: float = 0.6,
        min_label_videos: int = 3,
        sample_per_owner: int = 12,
        max_owners: int = None,
        start_pubdate: str = None,
        end_pubdate: str = None,
        max_scanned_videos: int = None,
        allow_full_scan: bool = False,
        label_groups: list[str] = None,
    ):
        self.mongo_collection = mongo_collection
        self.min_videos = min_videos
        self.dominant_ratio = dominant_ratio
        self.min_label_videos = min_label_videos
        self.sample_per_owner = sample_per_owner
        self.max_owners = max_owners
        self.start_pubdate = start_pubdate
        self.end_pubdate = end_pubdate
        self.max_scanned_videos = max_scanned_videos
        self.allow_full_scan = allow_full_scan
        self.label_groups = label_groups or DEFAULT_LABEL_GROUPS
        self.init_mongo()

    def validate_budget(self):
        if self.allow_full_scan:
            return
        if self.max_owners or self.max_scanned_videos:
            return
        if self.start_pubdate or self.end_pubdate:
            return
        raise ValueError(
            "Refusing full owner scan without sampling budget. Set --max-owners, "
            "--max-scanned-videos, a pubdate window, or pass --allow-full-scan explicitly."
        )

    def init_mongo(self):
        self.mongo = MongoOperator(
            configs=MONGO_ENVS, connect_cls=self.__class__, verbose_args=False
        )
        self.db = self.mongo.client[MONGO_ENVS.get("dbname", "bili")]
        self.videos_col = self.db[self.mongo_collection]

    def build_query(self) -> dict:
        query = {"owner.mid": {"$exists": True}, "owner.name": {"$exists": True}}
        pubdate_filter = {}
        if self.start_pubdate:
            pubdate_filter["$gte"] = str_to_ts(self.start_pubdate)
        if self.end_pubdate:
            pubdate_filter["$lte"] = str_to_ts(self.end_pubdate)
        if pubdate_filter:
            query["pubdate"] = pubdate_filter
        return query

    def get_cursor(self):
        self.validate_budget()
        query = self.build_query()
        logger.note("> Owner domain sample query:")
        logger.mesg(dict_to_str(query), indent=2)
        return (
            self.videos_col.find(query, VIDEO_PROJECTION)
            .sort("owner.mid", 1)
            .batch_size(5000)
        )

    def _extract_top_tags(self, videos: list[dict], top_k: int = 8) -> list[str]:
        counter = Counter()
        for video in videos:
            tags = (video.get("tags") or "").split(",")
            for tag in tags:
                tag = tag.strip()
                if tag:
                    counter[tag] += 1
        return [tag for tag, _ in counter.most_common(top_k)]

    def _select_titles(self, videos: list[dict], top_k: int = 8) -> list[str]:
        ranked = sorted(
            videos,
            key=lambda item: (
                int(dict_get(item, "stat.view", 0) or 0),
                int(item.get("pubdate") or 0),
            ),
            reverse=True,
        )
        titles = []
        seen = set()
        for video in ranked:
            title = (video.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            titles.append(title)
            if len(titles) >= top_k:
                break
        return titles

    def build_owner_sample(self, videos: list[dict]) -> Optional[dict]:
        if not videos or len(videos) < self.min_videos:
            return None

        label_info = select_owner_label(
            videos,
            label_groups=self.label_groups,
            dominant_ratio=self.dominant_ratio,
            min_label_videos=self.min_label_videos,
        )
        if not label_info:
            return None

        owner = videos[0].get("owner") or {}
        owner_name = owner.get("name") or ""
        mid = owner.get("mid")
        top_tags = self._extract_top_tags(videos)
        titles = self._select_titles(videos, top_k=self.sample_per_owner)
        desc_samples = []
        for video in videos:
            desc = (video.get("desc") or "").strip()
            if desc and desc != "-":
                desc_samples.append(desc[:80])
            if len(desc_samples) >= 4:
                break

        text_parts = [owner_name]
        if top_tags:
            text_parts.append(" ".join(top_tags))
        if titles:
            text_parts.append(" ".join(titles))
        if desc_samples:
            text_parts.append(" ".join(desc_samples))

        return {
            "mid": mid,
            "owner_name": owner_name,
            "label": label_info["label"],
            "label_ratio": label_info["label_ratio"],
            "label_count": label_info["label_count"],
            "group_counts": label_info["group_counts"],
            "total_videos": len(videos),
            "top_tags": top_tags,
            "sample_titles": titles,
            "text": " ".join(part for part in text_parts if part).strip(),
        }

    def build_samples(self) -> list[dict]:
        cursor = self.get_cursor()
        samples = []
        current_mid = None
        owner_videos = []
        scanned_videos = 0

        for video in cursor:
            scanned_videos += 1
            if self.max_scanned_videos and scanned_videos > self.max_scanned_videos:
                break
            mid = dict_get(video, "owner.mid")
            if mid is None:
                continue
            if current_mid is None:
                current_mid = mid
            if mid != current_mid:
                sample = self.build_owner_sample(owner_videos)
                if sample:
                    samples.append(sample)
                    if self.max_owners and len(samples) >= self.max_owners:
                        break
                current_mid = mid
                owner_videos = [video]
            else:
                owner_videos.append(video)

        if owner_videos and (not self.max_owners or len(samples) < self.max_owners):
            sample = self.build_owner_sample(owner_videos)
            if sample:
                samples.append(sample)

        logger.success(
            f"  ✓ Owner samples built: {len(samples):,} from {scanned_videos:,} videos"
        )
        logger.note("> Sampling budget:")
        logger.mesg(
            dict_to_str(
                {
                    "max_owners": self.max_owners,
                    "max_scanned_videos": self.max_scanned_videos,
                    "start_pubdate": self.start_pubdate,
                    "end_pubdate": self.end_pubdate,
                    "allow_full_scan": self.allow_full_scan,
                }
            ),
            indent=2,
        )
        return samples

    def dump_samples(self, samples: list[dict], path: Path = OWNER_DOMAIN_SAMPLES_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as wf:
            for sample in samples:
                wf.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.file(f"  * samples: {path}")


class OwnerDomainCentroidClassifier:
    def __init__(self, min_token_freq: int = 2):
        self.min_token_freq = min_token_freq
        self.idf = {}
        self.centroids = {}
        self.label_doc_counts = Counter()

    def vectorize(self, text: str) -> dict[str, float]:
        token_counts = Counter(tokenize_text(text))
        return {
            token: float(count)
            for token, count in token_counts.items()
            if token in self.idf
        }

    def fit(self, samples: list[dict]):
        doc_freq = Counter()
        label_token_totals = defaultdict(Counter)
        total_docs = len(samples)

        for sample in samples:
            label = sample["label"]
            tokens = tokenize_text(sample.get("text", ""))
            if not tokens:
                continue
            self.label_doc_counts[label] += 1
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
            label_token_totals[label].update(tokens)

        self.idf = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
            if freq >= self.min_token_freq
        }

        self.centroids = {}
        for label, token_totals in label_token_totals.items():
            weights = {}
            norm = 0.0
            for token, tf in token_totals.items():
                if token not in self.idf:
                    continue
                weight = tf * self.idf[token]
                weights[token] = weight
                norm += weight * weight
            norm = math.sqrt(norm) or 1.0
            self.centroids[label] = {
                token: weight / norm for token, weight in weights.items()
            }

    def predict(self, text: str) -> tuple[Optional[str], dict[str, float]]:
        vector = self.vectorize(text)
        if not vector or not self.centroids:
            return None, {}

        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vector = {token: value / norm for token, value in vector.items()}
        scores = {}
        for label, centroid in self.centroids.items():
            score = 0.0
            for token, value in vector.items():
                score += value * centroid.get(token, 0.0)
            scores[label] = round(score, 6)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return (ranked[0][0] if ranked else None), dict(ranked)

    def evaluate(
        self, samples: list[dict], test_ratio: float = 0.2, seed: int = 42
    ) -> dict:
        return evaluate_classifier(self, samples, test_ratio=test_ratio, seed=seed)


class OwnerDomainNaiveBayesClassifier:
    def __init__(self, min_token_freq: int = 2, alpha: float = 1.0):
        self.min_token_freq = min_token_freq
        self.alpha = alpha
        self.vocab = set()
        self.label_doc_counts = Counter()
        self.label_token_counts = defaultdict(Counter)
        self.label_total_tokens = Counter()
        self.labels = []

    def fit(self, samples: list[dict]):
        global_freq = Counter()
        label_doc_counts = Counter()
        label_token_counts = defaultdict(Counter)

        for sample in samples:
            label = sample["label"]
            tokens = tokenize_text(sample.get("text", ""))
            if not tokens:
                continue
            label_doc_counts[label] += 1
            label_token_counts[label].update(tokens)
            global_freq.update(tokens)

        self.vocab = {
            token for token, freq in global_freq.items() if freq >= self.min_token_freq
        }
        self.label_doc_counts = label_doc_counts
        self.labels = sorted(label_doc_counts.keys())
        self.label_token_counts = defaultdict(Counter)
        self.label_total_tokens = Counter()

        for label, token_counts in label_token_counts.items():
            filtered = Counter(
                {
                    token: count
                    for token, count in token_counts.items()
                    if token in self.vocab
                }
            )
            self.label_token_counts[label] = filtered
            self.label_total_tokens[label] = sum(filtered.values())

    def predict(self, text: str) -> tuple[Optional[str], dict[str, float]]:
        if not self.labels:
            return None, {}

        token_counts = Counter(
            token for token in tokenize_text(text) if token in self.vocab
        )
        if not token_counts:
            return None, {}

        total_docs = sum(self.label_doc_counts.values())
        vocab_size = max(len(self.vocab), 1)
        log_scores = {}
        for label in self.labels:
            prior = math.log(self.label_doc_counts[label] / total_docs)
            denom = self.label_total_tokens[label] + self.alpha * vocab_size
            score = prior
            for token, count in token_counts.items():
                token_prob = (
                    self.label_token_counts[label].get(token, 0) + self.alpha
                ) / denom
                score += count * math.log(token_prob)
            log_scores[label] = score

        max_score = max(log_scores.values())
        exp_scores = {
            label: math.exp(score - max_score) for label, score in log_scores.items()
        }
        total_score = sum(exp_scores.values()) or 1.0
        probs = {
            label: round(score / total_score, 6)
            for label, score in sorted(
                exp_scores.items(), key=lambda item: item[1], reverse=True
            )
        }
        best_label = next(iter(probs.keys()), None)
        return best_label, probs

    def evaluate(
        self, samples: list[dict], test_ratio: float = 0.2, seed: int = 42
    ) -> dict:
        return evaluate_classifier(self, samples, test_ratio=test_ratio, seed=seed)


class OwnerDomainLinearClassifier:
    def __init__(self, min_token_freq: int = 2, epochs: int = 8):
        self.min_token_freq = min_token_freq
        self.epochs = epochs
        self.vocab = set()
        self.labels = []
        self.weights = defaultdict(dict)
        self.bias = defaultdict(float)

    def _vectorize(self, text: str) -> dict[str, float]:
        return {
            token: float(count)
            for token, count in Counter(tokenize_text(text)).items()
            if token in self.vocab
        }

    def _score_label(self, features: dict[str, float], label: str) -> float:
        score = self.bias[label]
        label_weights = self.weights[label]
        for token, value in features.items():
            score += label_weights.get(token, 0.0) * value
        return score

    def fit(self, samples: list[dict]):
        token_freq = Counter()
        labels = set()
        prepared = []
        for sample in samples:
            label = sample["label"]
            tokens = tokenize_text(sample.get("text", ""))
            if not tokens:
                continue
            labels.add(label)
            token_freq.update(tokens)
            prepared.append((label, Counter(tokens)))

        self.vocab = {
            token for token, freq in token_freq.items() if freq >= self.min_token_freq
        }
        self.labels = sorted(labels)
        prepared = [
            (
                label,
                {
                    token: float(count)
                    for token, count in counts.items()
                    if token in self.vocab
                },
            )
            for label, counts in prepared
        ]

        for _ in range(self.epochs):
            for label, features in prepared:
                if not features:
                    continue
                pred_label, _ = self.predict_from_features(features)
                if pred_label == label:
                    continue
                for token, value in features.items():
                    self.weights[label][token] = (
                        self.weights[label].get(token, 0.0) + value
                    )
                    self.weights[pred_label][token] = (
                        self.weights[pred_label].get(token, 0.0) - value
                    )
                self.bias[label] += 1.0
                self.bias[pred_label] -= 1.0

    def predict_from_features(
        self, features: dict[str, float]
    ) -> tuple[Optional[str], dict[str, float]]:
        if not self.labels:
            return None, {}
        scores = {
            label: round(self._score_label(features, label), 6) for label in self.labels
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return (ranked[0][0] if ranked else None), dict(ranked)

    def predict(self, text: str) -> tuple[Optional[str], dict[str, float]]:
        features = self._vectorize(text)
        if not features:
            return None, {}
        return self.predict_from_features(features)

    def evaluate(
        self, samples: list[dict], test_ratio: float = 0.2, seed: int = 42
    ) -> dict:
        return evaluate_classifier(self, samples, test_ratio=test_ratio, seed=seed)


def split_samples(
    samples: list[dict], test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    samples_by_label = defaultdict(list)
    for sample in samples:
        samples_by_label[sample["label"]].append(sample)

    train_samples = []
    test_samples = []
    rng = random.Random(seed)
    for label, label_samples in samples_by_label.items():
        if len(label_samples) < 2:
            continue
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_ratio))
        test_count = min(test_count, len(shuffled) - 1)
        test_samples.extend(shuffled[:test_count])
        train_samples.extend(shuffled[test_count:])

    if not train_samples or not test_samples:
        raise ValueError("Not enough labeled owner samples for evaluation")

    return train_samples, test_samples, samples_by_label


def evaluate_classifier(
    classifier,
    samples: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    train_samples, test_samples, samples_by_label = split_samples(
        samples, test_ratio=test_ratio, seed=seed
    )
    classifier.fit(train_samples)

    correct = 0
    predictions = []
    confusion = defaultdict(Counter)
    for sample in test_samples:
        pred_label, scores = classifier.predict(sample.get("text", ""))
        truth = sample["label"]
        if pred_label == truth:
            correct += 1
        confusion[truth][pred_label or "<none>"] += 1
        predictions.append(
            {
                "mid": sample["mid"],
                "owner_name": sample["owner_name"],
                "truth": truth,
                "pred": pred_label,
                "scores": scores,
            }
        )

    accuracy = correct / max(len(test_samples), 1)
    metrics = {
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "label_count": len({sample["label"] for sample in train_samples}),
        "accuracy": round(accuracy, 4),
        "labels": {
            label: len(label_samples)
            for label, label_samples in samples_by_label.items()
        },
        "confusion": {label: dict(counter) for label, counter in confusion.items()},
        "model": classifier.__class__.__name__,
    }
    return {"metrics": metrics, "predictions": predictions}


def evaluate_multiple_models(
    model_name: str,
    samples: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    if model_name == "centroid":
        classifier = OwnerDomainCentroidClassifier()
        return evaluate_classifier(
            classifier, samples, test_ratio=test_ratio, seed=seed
        )
    if model_name == "naive_bayes":
        classifier = OwnerDomainNaiveBayesClassifier()
        return evaluate_classifier(
            classifier, samples, test_ratio=test_ratio, seed=seed
        )
    if model_name == "linear":
        classifier = OwnerDomainLinearClassifier()
        return evaluate_classifier(
            classifier, samples, test_ratio=test_ratio, seed=seed
        )
    if model_name == "compare":
        train_samples, test_samples, samples_by_label = split_samples(
            samples, test_ratio=test_ratio, seed=seed
        )
        results = {}
        for name, classifier in {
            "centroid": OwnerDomainCentroidClassifier(),
            "naive_bayes": OwnerDomainNaiveBayesClassifier(),
            "linear": OwnerDomainLinearClassifier(),
        }.items():
            classifier.fit(train_samples)
            correct = 0
            confusion = defaultdict(Counter)
            predictions = []
            for sample in test_samples:
                pred_label, scores = classifier.predict(sample.get("text", ""))
                truth = sample["label"]
                if pred_label == truth:
                    correct += 1
                confusion[truth][pred_label or "<none>"] += 1
                predictions.append(
                    {
                        "mid": sample["mid"],
                        "owner_name": sample["owner_name"],
                        "truth": truth,
                        "pred": pred_label,
                        "scores": scores,
                    }
                )
            metrics = {
                "train_size": len(train_samples),
                "test_size": len(test_samples),
                "label_count": len({sample["label"] for sample in train_samples}),
                "accuracy": round(correct / max(len(test_samples), 1), 4),
                "labels": {
                    label: len(label_samples)
                    for label, label_samples in samples_by_label.items()
                },
                "confusion": {
                    label: dict(counter) for label, counter in confusion.items()
                },
                "model": classifier.__class__.__name__,
            }
            results[name] = {
                "metrics": metrics,
                "predictions": predictions,
            }
        return results
    raise ValueError(f"Unsupported model: {model_name}")


def dump_metrics(
    metrics: dict,
    predictions,
    metrics_path: Path = OWNER_DOMAIN_METRICS_PATH,
    predictions_path: Path = OWNER_DOMAIN_PREDICTIONS_PATH,
):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as wf:
        json.dump(metrics, wf, ensure_ascii=False, indent=2)
    with open(predictions_path, "w", encoding="utf-8") as wf:
        if isinstance(predictions, list):
            for prediction in predictions:
                wf.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        else:
            json.dump(predictions, wf, ensure_ascii=False, indent=2)
    logger.file(f"  * metrics: {metrics_path}")
    logger.file(f"  * predictions: {predictions_path}")


class OwnerDomainArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-c", "--mongo-collection", type=str, default="videos")
        self.add_argument("-m", "--max-owners", type=int, default=None)
        self.add_argument("--min-videos", type=int, default=8)
        self.add_argument("--dominant-ratio", type=float, default=0.6)
        self.add_argument("--min-label-videos", type=int, default=3)
        self.add_argument("--sample-per-owner", type=int, default=12)
        self.add_argument("-s", "--start-pubdate", type=str, default=None)
        self.add_argument("-e", "--end-pubdate", type=str, default=None)
        self.add_argument("--max-scanned-videos", type=int, default=None)
        self.add_argument("--allow-full-scan", action="store_true")
        self.add_argument(
            "--model",
            choices=["centroid", "naive_bayes", "linear", "compare"],
            default="centroid",
        )
        self.add_argument("--test-ratio", type=float, default=0.2)
        self.add_argument("--seed", type=int, default=42)
        self.add_argument("--build-only", action="store_true")
        self.add_argument("--eval-only", action="store_true")
        self.add_argument(
            "--samples-path", type=Path, default=OWNER_DOMAIN_SAMPLES_PATH
        )
        self.add_argument(
            "--metrics-path", type=Path, default=OWNER_DOMAIN_METRICS_PATH
        )
        self.add_argument(
            "--predictions-path",
            type=Path,
            default=OWNER_DOMAIN_PREDICTIONS_PATH,
        )


def load_samples(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as rf:
        return [json.loads(line) for line in rf if line.strip()]


def main(args: argparse.Namespace):
    logger.note("> Owner domain experiment:")
    logger.mesg(dict_to_str(vars(args)), indent=2)

    samples = None
    if not args.eval_only:
        builder = OwnerDomainSampleBuilder(
            mongo_collection=args.mongo_collection,
            min_videos=args.min_videos,
            dominant_ratio=args.dominant_ratio,
            min_label_videos=args.min_label_videos,
            sample_per_owner=args.sample_per_owner,
            max_owners=args.max_owners,
            start_pubdate=args.start_pubdate,
            end_pubdate=args.end_pubdate,
            max_scanned_videos=args.max_scanned_videos,
            allow_full_scan=args.allow_full_scan,
        )
        samples = builder.build_samples()
        builder.dump_samples(samples, path=args.samples_path)

    if args.build_only:
        return

    if samples is None:
        samples = load_samples(args.samples_path)

    eval_result = evaluate_multiple_models(
        args.model,
        samples,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.model == "compare":
        metrics = {
            model_name: result["metrics"] for model_name, result in eval_result.items()
        }
        predictions = {
            model_name: result["predictions"]
            for model_name, result in eval_result.items()
        }
        metrics_path = args.metrics_path
        predictions_path = args.predictions_path
        dump_metrics(
            metrics,
            predictions,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )
        logger.success(
            "  ✓ Compare accuracy: "
            + ", ".join(
                f"{name}={logstr.success(result['metrics']['accuracy'])}"
                for name, result in eval_result.items()
            )
        )
    else:
        dump_metrics(
            eval_result["metrics"],
            eval_result["predictions"],
            metrics_path=args.metrics_path,
            predictions_path=args.predictions_path,
        )
        logger.success(
            f"  ✓ Accuracy: {logstr.success(eval_result['metrics']['accuracy'])} "
            f"on {eval_result['metrics']['test_size']} owners"
        )


if __name__ == "__main__":
    parsed_args = OwnerDomainArgParser().parse_args(sys.argv[1:])
    main(parsed_args)
