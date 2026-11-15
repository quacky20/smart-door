import argparse
import csv
import json
import os
import re
import statistics
import time
from collections import Counter

import cv2
import dlib

from embedding import get_face_embedding
from face_recognition import recognize_face
from face_utils import extract_face, largest_face
from variables import MATCH_THRESHOLD_L2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FILENAME_PATTERN = re.compile(r"^(.+)_([0-9]+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition against name_number image labels."
    )
    parser.add_argument("--test-dir", default="test_faces")
    parser.add_argument("--results", default="evaluation_results.json")
    parser.add_argument("--confusion-matrix", default="confusion_matrix.csv")
    return parser.parse_args()


def metric_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def r2_score_integer_labels(actual, predicted, labels):
    """Return R² over deterministic integer label IDs.

    R² is a regression metric and is not meaningful for nominal identities.
    This value is included only because it was requested, with that caveat in
    the generated report.
    """
    label_ids = {label: index for index, label in enumerate(labels)}
    actual_ids = [label_ids[label] for label in actual]
    predicted_ids = [label_ids[label] for label in predicted]
    mean_actual = statistics.mean(actual_ids)
    total = sum((value - mean_actual) ** 2 for value in actual_ids)
    residual = sum(
        (actual_value - predicted_value) ** 2
        for actual_value, predicted_value in zip(actual_ids, predicted_ids)
    )
    return 1.0 - residual / total if total else 0.0


def calculate_metrics(actual, predicted, distances, detected_count, total_time):
    labels = sorted(set(actual) | set(predicted))
    confusion = {actual_label: Counter() for actual_label in labels}
    for actual_label, predicted_label in zip(actual, predicted):
        confusion[actual_label][predicted_label] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1_scores = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            row[label] for row_label, row in confusion.items() if row_label != label
        )
        false_negative = sum(value for name, value in confusion[label].items() if name != label)
        support = sum(confusion[label].values())
        precision = metric_divide(true_positive, true_positive + false_positive)
        recall = metric_divide(true_positive, true_positive + false_negative)
        f1 = metric_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if label != "Unknown":
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    correct = sum(actual_label == predicted_label for actual_label, predicted_label in zip(actual, predicted))
    known_actual = [label for label in actual if label != "Unknown"]
    known_labels = set(known_actual)
    known_correct = sum(
        actual_label == predicted_label
        for actual_label, predicted_label in zip(actual, predicted)
        if actual_label != "Unknown"
    )
    macro_precision = metric_divide(sum(precisions), len(precisions))
    macro_recall = metric_divide(sum(recalls), len(recalls))
    macro_f1 = metric_divide(sum(f1_scores), len(f1_scores))

    return {
        "samples": len(actual),
        "correct": correct,
        "accuracy": metric_divide(correct, len(actual)),
        "precision_macro_known": macro_precision,
        "recall_macro_known": macro_recall,
        "f1_macro_known": macro_f1,
        "precision_weighted_known": metric_divide(
            sum(
                per_class[label]["precision"] * per_class[label]["support"]
                for label in known_labels
            ),
            len(known_actual),
        ),
        "recall_weighted_known": metric_divide(
            sum(
                per_class[label]["recall"] * per_class[label]["support"]
                for label in known_labels
            ),
            len(known_actual),
        ),
        "f1_weighted_known": metric_divide(
            sum(
                per_class[label]["f1"] * per_class[label]["support"]
                for label in known_labels
            ),
            len(known_actual),
        ),
        "balanced_accuracy_known": metric_divide(
            sum(per_class[label]["recall"] for label in known_labels),
            len(known_labels),
        ),
        "detection_rate": metric_divide(detected_count, len(actual)),
        "no_face_count": len(actual) - detected_count,
        "known_identity_accuracy": metric_divide(known_correct, len(known_actual)),
        "unknown_prediction_count": sum(label == "Unknown" for label in predicted),
        "mean_distance_detected": statistics.mean(distances) if distances else None,
        "median_distance_detected": statistics.median(distances) if distances else None,
        "r2_integer_label_encoding": r2_score_integer_labels(actual, predicted, labels),
        "r2_note": (
            "R² is not a valid nominal-classification metric; this is included "
            "only as requested using deterministic integer identity labels."
        ),
        "per_class": per_class,
        "labels": labels,
        "elapsed_seconds": total_time,
        "images_per_second": metric_divide(len(actual), total_time),
        "threshold_l2": MATCH_THRESHOLD_L2,
    }, confusion


def evaluate(test_dir):
    detector = dlib.get_frontal_face_detector()
    actual = []
    predicted = []
    distances = []
    confusion_inputs = []
    detected_count = 0
    skipped_count = 0
    start = time.perf_counter()

    paths = sorted(
        (
            os.path.join(root, filename)
            for root, _, filenames in os.walk(test_dir)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ),
        key=str.lower,
    )
    for image_path in paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        match = FILENAME_PATTERN.match(stem)
        if not match:
            skipped_count += 1
            continue

        frame = cv2.imread(image_path)
        actual_label = match.group(1)
        predicted_label = "Unknown"
        distance = None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)
            if len(faces) > 0:
                face = largest_face(faces)
                face_image = extract_face(frame, face)
                if face_image is not None and face_image.size > 0:
                    embedding = get_face_embedding(face_image)
                    predicted_label, distance = recognize_face(embedding)
                    detected_count += 1
                    distances.append(distance)

        actual.append(actual_label)
        predicted.append(predicted_label)
    elapsed = time.perf_counter() - start
    metrics, confusion = calculate_metrics(
        actual, predicted, distances, detected_count, elapsed
    )
    metrics["skipped_unlabeled_files"] = skipped_count
    return metrics, confusion


def write_confusion_matrix(path, confusion, labels):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["actual\\predicted"] + labels)
        for actual_label in labels:
            writer.writerow(
                [actual_label] + [confusion[actual_label][label] for label in labels]
            )


def main():
    args = parse_args()
    metrics, confusion = evaluate(args.test_dir)
    with open(args.results, "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    write_confusion_matrix(args.confusion_matrix, confusion, metrics["labels"])

    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2))
    print(f"\nDetailed results: {args.results}")
    print(f"Confusion matrix: {args.confusion_matrix}")


if __name__ == "__main__":
    main()
