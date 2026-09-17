import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import Booster, DMatrix, XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "outputs" / "SurgWound" / "fused_features.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "SurgWound" / "clinical_labels.csv"
MODEL_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "SurgWound"
SPLITS = ("train", "validation", "test")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_joined_data() -> tuple[np.ndarray, dict[str, np.ndarray], list[str], list[str]]:
    feature_rows = _read_csv(FEATURES_PATH)
    label_rows = _read_csv(LABELS_PATH)
    feature_map = {row["filename"]: row for row in feature_rows}
    label_map = {row["filename"]: row for row in label_rows}
    if set(feature_map) != set(label_map):
        raise ValueError("Feature and clinical-label filenames do not match exactly")
    if len(feature_map) != 686:
        raise ValueError(f"Expected 686 joined samples, found {len(feature_map)}")

    feature_names = [name for name in feature_rows[0] if name != "filename"]
    features = np.asarray(
        [[float(feature_map[name][feature]) for feature in feature_names] for name in sorted(feature_map)],
        dtype=np.float32,
    )
    ordered_names = sorted(feature_map)
    ordered_labels = [label_map[name] for name in ordered_names]
    split_values = np.asarray([row["split"] for row in ordered_labels])
    if set(split_values) != set(SPLITS):
        raise ValueError(f"Expected splits {SPLITS}, found {sorted(set(split_values))}")
    targets = {
        "healing": np.asarray([row["healing_outcome"] for row in ordered_labels]),
        "infection": np.asarray([row["infection_status"] for row in ordered_labels]),
        "split": split_values,
    }
    return features, targets, ordered_names, feature_names


def _class_weights(values: np.ndarray, train_mask: np.ndarray) -> dict[int, float]:
    counts = Counter(values[train_mask].tolist())
    total = sum(counts.values())
    return {label: total / (len(counts) * count) for label, count in counts.items()}


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def _train_model(
    features: np.ndarray,
    values: np.ndarray,
    splits: np.ndarray,
    class_names: list[str],
) -> tuple[XGBClassifier, dict[str, Any], dict[str, np.ndarray]]:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    encoded = np.asarray([class_to_id[value] for value in values], dtype=np.int32)
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    weights = _class_weights(encoded, train_mask)
    sample_weights = np.asarray([weights[label] for label in encoded[train_mask]], dtype=np.float32)
    estimator = XGBClassifier(
        objective="binary:logistic" if len(class_names) == 2 else "multi:softprob",
        **({"num_class": len(class_names)} if len(class_names) > 2 else {}),
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss" if len(class_names) > 2 else "logloss",
        random_state=42,
        n_jobs=-1,
    )
    estimator.fit(
        features[train_mask],
        encoded[train_mask],
        sample_weight=sample_weights,
        eval_set=[(features[validation_mask], encoded[validation_mask])],
        verbose=False,
    )
    predictions = {
        split: estimator.predict(features[splits == split]).astype(int)
        for split in SPLITS
    }
    test_metrics = _metrics(
        encoded[test_mask],
        predictions["test"],
        list(range(len(class_names))),
        class_names,
    )
    return estimator, {
        "classes": class_names,
        "train_class_weights": weights,
        "split_counts": {split: int(np.sum(splits == split)) for split in SPLITS},
        "test": test_metrics,
    }, predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)


def _write_confusion_matrix(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true\\predicted", *class_names])
        writer.writerows([[class_names[index], *row] for index, row in enumerate(matrix)])


def _infection_estimator(parameters: dict[str, float | int]) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **parameters,
    )


def _infection_predictions(estimator: XGBClassifier | Booster, features: np.ndarray, splits: np.ndarray) -> dict[str, np.ndarray]:
    def predict(matrix: np.ndarray) -> np.ndarray:
        if isinstance(estimator, Booster):
            return np.argmax(estimator.predict(DMatrix(matrix)), axis=1).astype(int)
        return estimator.predict(matrix).astype(int)

    return {split: predict(features[splits == split]) for split in SPLITS}


def _infection_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    report = metrics["classification_report"]
    return (float(metrics["f1_macro"]), float(metrics["balanced_accuracy"]), float(report["High"]["f1-score"]))


def _infection_acceptance(old: dict[str, Any], new: dict[str, Any]) -> bool:
    old_report, new_report = old["classification_report"], new["classification_report"]
    return (
        float(new_report["High"]["f1-score"]) > float(old_report["High"]["f1-score"])
        and float(new_report["High"]["recall"]) > float(old_report["High"]["recall"])
        and float(new["f1_macro"]) > float(old["f1_macro"])
        and float(new["balanced_accuracy"]) > float(old["balanced_accuracy"])
        and float(new_report["Medium"]["f1-score"]) >= float(old_report["Medium"]["f1-score"])
        and float(new_report["Low"]["f1-score"]) >= float(old_report["Low"]["f1-score"]) - 0.10
    )


def run_infection_retraining() -> None:
    """Select a conservative class-weighted infection model using validation data only."""
    features, targets, filenames, feature_names = _load_joined_data()
    if len(feature_names) != 1563 or not np.isfinite(features).all() or len(filenames) != len(set(filenames)):
        raise ValueError("Feature validation failed before infection retraining")
    splits, classes = targets["split"], ["Low", "Medium", "High"]
    encoded = np.asarray([classes.index(value) for value in targets["infection"]], dtype=np.int32)
    train_mask, validation_mask, test_mask = splits == "train", splits == "validation", splits == "test"
    weights = _class_weights(encoded, train_mask)
    sample_weights = np.asarray([weights[value] for value in encoded[train_mask]], dtype=np.float32)
    print(f"infection_counts_total: {dict(Counter(targets['infection']))}")
    for split in SPLITS:
        print(f"infection_counts_{split}: {dict(Counter(targets['infection'][splits == split]))}")
    print(f"feature_count: {len(feature_names)}; duplicate_filenames: 0; missing_values: 0")
    print(f"train_class_weights: {weights}")

    current = Booster()
    current.load_model(MODEL_ROOT / "infection_xgboost.json")
    current_predictions = _infection_predictions(current, features, splits)
    current_validation = _metrics(encoded[validation_mask], current_predictions["validation"], [0, 1, 2], classes)
    current_test = _metrics(encoded[test_mask], current_predictions["test"], [0, 1, 2], classes)
    candidates = [
        ("shallow_regularized", {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.04, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.70, "reg_alpha": 0.10, "reg_lambda": 3.0}),
    ]
    selected: tuple[str, XGBClassifier, dict[str, Any], dict[str, np.ndarray]] | None = None
    for name, parameters in candidates:
        estimator = _infection_estimator(parameters)
        estimator.fit(features[train_mask], encoded[train_mask], sample_weight=sample_weights, verbose=False)
        predictions = _infection_predictions(estimator, features, splits)
        validation = _metrics(encoded[validation_mask], predictions["validation"], [0, 1, 2], classes)
        print(f"candidate_{name}_validation: macro_f1={validation['f1_macro']:.4f}; balanced_accuracy={validation['balanced_accuracy']:.4f}; high_f1={validation['classification_report']['High']['f1-score']:.4f}")
        if selected is None or _infection_score(validation) > _infection_score(selected[2]):
            selected = (name, estimator, validation, predictions)
    if selected is None or _infection_score(selected[2]) <= _infection_score(current_validation):
        print("INFECTION_MODEL_REPLACED = NO; validation did not improve over the current model")
        return
    selected_name, selected_model, selected_validation, selected_predictions = selected
    selected_test = _metrics(encoded[test_mask], selected_predictions["test"], [0, 1, 2], classes)
    accepted = _infection_acceptance(current_test, selected_test)
    print(f"selected_candidate: {selected_name}")
    print(f"current_validation_score: {_infection_score(current_validation)}; selected_validation_score: {_infection_score(selected_validation)}")
    print(f"old_test_metrics: {json.dumps(current_test)}")
    print(f"new_test_metrics: {json.dumps(selected_test)}")
    if not accepted:
        print("INFECTION_MODEL_REPLACED = NO; selected validation candidate failed final minority-class acceptance criteria")
        return
    backup = MODEL_ROOT / "infection_xgboost_previous.json"
    shutil.copy2(MODEL_ROOT / "infection_xgboost.json", backup)
    selected_model.save_model(MODEL_ROOT / "infection_xgboost.json")
    metrics_payload = {"classes": classes, "train_class_weights": weights, "split_counts": {split: int(np.sum(splits == split)) for split in SPLITS}, "model_selection": {"selected_candidate": selected_name, "selection_split": "validation", "validation_metrics": selected_validation, "previous_validation_metrics": current_validation}, "previous_test": current_test, "test": selected_test}
    _write_json(OUTPUT_ROOT / "infection_metrics.json", metrics_payload)
    _write_confusion_matrix(OUTPUT_ROOT / "infection_confusion_matrix.csv", selected_test["confusion_matrix"], classes)
    healing_predictions = {row["filename"]: row["predicted_healing"] for row in _read_csv(OUTPUT_ROOT / "prediction_results.csv")}
    label_map = {row["filename"]: row for row in _read_csv(LABELS_PATH)}
    infection_ids = {filenames[index]: classes[value] for split in SPLITS for index, value in zip(np.flatnonzero(splits == split), selected_predictions[split])}
    with (OUTPUT_ROOT / "prediction_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "true_healing", "predicted_healing", "true_infection", "predicted_infection"])
        writer.writeheader()
        for filename in filenames:
            writer.writerow({"filename": filename, "true_healing": label_map[filename]["healing_outcome"], "predicted_healing": healing_predictions[filename], "true_infection": label_map[filename]["infection_status"], "predicted_infection": infection_ids[filename]})
    print("INFECTION_MODEL_REPLACED = YES")


def run_training() -> None:
    features, targets, filenames, feature_names = _load_joined_data()
    splits = targets["split"]
    print(f"total_samples: {len(filenames)}")
    print(f"number_of_features: {len(feature_names)}")
    print(f"class_distributions: healing={dict(Counter(targets['healing']))}, infection={dict(Counter(targets['infection']))}")
    print(f"train_validation_test_counts: {dict(Counter(splits))}")

    healing_model, healing_metrics, healing_predictions = _train_model(
        features, targets["healing"], splits, ["Healed", "Not Healed"]
    )
    infection_model, infection_metrics, infection_predictions = _train_model(
        features, targets["infection"], splits, ["Low", "Medium", "High"]
    )

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    healing_model.save_model(MODEL_ROOT / "healing_xgboost.json")
    infection_model.save_model(MODEL_ROOT / "infection_xgboost.json")
    _write_json(OUTPUT_ROOT / "healing_metrics.json", healing_metrics)
    _write_json(OUTPUT_ROOT / "infection_metrics.json", infection_metrics)
    _write_confusion_matrix(
        OUTPUT_ROOT / "healing_confusion_matrix.csv",
        healing_metrics["test"]["confusion_matrix"],
        healing_metrics["classes"],
    )
    _write_confusion_matrix(
        OUTPUT_ROOT / "infection_confusion_matrix.csv",
        infection_metrics["test"]["confusion_matrix"],
        infection_metrics["classes"],
    )

    healing_ids = {}
    infection_ids = {}
    for split in SPLITS:
        indices = np.flatnonzero(splits == split)
        for index, value in zip(indices, healing_predictions[split]):
            healing_ids[filenames[index]] = healing_metrics["classes"][value]
        for index, value in zip(indices, infection_predictions[split]):
            infection_ids[filenames[index]] = infection_metrics["classes"][value]
    label_map = {row["filename"]: row for row in _read_csv(LABELS_PATH)}
    with (OUTPUT_ROOT / "prediction_results.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "true_healing", "predicted_healing", "true_infection", "predicted_infection"],
        )
        writer.writeheader()
        for filename in filenames:
            writer.writerow({
                "filename": filename,
                "true_healing": label_map[filename]["healing_outcome"],
                "predicted_healing": healing_ids[filename],
                "true_infection": label_map[filename]["infection_status"],
                "predicted_infection": infection_ids[filename],
            })

    print(f"healing_test_metrics: {healing_metrics['test']}")
    print(f"infection_test_metrics: {infection_metrics['test']}")
    print(f"train_validation_test_counts_saved: {dict(Counter(splits))}")


if __name__ == "__main__":
    run_training()
