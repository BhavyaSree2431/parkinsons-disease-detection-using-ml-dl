import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from fusion_model import predict_image_proba, predict_voice_proba


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VOICE_DATA = os.path.join(BASE_DIR, "dataset", "voice", "pd_speech_features.csv")

HEALTHY_DIR = os.path.join(BASE_DIR, "dataset", "handwriting", "Healthy")
PARKINSON_DIR = os.path.join(BASE_DIR, "dataset", "handwriting", "Parkinson")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_graphs")


def evaluate_fusion():

    print("\nEvaluating Fusion Model...\n")

    voice_df = pd.read_csv(VOICE_DATA)

    healthy_voice = voice_df[voice_df["class"] == 0].drop(columns=["id", "class"]).reset_index(drop=True)
    parkinson_voice = voice_df[voice_df["class"] == 1].drop(columns=["id", "class"]).reset_index(drop=True)

    healthy_images = [os.path.join(HEALTHY_DIR, img) for img in os.listdir(HEALTHY_DIR)]
    parkinson_images = [os.path.join(PARKINSON_DIR, img) for img in os.listdir(PARKINSON_DIR)]

    healthy_count = min(len(healthy_images), len(healthy_voice))
    parkinson_count = min(len(parkinson_images), len(parkinson_voice))

    actual = []
    image_probs = []
    voice_probs = []

    for i in range(healthy_count):

        print(f"Processing Healthy {i+1}/{healthy_count}")

        image_prob = float(predict_image_proba(healthy_images[i])[1])
        voice_prob = float(predict_voice_proba(healthy_voice.iloc[i]))

        image_probs.append(image_prob)
        voice_probs.append(voice_prob)
        actual.append(0)

    for i in range(parkinson_count):

        print(f"Processing Parkinson {i+1}/{parkinson_count}")

        image_prob = float(predict_image_proba(parkinson_images[i])[1])
        voice_prob = float(predict_voice_proba(parkinson_voice.iloc[i]))

        image_probs.append(image_prob)
        voice_probs.append(voice_prob)
        actual.append(1)

    actual = np.array(actual)
    image_probs = np.array(image_probs)
    voice_probs = np.array(voice_probs)

    (
        y_val,
        y_test,
        image_val,
        image_test,
        voice_val,
        voice_test,
    ) = split_for_validation_and_test(actual, image_probs, voice_probs)

    best_result = tune_fusion(y_val, image_val, voice_val)
    test_probs = combine_probs(
        image_test,
        voice_test,
        best_result["image_weight"],
        best_result["voice_weight"],
    )
    predictions = (test_probs >= best_result["threshold"]).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    print("\nValidation-Tuned Test Accuracy:", round(accuracy * 100, 2), "%")
    print(
        "Best Validation Settings:",
        f"image_weight={best_result['image_weight']:.2f},",
        f"voice_weight={best_result['voice_weight']:.2f},",
        f"threshold={best_result['threshold']:.2f}",
    )

    cv_accuracy, cv_f1 = cross_validate_fusion(actual, image_probs, voice_probs)
    print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%")
    print(f"5-Fold Cross-Validation Macro F1: {cv_f1:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_confusion_matrix(y_test, predictions)
    plot_roc(y_test, test_probs)
    plot_precision_recall(y_test, test_probs)
    plot_f1_score(y_test, predictions)


def split_for_validation_and_test(y_true, image_probs, voice_probs):

    return train_test_split(
        y_true,
        image_probs,
        voice_probs,
        test_size=0.30,
        stratify=y_true,
        random_state=42,
    )


def combine_probs(image_probs, voice_probs, image_weight, voice_weight):

    return (image_weight * image_probs) + (voice_weight * voice_probs)


def tune_fusion(y_true, image_probs, voice_probs):

    best_accuracy = -1.0
    best_result = None

    for image_weight in np.arange(0.1, 1.0, 0.1):
        voice_weight = 1.0 - image_weight

        fused_probs = combine_probs(image_probs, voice_probs, image_weight, voice_weight)

        for threshold in np.arange(0.30, 0.71, 0.02):
            predictions = (fused_probs >= threshold).astype(int)
            accuracy = accuracy_score(y_true, predictions)
            macro_f1 = f1_score(y_true, predictions, average="macro")

            if (
                accuracy > best_accuracy or
                (
                    np.isclose(accuracy, best_accuracy) and
                    best_result is not None and
                    macro_f1 > best_result["macro_f1"]
                )
            ):
                best_accuracy = accuracy
                best_result = {
                    "image_weight": float(image_weight),
                    "voice_weight": float(voice_weight),
                    "threshold": float(threshold),
                    "predictions": predictions,
                    "macro_f1": float(macro_f1),
                }

    return best_result


def cross_validate_fusion(y_true, image_probs, voice_probs, folds=5):

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies = []
    macro_f1_scores = []

    for train_index, test_index in skf.split(image_probs, y_true):
        y_train = y_true[train_index]
        y_fold_test = y_true[test_index]
        image_train = image_probs[train_index]
        image_fold_test = image_probs[test_index]
        voice_train = voice_probs[train_index]
        voice_fold_test = voice_probs[test_index]

        tuned = tune_fusion(y_train, image_train, voice_train)
        fold_probs = combine_probs(
            image_fold_test,
            voice_fold_test,
            tuned["image_weight"],
            tuned["voice_weight"],
        )
        fold_predictions = (fold_probs >= tuned["threshold"]).astype(int)

        accuracies.append(accuracy_score(y_fold_test, fold_predictions))
        macro_f1_scores.append(f1_score(y_fold_test, fold_predictions, average="macro"))

    return float(np.mean(accuracies)), float(np.mean(macro_f1_scores))


# ----------------------------
# Confusion Matrix
# ----------------------------

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0,1], ["Healthy","Parkinson"])
    plt.yticks([0,1], ["Healthy","Parkinson"])

    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.show()


# ----------------------------
# ROC Curve
# ----------------------------

def plot_roc(y_true, probs):

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1])

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "ROC_curve.png"), bbox_inches="tight")
    plt.show()


# ----------------------------
# Precision Recall Curve
# ----------------------------

def plot_precision_recall(y_true, probs):

    precision, recall, _ = precision_recall_curve(y_true, probs)

    plt.figure()
    plt.plot(recall, precision)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig(os.path.join(OUTPUT_DIR, "Precision-recall_Curve.png"), bbox_inches="tight")
    plt.show()


# ----------------------------
# F1 Score Graph
# ----------------------------

def plot_f1_score(y_true, y_pred):

    healthy_f1 = f1_score(y_true, y_pred, pos_label=0)
    parkinson_f1 = f1_score(y_true, y_pred, pos_label=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    labels = ["Healthy", "Parkinson", "Macro Avg"]
    scores = [healthy_f1, parkinson_f1, macro_f1]
    colors = ["#7bb274", "#d96c75", "#5b8def"]

    plt.figure()
    bars = plt.bar(labels, scores, color=colors)
    plt.ylim(0, 1.0)
    plt.title("F1-Score Comparison")
    plt.ylabel("F1 Score")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.2f}",
            ha="center",
        )

    plt.savefig(os.path.join(OUTPUT_DIR, "F1_score_graph.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    evaluate_fusion()