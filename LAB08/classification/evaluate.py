import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_comparison_curves(histories, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for name, hist in histories.items():
        max_acc = max(hist["accuracy"]) * 100
        axes[0, 0].plot(hist["accuracy"], label=f"{name}: {max_acc:.2f}%")
    axes[0, 0].set_title("Training Performance")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend()

    for name, hist in histories.items():
        max_val = max(hist["val_accuracy"]) * 100
        axes[0, 1].plot(hist["val_accuracy"], label=f"{name}: {max_val:.2f}%")
    axes[0, 1].set_title("Validation Performance")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Validation Accuracy (%)")
    axes[0, 1].legend()

    for name, hist in histories.items():
        min_loss = min(hist["loss"])
        axes[1, 0].plot(hist["loss"], label=f"{name}: {min_loss:.4f}")
    axes[1, 0].set_title("Loss Curve (Training)")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    for name, hist in histories.items():
        min_val_loss = min(hist["val_loss"])
        axes[1, 1].plot(hist["val_loss"], label=f"{name}: {min_val_loss:.4f}")
    axes[1, 1].set_title("Loss Curve (Validation)")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison curves to: {save_path}")


def plot_sample_predictions(model, X_test, y_test, classes, save_path, n_samples=6):
    """ สุ่มเลือกภาพจาก Test Set มาทำนายผลและบันทึกเป็นรูปภาพ """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    model.eval()
    model.to(device)
    inputs = torch.tensor(X_sample, dtype=torch.uint8).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        if len(classes) == 2:
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            predictions = (probs > 0.5).astype(int)
            confidence = np.where(predictions == 1, probs, 1 - probs)
        else:
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = probs.argmax(axis=1)
            confidence = probs.max(axis=1)

    cols = 3
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_samples):
        pred_label = classes[predictions[i]]
        true_label = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        axes[i].imshow(X_sample[i])
        axes[i].axis("off")
        axes[i].set_title(
            f"Pred: {pred_label} ({confidence[i]*100:.1f}%)\nTrue: {true_label}",
            color=color
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved predictions plot to: {save_path}")