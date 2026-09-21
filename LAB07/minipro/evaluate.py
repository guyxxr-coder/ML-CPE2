import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
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
    """สร้างกราฟ 2x2 ตกแต่งสไตล์ IEEE Access"""

    epochs = range(1, len(history.history["accuracy"]) + 1)
    
    # ดึงค่าล่าสุดมาแสดงใน Legend
    final_tr_acc = history.history["accuracy"][-1] * 100
    final_val_acc = history.history["val_accuracy"][-1] * 100
    final_tr_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    # สร้าง Grid แบบ 2x2 (2 แถว 2 คอลัมน์)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    
    bg_color = "#EAEAF2"       # สีพื้นหลังสไตล์ IEEE
    line_color_tr = "#1f77b4"  # เส้นสีน้ำเงิน (Training)
    line_color_val = "#d62728" # เส้นสีแดง (Validation)

    # (a) Training Accuracy
    ax1 = axes[0, 0]
    ax1.set_facecolor(bg_color)
    ax1.plot(epochs, [x * 100 for x in history.history["accuracy"]], 
             color=line_color_tr, linewidth=2, label=f"CNN Model: {final_tr_acc:.2f}%")
    ax1.set_title("Training Performance", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=9)
    ax1.set_ylabel("Accuracy (%)", fontsize=9)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, color="white", linestyle="-", linewidth=1.2)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    ax1.text(0.5, -0.22, "(a) Training accuracy of model.", transform=ax1.transAxes,
             ha="center", fontsize=10, fontweight="bold")

    # (b) Validation Accuracy
    ax2 = axes[0, 1]
    ax2.set_facecolor(bg_color)
    ax2.plot(epochs, [x * 100 for x in history.history["val_accuracy"]], 
             color=line_color_val, linewidth=2, label=f"CNN Model: {final_val_acc:.2f}%")
    ax2.set_title("Training Performance", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=9)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(True, color="white", linestyle="-", linewidth=1.2)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    ax2.text(0.5, -0.22, "(b) Validation accuracy of model.", transform=ax2.transAxes,
             ha="center", fontsize=10, fontweight="bold")

    # (c) Training Loss
    ax3 = axes[1, 0]
    ax3.set_facecolor(bg_color)
    ax3.plot(epochs, history.history["loss"], 
             color=line_color_tr, linewidth=2, label=f"CNN Model: {final_tr_loss:.4f}")
    ax3.set_title("Loss Curve", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epochs", fontsize=9)
    ax3.set_ylabel("Loss", fontsize=9)
    ax3.grid(True, color="white", linestyle="-", linewidth=1.2)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    ax3.text(0.5, -0.22, "(c) Training loss of model.", transform=ax3.transAxes,
             ha="center", fontsize=10, fontweight="bold")

    # (d) Validation Loss
    ax4 = axes[1, 1]
    ax4.set_facecolor(bg_color)
    ax4.plot(epochs, history.history["val_loss"], 
             color=line_color_val, linewidth=2, label=f"CNN Model: {final_val_loss:.4f}")
    ax4.set_title("Loss Curve", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Epochs", fontsize=9)
    ax4.set_ylabel("Validation Loss", fontsize=9)
    ax4.grid(True, color="white", linestyle="-", linewidth=1.2)
    ax4.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    ax4.text(0.5, -0.22, "(d) Validation loss of model.", transform=ax4.transAxes,
             ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved IEEE-style plot: {save_path}")
