import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from vgg_model import train_model, predict_model, VGG16_CONFIG, VGG_SMALL_CONFIG
from evaluate import evaluate_model, plot_history, plot_comparison_curves, plot_sample_predictions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "PetImages")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.2
MAX_PER_CLASS = 2000
BATCH_SIZE = 64


def main():
    print("=" * 60)
    print("VGG (PyTorch Deep CNN) - GPU Accelerated Training")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1-3: Load & Preprocess
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    X = to_features(images)
    y = labels

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    # Step 4: Run Experiments
    experiments = [
        {"name": "VGG_SMALL_15Epochs", "blocks": VGG_SMALL_CONFIG, "epochs": 15},
        {"name": "VGG_SMALL_30Epochs", "blocks": VGG_SMALL_CONFIG, "epochs": 30},
        {"name": "VGG16_15Epochs",      "blocks": VGG16_CONFIG,     "epochs": 15},
        {"name": "VGG16_30Epochs",      "blocks": VGG16_CONFIG,     "epochs": 30},
    ]

    histories = {}
    accuracy_scores = {}

    for exp in experiments:
        exp_name = exp["name"]
        exp_dir = os.path.join(OUTPUT_DIR, exp_name)
        
        print(f"\nRunning Experiment: {exp_name}")
        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            output_dir=exp_dir,
            epochs=exp["epochs"],
            batch_size=BATCH_SIZE,
            blocks=exp["blocks"]
        )

        predictions = predict_model(model, X_test)
        
        # 1. Save Confusion Matrix
        acc = evaluate_model(y_test, predictions, classes, save_path=os.path.join(exp_dir, "confusion_matrix.png"))
        
        # 2. Save Loss & Accuracy Curve per experiment
        plot_history(history, os.path.join(exp_dir, "history.png"))
        
        # 3. Save Sample Predictions Plot (สร้างรูปภาพตัวอย่างผลการทำนาย)
        plot_sample_predictions(
            model, X_test, y_test, classes,
            save_path=os.path.join(exp_dir, "sample_predictions.png")
        )

        histories[exp_name] = history.history
        accuracy_scores[exp_name] = acc

    # Step 5: Save Comparison Graph across all experiments
    plot_comparison_curves(histories, os.path.join(OUTPUT_DIR, "models_comparison.png"))

    print("\n" + "=" * 60)
    print(" SUMMARY OF ACCURACY SCORES ")
    print("=" * 60)
    for name, acc in accuracy_scores.items():
        print(f"{name:<25}: Accuracy = {acc * 100:.2f}%")


if __name__ == "__main__":
    main()