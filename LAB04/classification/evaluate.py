import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_models(models_dict, X_test, y_test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = list(models_dict.keys())
    accuracies = []
    best_k = None
    best_acc = -1
    best_model = None

    for k, model in models_dict.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"Accuracy for k={k}: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_model = model

    # 1. บันทึกกราฟ K-Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
    plt.title("KNN Accuracy vs. K Value")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "01_k_curve.png"))
    plt.close()

    # 2. บันทึก Confusion Matrix ของ Best K
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Best k={best_k})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(output_dir, "02_confusion_matrix.png"))
    plt.close()

    return best_k, best_acc, best_preds