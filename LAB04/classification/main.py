import os
import pandas as pd
from data_loader import load_and_preprocess_classification
from knn_tf import train_knn
from evaluate import evaluate_models

def main():
    print("--- Starting Classification Task ---")
    data_path = "../data-animal/animal_dataset.csv"
    X_train, X_test, y_train, y_test, X_test_raw, _ = load_and_preprocess_classification(data_path)

    # ทดสอบค่า k = 3, 5, 7 (หรือปรับเปลี่ยนได้)
    k_list = [3, 5, 7]
    models = {}
    
    for k in k_list:
        models[k] = train_knn(X_train, y_train, k=k)

    # ประเมินผล
    output_dir = "outputs"
    best_k, best_acc, best_preds = evaluate_models(models, X_test, y_test, output_dir=output_dir)

    # บันทึกผลลัพธ์ Predictions
    output_df = X_test_raw.copy()
    output_df['Actual_Class'] = y_test.values
    output_df['Predicted_Class'] = best_preds
    output_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    print("\n--- Summary ---")
    print(f"Best K Value: {best_k}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Outputs saved to: {output_dir}/")

if __name__ == "__main__":
    main()