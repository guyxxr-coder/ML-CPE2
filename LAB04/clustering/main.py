import os
import pandas as pd
from data_loader import load_and_preprocess_clustering
from kmeans_tf import run_kmeans
from visualize import plot_elbow, plot_clusters

def main():
    print("--- Starting Clustering Task ---")
    data_path = "../data-animal/animal_dataset.csv"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    df, X_scaled = load_and_preprocess_clustering(data_path)

    # 1. Elbow Method เพื่อเลือก k ที่เหมาะสม
    plot_elbow(X_scaled, max_k=10, output_dir=output_dir)

    # 2. ทำ Clustering (เช่น เลือก k=4 หรือตามเหมาะสม)
    optimal_k = 4
    kmeans, labels = run_kmeans(X_scaled, n_clusters=optimal_k)

    # 3. Visualizations
    plot_clusters(X_scaled, labels, output_dir=output_dir)

    # 4. Save CSV Outputs
    df['Cluster'] = labels
    df.to_csv(os.path.join(output_dir, "clustered_animals.csv"), index=False)

    # สร้าง cluster summary
    summary = df.groupby('Cluster').mean(numeric_only=True)
    summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"))

    print(f"Clustering complete. Outputs saved to: {output_dir}/")

if __name__ == "__main__":
    main()