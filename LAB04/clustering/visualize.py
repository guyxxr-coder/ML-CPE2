import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_elbow(X, max_k=10, output_dir="outputs"):
    from sklearn.cluster import KMeans
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "01_elbow.png"))
    plt.close()

def plot_clusters(X, labels, output_dir="outputs"):
    # ลดมิติข้อมูลด้วย PCA เพื่อแสดงผลเป็น 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=70)
    plt.title('Clusters Visualization (PCA 2D)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(os.path.join(output_dir, "02_clusters.png"))
    plt.close()