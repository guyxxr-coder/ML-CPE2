## Data

Kaggle Cats and Dogs Dataset: Kaggle https://www.kaggle.com/datasets/shyamalb2/animal-dataset-csv

## Structure

```text
LAB04/
│
├── data-animal/
│   └── animal_dataset.csv
│
├── classification/
│   ├── main.py
│   ├── data_loader.py
│   ├── knn_tf.py
│   ├── evaluate.py
│   └── outputs/
│       ├── 01_k_curve.png
│       ├── 02_confusion_matrix.png
│       └── predictions.csv
│
├── clustering/
│   ├── main.py
│   ├── data_loader.py
│   ├── kmeans_tf.py
│   ├── knn_tools.py
│   ├── visualize.py
│   └── outputs/
│       ├── 01_elbow.png
│       ├── 02_clusters.png
│       ├── cluster_summary.csv
│       └── clustered_animals.csv
│
├── requirements.txt
└── link-data.txt
```

## Summary

This project demonstrates KNN for classification and clustering using an animal dataset. It includes data loading, preprocessing, model training, evaluation, and visualization.
