DATA Kaggle Cats and Dogs Dataset: [https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/patricia2025131053/duck-and-goose)

## Structure

```text
LAB05/
├── PetImages/
│   ├── duck/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └── goose/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
├── minipro/
│   ├── main.py
│   ├── test_svm.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── svm_model.py
│   ├── evaluate.py
│   └── outputs/
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── svm_model.pkl
│       └── confusion_matrix.png
├── requirements.txt
└── link-data.txt
```

## Summary
The project uses SVM for duck and goose image recognition. Images are loaded from class directories, resized, converted into feature vectors, scaled, and then used to train an SVM classifier. The trained model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.


