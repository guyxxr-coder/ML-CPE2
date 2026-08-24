DATA  Dataset: (https://www.kaggle.com/datasets/patricia2025131053/duck-and-goose)

## Structure

```text
LAB06/
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
│   ├── test_nn.py
│   ├── evaluate.py
│   └── outputs/
│       ├──features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── nn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
├── requirements.txt
└── link-data.txt
```

## Summary
The project uses a NN for Duck and Goose image recognition. Images are automatically loaded from the dataset directories, resized to a fixed resolution, and converted from BGR to RGB format during preprocessing. The dataset is then split into training, validation, and test sets before being used to train the neural network. The trained model is evaluated using accuracy, precision, recall, F1-score, a confusion matrix, and training history plots to assess its classification performance.
