DATA Kaggle Cats and Dogs Dataset: (https://www.kaggle.com/datasets/patricia2025131053/duck-and-goose)

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

## Result
<img width="494" height="433" alt="image" src="https://github.com/user-attachments/assets/9ebc220a-edf7-47fc-8457-750910cb8cf0" />
<img width="700" height="420" alt="c1413a68-00f1-4c59-a7d7-1fdbadc246e5" src="https://github.com/user-attachments/assets/59d21c28-ae94-4d46-b46e-b42f3880373c" />
<img width="459" height="453" alt="image (1)" src="https://github.com/user-attachments/assets/400d4cd1-bfbf-4360-97b5-a6d961559722" />



