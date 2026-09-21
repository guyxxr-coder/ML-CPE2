# ML-07-Convolutional Neural Network (CNN)

Build a CNN pipeline using Python for image recognition. The project covers image loading, preprocessing, dataset splitting, CNN model training, evaluation, and prediction.
# Data

Kaggle Duck vs. Goose: (https://www.kaggle.com/datasets/patricia2025131053/duck-and-goose) ,(https://www.kaggle.com/datasets/ronnichang/goose-identification-dataset),(https://www.kaggle.com/datasets/alicenkbaytop/duck-images)
# Project Structure
```text
ML-07-CNN/
│
├── PetImages/                  
│   ├── Duck/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   │
│   └── Goose/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
│
├── classification/
│   ├── main.py                     # Main training pipeline
│   ├── data_loader.py              # Load images and skip corrupted files
│   ├── preprocessing.py            # Resize images and convert BGR to RGB
│   ├── split_data.py               # Split data into training, validation, and test sets
│   ├── cnn_model.py                # Build, train, save, and predict with the CNN model
│   ├── evaluate.py                 # Accuracy, classification report, confusion matrix, and training plots
│   ├── test_cnn.py                 # Test the trained model using four random images
│   └── outputs/                    
│       ├── features.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       ├── y_test.npy
│       ├── cnn_model.keras
│       ├── history.json
│       ├── confusion_matrix.png
│       ├── training_history.png
│       └── prediction_sample.png
└── requirements.txt
```
# Summary
โปรเจกต์นี้ใช้ CNN ในการจำแนกรูปภาพแมวและสุนัข โดยระบบจะโหลดรูปภาพจากไดเรกทอรีของชุดข้อมูลโดยอัตโนมัติ ปรับขนาดภาพให้เป็นความละเอียดที่กำหนด และแปลงระบบสีจาก BGR เป็น RGB ในช่วงกระบวนการเตรียมข้อมูล (Preprocessing) จากนั้นชุดข้อมูลจะถูกแบ่งออกเป็นชุดฝึกสอน (Training set) ชุดปรับแต่ง (Validation set) และชุดทดสอบ (Test set) ก่อนนำไปใช้ฝึกสอนโมเดล CNN สำหรับการประเมินประสิทธิภาพการจำแนกประเภทของโมเดลที่ฝึกสอนเสร็จแล้ว จะใช้วัดด้วยค่าความถูกต้อง (Accuracy), ค่าความแม่นยำ (Precision), ค่าความระลึก (Recall), ค่า F1-score, เมทริกซ์ความสับสน (Confusion matrix) และกราฟประวัติการฝึกสอน (Training history plots)
