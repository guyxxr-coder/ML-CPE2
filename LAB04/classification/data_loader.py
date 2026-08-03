import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_classification(data_path="../data-animal/animal_dataset.csv"):
    df = pd.read_csv(data_path)
    
    # 1. แยก Target (Species) ออกมา
    y = df['Species']
    
    # 2. ลบ Animal_ID และ Species ออกจาก Feature (X)
    X = df.drop(columns=['Animal_ID', 'Species'], errors='ignore')
    
    # 3. แปลงคอลัมน์ที่เป็นข้อความ (Categorical) ให้เป็นตัวเลขด้วย One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # 4. แบ่งข้อมูล Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. ทำ Standardization ปรับสเกลข้อมูลตัวเลข
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test, scaler