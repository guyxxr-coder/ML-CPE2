import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_clustering(data_path="../data-animal/animal_dataset.csv"):
    df = pd.read_csv(data_path)
    
    # 1. ตัดคอลัมน์ Animal_ID ออกจากการคำนวณ
    X = df.drop(columns=['Animal_ID'], errors='ignore')
    
    # 2. แปลงข้อความทั้งหมด (รวมถึง Species, Diet_Type ฯลฯ) ให้เป็นตัวเลขด้วย One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 3. สเกลข้อมูลตัวเลข
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    return df, X_scaled