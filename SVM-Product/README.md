## ระบบทำนายทิศทางราคาหุ้นรายวัน (Up/Down) โดยใช้แบบจำลอง **Support Vector Classification (SVC)** ร่วมกับตัวชี้วัดทางเทคนิค (Technical Indicators) และคุณลักษณะการหน่วงเวลา (Time-Series Lag Features) เพื่อแก้ปัญหา Model Bias และยกระดับเสถียรภาพของการทำนายข้อมูลทางการเงิน

SVM-Product/
│
├── data/
│   └── stock_data.csv        # ไฟล์ข้อมูลราคาหุ้นที่บันทึกไว้
│
├── src/
│   ├── data_loader.py         # โมดูลดึงข้อมูลราคาหุ้นผ่าน yfinance API
│   ├── feature_engineering.py # โมดูลสร้าง Technical Indicators และ Lag Features
│   └── model_trainer.py       # โมดูลสเกลข้อมูล (StandardScaler) เทรน และวาด Confusion Matrix
│
├── main.py                    # ไฟล์หลักควบคุม Pipeline ทั้งหมด และทำ Backtesting
├── confusion_matrix.png       # กราฟ Heatmap แสดงผลการทำนายราย Class
├── cumulative_returns.png     # กราฟเปรียบเทียบผลตอบแทนสะสม
└── README.md                  # สรุปรายละเอียดโปรเจกต์

