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