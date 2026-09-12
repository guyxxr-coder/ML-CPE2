import os
import yfinance as yf

# 1. เช็กว่ามีโฟลเดอร์ data ไหม ถ้ายังไม่มีให้สร้างให้อัตโนมัติ
if not os.path.exists('data'):
    os.makedirs('data')

# 2. ดึงข้อมูลหุ้น
df = yf.download('NVDA', start='2021-01-01', end='2026-09-12')

# 3. เซฟไฟล์ CSV ไปไว้ในโฟลเดอร์ data
df.to_csv('data/stock_data.csv')

print("บันทึกไฟล์ไว้ที่: data/stock_data.csv เรียบร้อยแล้ว!")