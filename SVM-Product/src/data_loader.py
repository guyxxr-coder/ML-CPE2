import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        """ดึงข้อมูลราคาหุ้นจาก Yahoo Finance"""
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return df