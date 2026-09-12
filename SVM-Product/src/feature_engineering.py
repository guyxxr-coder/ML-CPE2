import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_technical_indicators(self) -> pd.DataFrame:
        close_price = self.df['Close']
        if isinstance(close_price, pd.DataFrame):
            close_price = close_price.iloc[:, 0]
        close_price = pd.to_numeric(close_price, errors='coerce')

        # Target: ทิศทางราคาปิดวันพรุ่งนี้เทียบกับวันนี้
        self.df['Tomorrow_Close'] = close_price.shift(-1)
        self.df['Target'] = np.where(self.df['Tomorrow_Close'] > close_price, 1, 0)
        
        # 1. Base Indicators
        self.df['Return_1D'] = close_price.pct_change(1)
        self.df['Return_5D'] = close_price.pct_change(5)
        
        sma_20 = close_price.rolling(window=20).mean()
        self.df['SMA_Ratio_20'] = close_price / sma_20

        # RSI (14 Days)
        delta = close_price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands %B
        std_20 = close_price.rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        self.df['BB_Percent'] = (close_price - lower_band) / (upper_band - lower_band)

        # 2. *** เพิ่ม Lag Features (ดึงค่าของวันอดีต t-1, t-2 มาใช้อธิบายวันปัจจุบัน) ***
        for lag in [1, 2, 3]:
            self.df[f'Return_1D_lag_{lag}'] = self.df['Return_1D'].shift(lag)
            self.df[f'RSI_lag_{lag}'] = self.df['RSI'].shift(lag)

        self.df = self.df.dropna()
        return self.df