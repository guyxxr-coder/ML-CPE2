import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from src.feature_engineering import FeatureEngineer
from src.model_trainer import SVMModelTrainer

def main():
    print("Fetching historical data...")
    df = yf.download('NVDA', start='2021-01-01', end='2026-09-12')

    engineer = FeatureEngineer(df)
    processed_df = engineer.add_technical_indicators()

    feature_cols = [
        'Return_1D', 'Return_5D', 'SMA_Ratio_20', 'RSI', 'BB_Percent',
        'Return_1D_lag_1', 'Return_1D_lag_2', 'Return_1D_lag_3',
        'RSI_lag_1', 'RSI_lag_2'
    ]
    
    X = processed_df[feature_cols]
    y = processed_df['Target']

    # เทรนโมเดล
    trainer = SVMModelTrainer(kernel='rbf', C=1.0, gamma='scale')
    accuracy, report, y_test, y_pred = trainer.train_and_evaluate(X, y)

    print("="*40)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("="*40)
    print("\nClassification Report:")
    print(report)

    # --- วาดกราฟเปรียบเทียบ Cumulative Returns (Backtesting Plot) ---
    test_df = processed_df.iloc[-len(y_test):].copy()
    test_df['Signal'] = y_pred
    # ปรับ Signal: ถ้านายว่าขึ้น (1) ให้ถือหุ้น, ถ้านายว่าลง (0) ให้ถือเงินสด (0)
    test_df['Strategy_Return'] = test_df['Return_1D'] * test_df['Signal']
    
    # คำนวณผลตอบแทนสะสม
    cum_actual = (1 + test_df['Return_1D']).cumprod()
    cum_strategy = (1 + test_df['Strategy_Return']).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(cum_actual.index, cum_actual, label='Buy & Hold Strategy', color='gray', linestyle='--')
    plt.plot(cum_strategy.index, cum_strategy, label='SVM Strategy', color='green', linewidth=2)
    plt.title('Strategy Performance vs Buy & Hold (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return Multiplier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png', dpi=300) # บันทึกเป็นไฟล์ภาพ
    plt.show()

if __name__ == "__main__":
    main()