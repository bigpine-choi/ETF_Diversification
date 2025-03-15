import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ ETF 리스트 및 파일 경로 설정
korean_etfs = [
    "428510", "456250", "456600", "465660", "463050", "466920", "478150", "486450", "475050", "468380",
    "153130", "475080", "481340", "434060"
]

df_list = []
for ticker in korean_etfs:
    filename = f"etf_{ticker}_2024-07-16_2025-03-14.csv"
    df = pd.read_csv(filename, usecols=["날짜", "종가", "거래량"])
    df["날짜"] = pd.to_datetime(df["날짜"])
    df.set_index("날짜", inplace=True)
    df.rename(columns={"종가": ticker + "_Close", "거래량": ticker + "_Volume"}, inplace=True)
    df_list.append(df)

# ✅ 데이터 병합
df_prices = pd.concat(df_list, axis=1).dropna()
df_prices.sort_index(inplace=True)

# ✅ 수익률 계산 (로그 수익률)
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ✅ 이동평균, 변동성, 모멘텀 등 특징 생성
for ticker in korean_etfs:
    df_prices[f"{ticker}_MA10"] = df_prices[f"{ticker}_Close"].rolling(window=10).mean()
    df_prices[f"{ticker}_MA50"] = df_prices[f"{ticker}_Close"].rolling(window=50).mean()
    df_prices[f"{ticker}_Volatility"] = df_returns[f"{ticker}_Close"].rolling(window=10).std()
    df_prices[f"{ticker}_Momentum"] = df_prices[f"{ticker}_Close"].diff(5)

df_prices.dropna(inplace=True)  # 결측값 제거

# ✅ 데이터 정규화
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_prices), columns=df_prices.columns, index=df_prices.index)

# ✅ 훈련 및 테스트 데이터 분리
train_size = int(len(df_scaled) * 0.8)
df_train = df_scaled.iloc[:train_size]
df_test = df_scaled.iloc[train_size:]

# ✅ LSTM 입력 데이터 변환
sequence_length = 30  # 30일간의 데이터를 입력으로 사용
features = df_train.shape[1]

generator_train = TimeseriesGenerator(df_train.values, df_train.values, length=sequence_length, batch_size=32)
generator_test = TimeseriesGenerator(df_test.values, df_test.values, length=sequence_length, batch_size=32)

# ✅ LSTM 모델 구축
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(features, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ✅ 모델 학습
history = model.fit(generator_train, validation_data=generator_test, epochs=50, batch_size=32, verbose=1)

# ✅ 학습 과정 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.xlabel('Epochs')
plt.ylabel('손실(MSE)')
plt.title('LSTM 모델 학습 손실')
plt.legend()
plt.grid()
plt.show()

# ✅ 모델 저장
model.save("lstm_etf_model.h5")

# ✅ 모델 로드 및 예측 수행
model = load_model("lstm_etf_model.h5")
predictions = model.predict(generator_test)

# ✅ 예측 결과 역정규화
predictions_rescaled = scaler.inverse_transform(predictions)
df_test_rescaled = scaler.inverse_transform(df_test.values)

# ✅ 예측 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(df_prices.index[-len(predictions):], df_test_rescaled[:, 0], label="실제 종가")
plt.plot(df_prices.index[-len(predictions):], predictions_rescaled[:, 0], label="예측 종가", linestyle='dashed')
plt.xlabel("날짜")
plt.ylabel("가격")
plt.title("LSTM 기반 ETF 가격 예측")
plt.legend()
plt.grid()
plt.show()

# ✅ 성능 평가 (MSE 출력)
mse = np.mean((predictions_rescaled - df_test_rescaled) ** 2)
print(f"예측 모델 MSE: {mse:.5f}")