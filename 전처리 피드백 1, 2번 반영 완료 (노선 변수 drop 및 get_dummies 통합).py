##################################################################### 
# 1. Data Preprocessing
#####################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터셋 활용 및 삭제 (Drop)
df = pd.read_csv('Clean_Dataset.csv')

# 2. 과적합을 유발하는 편명(flight)과 불필요한 인덱스 삭제
columns_to_drop = []
if 'Unnamed: 0' in df.columns:
    columns_to_drop.append('Unnamed: 0')
if 'flight' in df.columns:
    columns_to_drop.append('flight')
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 3. 변수 생성 (Feature Engineering)
# 3-1. 노선(Route) 변수 생성: 출발지_도착지, 중복 제거 
df['route'] = df['source_city'] + "_" + df['destination_city']

# [피드백 1번 반영] 노선 변수 생성 후 불필요해진 출발지, 도착지 컬럼 삭제
df.drop(columns=['source_city', 'destination_city'], inplace=True)

# 3-2. 예약 시점 구간 변수 생성
# -1부터 시작하여 0일 데이터 손실(결측치) 완벽 방지
bins = [-1, 7, 21, df['days_left'].max()]
labels = ['1~7일 전(출발 직전)', '8~21일 전(단기)', '22일 이상(장기)']
df['booking_period'] = pd.cut(df['days_left'], bins=bins, labels=labels)

# 3-3. 타겟 변수 로그 변환 (가격 변동성 학습 최적화)
df['price'] = np.log1p(df['price'])

# =====================================================================
# 2. 숫자로 변환 (Encoding)
# =====================================================================
# ① class (절대 지우지 않고 0과 1로 매핑)
df['class'] = df['class'].map({'Economy': 0, 'Business': 1})

# ② stops: 직항 0, 1회 경유 1, 2회 이상 2
df['stops'] = df['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})

# [피드백 2번 반영] ③ 문자열 데이터 원-핫 인코딩
# 개별 인코딩 대신 get_dummies를 사용하고, 팀원분 요청대로 df_encoded 변수명 적용
df_encoded = pd.get_dummies(df, drop_first=True)

# =====================================================================
# 4. 데이터 분리 (Train/Test Split - 8:2 비율)
# =====================================================================
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== 최종 전처리 및 데이터 분리 완료 ===")
print(f"X_train 데이터 크기: {X_train.shape}")
print(f"X_test 데이터 크기: {X_test.shape}")
print(f"y_train 데이터 크기: {y_train.shape}")
print(f"y_test 데이터 크기: {y_test.shape}")

# =====================================================================
# 5. CSV 파일로 저장하기
# =====================================================================
# 1. 전체 전처리 완료된 데이터 하나로 저장하기 (df_encoded 기준)
df_encoded.to_csv('final_preprocessed_data.csv', index=False, encoding='utf-8-sig')

# 2. 머신러닝용으로 분리된 Train / Test 데이터를 각각 저장하기
train_df = pd.concat([X_train, y_train], axis=1)
train_df.to_csv('train_data.csv', index=False, encoding='utf-8-sig')

test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv('test_data.csv', index=False, encoding='utf-8-sig')

print("CSV 파일 저장이 완료되었습니다. 폴더를 확인해 주세요.")
