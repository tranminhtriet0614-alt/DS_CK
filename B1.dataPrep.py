import os
import numpy as np
import pandas as pd

print("==========================================================")
print("     TIẾN HÀNH BƯỚC: TIỀN XỬ LÝ VÀ CHUẨN BỊ DỮ LIỆU")
print("==========================================================")

INPUT_FILE  = 'Data_bank_o.csv'
OUTPUT_FILE = 'Data_bank_encoded.csv'

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc '{INPUT_FILE}'.")

# =========================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU VÀ LỌC CUỘC GỌI ẢO (DURATION = 0)
# =========================================================================
df = pd.read_csv(INPUT_FILE, encoding='latin1')
print(f"-> Quy mô dữ liệu thô ban đầu: {df.shape[0]:,} dòng.")

df = df[df['duration'] > 0].copy()
print(f"-> Đã lọc bỏ cuộc gọi ảo (duration = 0). Quy mô hiện tại: {df.shape[0]:,} dòng.")

# =========================================================================
# BƯỚC 2: MÃ HÓA CÁC BIẾN NHỊ PHÂN (HOUSING, LOAN, DEFAULT)
# =========================================================================
print("\n[HỆ THỐNG TIẾN HÀNH MÃ HÓA BIẾN NHỊ PHÂN]")
binary_map = {'yes': 1, 'no': 0, 'unknown': 0}
for col in ['housing', 'loan', 'default']:
    df[col] = df[col].str.lower().map(binary_map).fillna(0).astype(int)

# =========================================================================
# BƯỚC 3: ÁNH XẠ CÁC BIẾN CÓ THỨ TỰ (ORDINAL MAPPING)
# =========================================================================
print("\n[HỆ THỐNG TIẾN HÀNH ÁNH XẠ BIẾN CÓ THỨ TỰ]")

df['contact'] = df['contact'].str.lower().map({'cellular': 1, 'telephone': 0}).fillna(0).astype(int)

month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
             'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month'] = df['month'].str.lower().map(month_map).fillna(5).astype(int)

dow_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
df['day_of_week'] = df['day_of_week'].str.lower().map(dow_map).fillna(1).astype(int)

df['poutcome'] = df['poutcome'].str.lower().map({'failure':0, 'nonexistent':1, 'success':2}).fillna(1).astype(int)
df['y']        = df['y'].str.lower().map({'yes':1, 'no':0}).fillna(0).astype(int)

# =========================================================================
# BƯỚC 4: RỜI RẠC HÓA THỜI LƯỢNG VÀ SỐ LẦN GỌI
# =========================================================================
df['duration_group'] = pd.cut(
    df['duration'],
    bins=[0, 119, 299, 599, np.inf],
    labels=[1, 2, 3, 4]
).astype(int)

df['campaign_group'] = df['campaign'].apply(lambda x: str(int(x)) if x < 5 else '5+')

# =========================================================================
# BƯỚC 5: XUẤT FILE ENCODED (Chưa có economic_phase — sẽ bổ sung ở bước Clustering)
# =========================================================================
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("==========================================================")
print(f" HOÀN TẤT! Đã xuất file chuẩn bị dữ liệu tại '{OUTPUT_FILE}'")
print("==========================================================")
