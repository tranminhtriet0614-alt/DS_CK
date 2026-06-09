import os
import numpy as np
import pandas as pd

print("==========================================================")
print("     TIẾN HÀNH BƯỚC: TIỀN XỬ LÝ VÀ CHUẨN BỊ DỮ LIỆU")
print("==========================================================")

input_origin = 'Data_bank_o.csv'
if not os.path.exists(input_origin):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc '{input_origin}'.")

# 1. Đọc dữ liệu gốc hoàn toàn từ ban đầu
df = pd.read_csv(input_origin, encoding='latin1')
print(f"-> Quy mô dữ liệu thô ban đầu (Data_bank_o): {df.shape[0]} dòng.")

# 2. XỬ LÝ DURATION = 0
df = df[df['duration'] > 0].copy()
print(f"-> Đã lọc bỏ các cuộc gọi ảo. Quy mô dữ liệu thực tế hiện tại: {df.shape[0]} dòng.")

# 3. MẠ KHÓA CÁC BIẾN NHỊ PHÂN GỐC (HOUSING, LOAN, DEFAULT)
print("\n[HỆ THỐNG TIẾN HÀNH MÃ HÓA BIẾN NHỊ PHÂN]")
binary_map = {'yes': 1, 'no': 0, 'unknown': 0}
df['housing'] = df['housing'].str.lower().map(binary_map).fillna(0).astype(int)
df['loan'] = df['loan'].str.lower().map(binary_map).fillna(0).astype(int)
df['default'] = df['default'].str.lower().map(binary_map).fillna(0).astype(int)

# 4. ÁP DỤNG MAPPING ĐỒNG BỘ THEO ĐỀ CƯƠNG (TRỪ ECONOMIC_PHASE SẼ LÀM Ở BƯỚC CLUSTER)
print("\n[HỆ THỐNG TIẾN HÀNH ÁNH XẠ BIẾN CÓ THỨ TỰ - MAPPING]")
df['contact'] = df['contact'].str.lower().map({'cellular': 1, 'telephone': 0}).fillna(0).astype(int)

months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month'] = df['month'].str.lower().map(months).fillna(5).astype(int)

dow = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
df['day_of_week'] = df['day_of_week'].str.lower().map(dow).fillna(1).astype(int)

df['poutcome'] = df['poutcome'].str.lower().map({'failure':0, 'nonexistent':1, 'success':2}).fillna(1).astype(int)
df['y'] = df['y'].str.lower().map({'yes':1, 'no':0}).fillna(0).astype(int)

# 5. RỜI RẠC HÓA DỮ LIỆU THỜI LƯỢNG VÀ SỐ LẦN GỌI
df['duration_group'] = pd.cut(df['duration'], bins=[0, 119, 299, 599, np.inf], labels=[1, 2, 3, 4]).astype(int)
df['campaign_group'] = df['campaign'].apply(lambda x: str(int(x)) if x < 5 else '5+')

# 6. XUẤT FILE ENCODED TẠM THỜI (Chưa có phase)
output_path = 'Data_bank_encoded.csv'
if os.path.exists(output_path):
    os.remove(output_path)
df.to_csv(output_path, index=False)

print("==========================================================")
print(f" 🎉 🎉 Đã xuất file chuẩn bị dữ liệu tại '{output_path}'")
print("==========================================================")