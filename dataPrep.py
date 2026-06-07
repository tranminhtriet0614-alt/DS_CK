import pandas as pd
import numpy as np
import os

print("==========================================================")
print("     TIẾN HÀNH BƯỚC: TIỀN XỬ LÝ VÀ CHUẨN BỊ DỮ LIỆU")
print("==========================================================")

# 1. Đọc dữ liệu gốc (Kế thừa từ kết quả bước phân cụm/giai đoạn trước)
df = pd.read_csv('.\\Data_bank_final_results.csv')
print(f"-> Quy mô dữ liệu ban đầu: {df.shape[0]} dòng.")


# 2. XỬ LÝ DURATION = 0 (Yêu cầu bắt buộc để phục vụ chính xác cho RQ2 và Mô hình)
# Lọc bỏ các dòng duration == 0 vì cuộc gọi không diễn ra thực sự, giữ lại sẽ làm lệch phân tích thời lượng
df = df[df['duration'] > 0].copy()
print(f"-> Đã lọc bỏ các cuộc gọi ảo. Quy mô dữ liệu thực tế hiện tại: {df.shape[0]} dòng.")


# 3. LỌC BỎ CỘT CHỮ VÀ ĐỔI TÊN CỘT BINARY HOUSING, LOAN (Theo yêu cầu mới)
# Xóa bỏ 2 cột chứa chữ 'yes'/'no' cũ để không bị trùng tên
df = df.drop(columns=['housing', 'loan'])

# Đổi tên các cột '_bi' thành tên sạch không có chữ 'bi' như ban đầu
df = df.rename(columns={
    'housing_bi': 'housing',
    'loan_bi': 'loan'
})
print("-> Đã xóa bỏ các cột chữ cũ và chuẩn hóa cột Binary thành tên: 'housing', 'loan'.")


# 4. Áp dụng mapping cho các biến theo yêu cầu của đề cương
print("\n[HỆ THỐNG TIẾN HÀNH ÁNH XẠ BIẾN - MAPPING]")

# contact: binary: cellular=1, telephone=0
df['contact'] = df['contact'].str.lower().map({'cellular': 1, 'telephone': 0})

# month: ordinal theo thứ tự tháng (jan=1 ... dec=12)
months = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month'] = df['month'].str.lower().map(months)

# day_of_week: ordinal mon=1 ... fri=5
dow = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
df['day_of_week'] = df['day_of_week'].str.lower().map(dow)

# poutcome: ordinal mapping failure=0, nonexistent=1, success=2
df['poutcome'] = df['poutcome'].str.lower().map({'failure':0, 'nonexistent':1, 'success':2})

# y: binary yes=1, no=0
df['y'] = df['y'].str.lower().map({'yes':1, 'no':0})

# economic_phase: map nhãn chu kỳ vĩ mô sang số thứ tự [1, 2, 3]
# Đã sửa chữ 'Chạm đáy' viết hoa chữ C cho khớp chuẩn xác tuyệt đối với dữ liệu gốc
df['economic_phase'] = df['economic_phase'].map({
    'Giai đoạn Tăng trưởng': 1,
    'Giai đoạn Suy thoái': 2,
    'Giai đoạn Chạm đáy': 3
})


# 5. RỜI RẠC HÓA DỮ LIỆU THỜI LƯỢNG VÀ SỐ LẦN GỌI (Binning & Clipping)

# duration_group: 4 nhóm theo yêu cầu
# 1: cuộc gọi quá ngắn (<2p), 2: trao đổi ngắn (2-5p), 3: tư vấn đầy đủ (5-10p), 4: tư vấn chuyên sâu (>10p)
df['duration_group'] = pd.cut(
    df['duration'],
    bins=[0, 119, 299, 599, np.inf],
    labels=[1, 2, 3, 4]
).astype('Int64')

# campaign_group: 1-4 lần giữ nguyên, từ 5 lần trở lên nén toàn bộ thành nhóm 5 để tránh nhiễu đuôi dài
df['campaign_group'] = df['campaign'].where(df['campaign'] < 5, 5).astype('Int64')


# 6. KIỂM TRA LẠI DỮ LIỆU TRỰC QUAN (Data Inspection)
print("\n==========================================================")
print("              KIỂM TRA NHANH KẾT QUẢ MÃ HÓA")
print("==========================================================")
# Thêm housing và loan vào bảng kiểm tra để xem hiển thị
check_cols = ['contact','housing','loan','month','day_of_week','poutcome','y','economic_phase','duration_group','campaign_group']
print(df[check_cols].head())

print('\nDtypes định dạng các cột sau xử lý:\n', df[check_cols].dtypes)


# 7. XUẤT VÀ LƯU FILE ĐÃ MÃ HÓA (Đảm bảo dọn dẹp file cũ trước khi ghi)
output_path = '.\Data_bank_encoded.csv'
if os.path.exists(output_path):
    os.remove(output_path)

df.to_csv(output_path, index=False)

print("==========================================================")
print(f" 🎉 HOÀN TẤT: Xuất file sạch thành công tại '{output_path}'")
print("==========================================================")