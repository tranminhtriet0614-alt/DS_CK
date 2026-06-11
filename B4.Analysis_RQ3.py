

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


# Import các thư viện phân tích & Machine Learning (dự phòng cho các bước sau)
from scipy.stats import chi2_contingency, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve,
                             confusion_matrix, classification_report)


# =========================================================================
# CẤU HÌNH HỆ THỐNG & HIỂN THỊ
# =========================================================================
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


print("==========================================================")
print("     TIẾN HÀNH RQ3: HIỆU QUẢ TÁI TIẾP CẬN THEO CHU KỲ VĨ MÔ")
print("==========================================================")


# =========================================================================
# BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU ĐỒNG BỘ
# =========================================================================
input_file = './Data_bank_encoded.csv'
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Không tìm thấy file '{input_file}'. Vui lòng chạy file dữ liệu sạch trước!")


df = pd.read_csv(input_file)


# Định nghĩa các bộ từ điển nhãn hiển thị tiếng Việt chuẩn
poutcome_map = {
    1: '1. Nhóm lần đầu',
    0: '2. Nhóm tái tiếp cận',
    2: '3. Nhóm kiểm soát dương'
}


phase_map = {
    1: 'Giai đoạn Tăng trưởng',
    2: 'Giai đoạn Suy thoái',
    3: 'Giai đoạn chạm đáy'
}


# Ánh xạ nhãn
df['reapproach_group'] = df['poutcome'].map(poutcome_map)
df['phase_name'] = df['economic_phase'].map(phase_map)


# Làm sạch và ép kiểu dữ liệu
df = df.dropna(subset=['reapproach_group', 'economic_phase', 'y'])
df['economic_phase'] = df['economic_phase'].astype(int)
df['y'] = df['y'].astype(int)


# Thứ tự hiển thị chuẩn của các giai đoạn kinh tế vĩ mô
phase_order = ['Giai đoạn Tăng trưởng', 'Giai đoạn Suy thoái', 'Giai đoạn chạm đáy']




# =========================================================================
# BƯỚC 2: MÔ TẢ ĐẶC TRƯNG & TỶ LỆ CHUYỂN ĐỔI BASELINE
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 2: MÔ TẢ ĐẶC TRƯNG CẤU TRÚC CÁC NHÓM KHÁCH HÀNG")
print("="*50)


print("SỐ LƯỢNG MẪU THỰC TẾ MỖI NHÓM TƯƠNG TÁC:")
print(df['reapproach_group'].value_counts().to_string())
print("-" * 60)


# Tính toán tỷ lệ chuyển đổi baseline
conversion_rate = df.groupby('reapproach_group', observed=False)['y'].mean().reset_index()


# Trực quan hóa tỷ lệ chuyển đổi Baseline
plt.figure(figsize=(9, 5))
sns.barplot(data=conversion_rate, x='reapproach_group', y='y', hue='reapproach_group', palette='Set2', legend=False)
plt.title('Tỷ lệ Chuyển đổi giữa các Nhóm Khách hàng (RQ3 Baseline Benchmark)', fontsize=11, fontweight='bold')
plt.xlabel('Phân nhóm theo Lịch sử tương tác', fontsize=10)
plt.ylabel('Tỷ lệ Chuyển đổi (Conversion Rate)', fontsize=10)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))


# Hiển thị số liệu %
for index, row in conversion_rate.iterrows():
    plt.text(index, row['y'] + 0.01, f"{row['y']:.2%}", ha='center', fontweight='bold', fontsize=9)


plt.tight_layout()
plt.savefig('./RQ3_baseline_conversion.png', dpi=300)
plt.close()
print("-> [HỆ THỐNG] Đã xuất file biểu đồ Baseline thành công: './RQ3_baseline_conversion.png'\n")




# =========================================================================
# BƯỚC 3: BẢNG CHÉO TỶ LỆ CHUYỂN ĐỔI THEO GIAI ĐOẠN VÀ NHÓM KHÁCH HÀNG
# =========================================================================
# Trục dòng: Phân nhóm khách hàng, Trục cột: Giai đoạn vĩ mô
crosstab_rate = df.pivot_table(index='reapproach_group', columns='phase_name', values='y', aggfunc='mean') * 100
# Sắp xếp lại cột theo đúng trật tự thời gian
crosstab_rate = crosstab_rate[phase_order]


print("="*75)
print("BẢNG TRUNG TÂM RQ3: TỶ LỆ CHUYỂN ĐỔI THEO GIAI ĐOẠN VÀ NHÓM KHÁCH HÀNG (%)")
print("="*75)
print(crosstab_rate.round(2).to_string())
print("\n")




# =========================================================================
# BƯỚC 4: PHÂN PHỐI SỐ LƯỢNG & TỶ LỆ CẤU TRÚC MẪU
# =========================================================================
# Bảng 1: Số lượng mẫu (Counts)
df_counts = pd.crosstab(df['reapproach_group'], df['phase_name'])[phase_order]
print("="*80)
print("BẢNG 1: SỐ LƯỢNG MẪU THỰC TẾ (COUNT) PHÂN BỔ QUA CÁC GIAI ĐOẠN")
print("="*80)
display(df_counts)


# Bảng 2: Tỷ lệ cấu trúc (Row percentages)
df_pct = pd.crosstab(df['reapproach_group'], df['phase_name'], normalize='index')[phase_order] * 100
print("\n" + "="*80)
print("BẢNG 2: TỶ LỆ % CẤU TRÚC PHÂN PHỐI MẪU THEO TỪNG NHÓM KHÁCH HÀNG")
print("="*80)
display(df_pct.round(2))


# =========================================================================
# BƯỚC 5: BIỂU ĐỒ CỘT CHỒNG TRỰC QUAN (STACKED BAR CHART)
# =========================================================================
colors = ['#2b5c8f', '#d95f02', '#7570b3']
ax = df_pct.plot(kind='barh', stacked=True, color=colors, figsize=(10, 5), width=0.6)


plt.title('CẤU TRÚC PHÂN PHỐI CUỘC GỌI VÀO CÁC GIAI ĐOẠN KINH TẾ (%)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Tỷ lệ phần trăm (%)', fontsize=10)
plt.ylabel('Nhóm khách hàng', fontsize=10)
plt.xlim(0, 100)
plt.legend(title='Bối cảnh vĩ mô', bbox_to_anchor=(1.02, 1), loc='upper left')


# Tự động chèn số % vào từng phân đoạn thanh ngang
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if width > 5:  # Điều kiện chỉ hiện text nếu chiều rộng đủ lớn
        ax.text(x + width/2, y + height/2, f'{width:.1f}%',
                horizontalalignment='center', verticalalignment='center',
                color='white', fontweight='bold', fontsize=9)


plt.tight_layout()
plt.savefig('./distribution_simpson.png', dpi=300)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score
from IPython.display import display


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# =====================================================================
# BƯỚC 3.4: LOGISTIC REGRESSION ĐỐI KHÁNG (CHỈ GIỮ NONEXISTENT VÀ FAILURE)
# =====================================================================


# --- BƯỚC 1: ĐỌC DỮ LIỆU & LỌC BỎ NHÓM SUCCESS ---
INPUT_FILE = './Data_bank_encoded.csv' # Điều chỉnh đường dẫn nếu cần
df_raw = pd.read_csv(INPUT_FILE)
print("Shape dữ liệu ban đầu cấu trúc tổng thể:", df_raw.shape)


# Lọc bỏ nhóm 'Success' (mã 2)
df = df_raw[df_raw['poutcome'].isin([0, 1])].copy()
print("Shape dữ liệu sau khi lọc bỏ nhóm Success:", df.shape)


poutcome_map = {
    1: '1. Nhóm lần đầu (Nonexistent)',
    0: '2. Nhóm tái tiếp cận (Failure)'
}
df['reapproach_group'] = df['poutcome'].map(poutcome_map)


print("\n" + "="*70)
print("SỐ LƯỢNG MẪU THEO NHÓM ĐỐI KHÁNG (ĐÃ LOẠI BỎ SUCCESS)")
print("="*70)
print(df['reapproach_group'].value_counts().to_string())
print(f"Tỷ lệ chuyển đổi baseline trên tập dữ liệu lọc: {df['y'].mean():.2%}\n")




# --- BƯỚC 2: MÃ HÓA BIẾN GIẢ DUMMY, TẠO BIẾN TƯƠNG TÁC CHUẨN ---
df_model = df.dropna(subset=['contact', 'campaign', 'duration', 'age', 'job', 'economic_phase', 'y']).copy()


# 1. poutcome thành Dummy: 0 = Nhóm gốc (Nonexistent), 1 = Failure
df_model['poutcome_Failure'] = np.where(df_model['poutcome'] == 0, 1, 0)


# 2. One-hot Encoding cho biến job
df_model['job'] = df_model['job'].astype(str)
job_dummies = pd.get_dummies(df_model['job'], prefix='job', drop_first=True)
df_model = pd.concat([df_model, job_dummies], axis=1)
job_features = job_dummies.columns.tolist()


# 3. Chuẩn hóa Z-score các biến định lượng
scaler = StandardScaler()
numeric_features = ['contact', 'campaign', 'duration', 'age']
df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])


y = df_model['y'].astype(int).values


# 4. Tạo biến tương tác
df_model['poutcome_Failure_x_economic_phase'] = df_model['poutcome_Failure'] * df_model['economic_phase']


# Gom features
X_features = numeric_features + job_features + ['economic_phase', 'poutcome_Failure', 'poutcome_Failure_x_economic_phase']
X = df_model[X_features].values


print("="*70)
print("MA TRẬN DỮ LIỆU ĐƯA VÀO MÔ HÌNH HỒI QUY ĐỐI KHÁNG")
print("="*70)
print(f"Số lượng mẫu dữ liệu: {X.shape[0]}")
print(f"Số lượng đặc trưng đầu vào: {X.shape[1]}")
print(f"Cấu trúc lớp: Lớp 0 ({np.sum(y==0)} mẫu) | Lớp 1 ({np.sum(y==1)} mẫu)\n")




# --- BƯỚC 3: CROSS-VALIDATION ĐÁNH GIÁ MÔ HÌNH ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, f1_scores = [], []


for train_idx, val_idx in skf.split(X, y):
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[val_idx])
    y_pred_proba = model.predict_proba(X[val_idx])[:, 1]
   
    auc_scores.append(roc_auc_score(y[val_idx], y_pred_proba))
    f1_scores.append(f1_score(y[val_idx], y_pred))




# --- BƯỚC 4: HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG & XUẤT BẢNG KẾT QUẢ ĐẸP ---
log_reg_final = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
log_reg_final.fit(X, y)


# Khởi tạo DataFrame lưu trữ hệ số
coef_df = pd.DataFrame({
    'Biến': X_features,
    'Hệ số (Beta)': log_reg_final.coef_[0]
})


# Thêm Hệ số chặn (Intercept) từ mô hình Sklearn vào DataFrame
intercept_df = pd.DataFrame({
    'Biến': ['Intercept'],
    'Hệ số (Beta)': log_reg_final.intercept_[0]
})
coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)


# Tính Odds Ratio và % thay đổi
coef_df['Odds Ratio'] = np.exp(coef_df['Hệ số (Beta)'])
coef_df['Khả năng chuyển đổi thay đổi (%)'] = (coef_df['Odds Ratio'] - 1) * 100


# Hàm làm sạch tên biến cho gọn
def clean_feature_name(name):
    if name == 'Intercept': return 'Hệ số chặn'
    name = name.replace('poutcome_Failure_x_economic_phase', 'poutcome_x_economic_phase')
    name = name.replace('poutcome_Failure', 'poutcome_failure')
    name = name.replace('job_', 'job_') # Đã sạch từ pd.get_dummies
    return name


coef_df['Biến'] = coef_df['Biến'].apply(clean_feature_name)


# Sắp xếp theo Beta giảm dần, đẩy Hệ số chặn xuống cuối
coef_df = coef_df.sort_values(by='Hệ số (Beta)', ascending=False)
intercept_row = coef_df[coef_df['Biến'] == 'Hệ số chặn']
coef_df = coef_df[coef_df['Biến'] != 'Hệ số chặn']
coef_df = pd.concat([coef_df, intercept_row]).reset_index(drop=True)


# Làm tròn số liệu
coef_df['Hệ số (Beta)'] = coef_df['Hệ số (Beta)'].round(6)
coef_df['Odds Ratio'] = coef_df['Odds Ratio'].round(6)
coef_df['Khả năng chuyển đổi thay đổi (%)'] = coef_df['Khả năng chuyển đổi thay đổi (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")


print("="*85)
print(" BẢNG TỔNG HỢP HỆ SỐ HỒI QUY LOGISTIC (ĐÃ LÀM GỌN THEO CHUẨN)")
print("="*85)
# Set index là 'Biến' để in ra không bị vướng cột STT 0, 1, 2...
display(coef_df.set_index('Biến'))




# --- BƯỚC 5: TỔNG KẾT BÁO CÁO KHOA HỌC ---
y_pred = log_reg_final.predict(X)
interaction_row = coef_df[coef_df['Biến'] == 'poutcome_x_economic_phase'].iloc[0]


print("\n" + "="*85)
print(" KẾT LUẬN CUỐI CÙNG CHO CÂU HỎI RQ3 (KỊCH BẢN ĐỐI KHÁNG)")
print("="*85)
print(f"📊 Hiệu suất mô hình kiểm soát:")
print(f"   - AUC-ROC (Cross-Val): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"   - F1-Score (Cross-Val):  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


print(f"\n🎯 BIẾN TƯƠNG TÁC LÕI (poutcome_x_economic_phase):")
print(f"   - Hệ số hồi quy: {interaction_row['Hệ số (Beta)']}")
print(f"   - Odds Ratio: {interaction_row['Odds Ratio']}")
print(f"   - Mức độ tác động: {interaction_row['Khả năng chuyển đổi thay đổi (%)']}")




