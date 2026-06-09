
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve, 
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore')
from IPython.display import display, HTML

warnings.filterwarnings('ignore')
print("==========================================================")
print("     TIẾN HÀNH RQ3: HIỆU QUẢ TÁI TIẾP CẬN THEO CHU KỲ VĨ MÔ")
print("==========================================================")

# 1. ĐỌC DỮ LIỆU ĐÃ MÃ HÓA ĐỒNG BỘ
input_file = '.\Data_bank_encoded.csv'
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Không tìm thấy file '{input_file}'. Vui lòng chạy file dữ liệu sạch trước!")

df = pd.read_csv(input_file)

# Cấu hình hiển thị trọn vẹn bảng thống kê trên Terminal của nhóm
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Định nghĩa các bộ từ điển nhãn hiển thị tiếng Việt chuẩn
phase_names = {1: '1. Tăng trưởng', 2: '2. Suy thoái', 3: '3. Chạm đáy'}
poutcome_map = {
    1: '1. Nhóm lần đầu (Nonexistent)',
    0: '2. Nhóm tái tiếp cận (Failure)',
    2: '3. Nhóm kiểm soát dương (Success)'
}

# Ánh xạ nhãn phân nhóm lịch sử tương tác dựa trên cột poutcome số đã encoded
df['reapproach_group'] = df['poutcome'].map(poutcome_map)

# Loại bỏ các dòng khuyết thiếu cục bộ nếu có để đảm bảo tính toán cấu trúc sạch
df = df.dropna(subset=['reapproach_group', 'economic_phase', 'y'])
df['economic_phase'] = df['economic_phase'].astype(int)
df['y'] = df['y'].astype(int)


# =========================================================================
# BƯỚC 3.1 · TẠO VÀ MÔ TẢ ĐẶC ĐIỂM NHÓM TÁI TIẾP CẬN
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 3.1: MÔ TẢ ĐẶC TRƯNG CẤU TRÚC CÁC NHÓM KHÁCH HÀNG")
print("="*50)

# 1. Kiểm tra số lượng mẫu thực tế phân bổ
print("SỐ LƯỢNG MẪU THỰC TẾ MỖI NHÓM TƯƠNG TÁC:")
print(df['reapproach_group'].value_counts().to_string())
print("-" * 60)

# 2. Tính toán tỷ lệ chuyển đổi baseline (benchmark) giữa các nhóm
conversion_rate = df.groupby('reapproach_group', observed=False)['y'].mean().reset_index()

# Trực quan hóa Biểu đồ cột Tỷ lệ chuyển đổi Baseline
plt.figure(figsize=(9, 5))
# Dùng thuộc tính 'hue' để tránh cảnh báo phân rã bảng màu của seaborn mới
sns.barplot(data=conversion_rate, x='reapproach_group', y='y', hue='reapproach_group', palette='Set2', legend=False)
plt.title('Tỷ lệ Chuyển đổi giữa các Nhóm Khách hàng (RQ3 Baseline Benchmark)', fontsize=11, fontweight='bold')
plt.xlabel('Phân nhóm theo Lịch sử tương tác (poutcome)', fontsize=10)
plt.ylabel('Tỷ lệ Chuyển đổi (Conversion Rate)', fontsize=10)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Hiển thị số liệu % cụ thể trên đầu mỗi cột dữ liệu
for index, row in conversion_rate.iterrows():
    plt.text(index, row['y'] + 0.01, f"{row['y']:.2%}", ha='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('.\RQ3_baseline_conversion.png', dpi=300)
plt.close() # Giải phóng bộ nhớ ngầm, không gây treo luồng Terminal
print("-> [HỆ THỐNG] Đã xuất file biểu đồ Baseline thành công: '.\RQ3_baseline_conversion.png'")

# 3. Thống kê mô tả các biến định lượng cốt lõi (Độ tuổi và Tần suất gọi)
print("\n" + "-"*60)
print("THỐNG KÊ ĐẶC ĐIỂM ĐỊNH LƯỢNG TRUNG BÌNH THEO PHÂN NHÓM:")
print("-" * 60)
num_desc = df.groupby('reapproach_group', observed=False)[['age', 'campaign']].mean()
print(num_desc.round(2).to_string())

# 4. Thống kê cấu trúc các biến định tính (Nghề nghiệp và Học vấn) theo tỷ lệ % nội bộ cột
print("\n" + "-"*60)
print("BẢNG PHÂN BỔ CẤU TRÚC NGHỀ NGHIỆP (JOB) THEO TỪNG NHÓM (%)")
print("-" * 60)
job_ct = pd.crosstab(df['job'], df['reapproach_group'], normalize='columns') * 100
print(job_ct.round(2).to_string())

print("\n" + "-"*60)
print("BẢNG PHÂN BỔ TRÌNH ĐỘ HỌC VẤN (EDUCATION) THEO TỪNG NHÓM (%)")
print("-" * 60)
edu_ct = pd.crosstab(df['education'], df['reapproach_group'], normalize='columns') * 100
print(edu_ct.round(2).to_string())

# =========================================================================
# BƯỚC 3.2 · SO SÁNH TỶ LỆ CHUYỂN ĐỔI GIỮA CÁC NHÓM THEO GIAI ĐOẠN
# =========================================================================
print("\n" + "="*75)
print(" BƯỚC 3.2: MA TRẬN CHUYỂN ĐỔI VÀ KIỂM ĐỊNH CHI-SQUARE ĐỐI KHÁNG")
print("="*75)

# 1. Tạo bảng chéo trung tâm (Pivot Table) tính tỷ lệ chuyển đổi (%) cho 3 nhóm qua 3 giai đoạn
crosstab_rate = df.pivot_table(index='reapproach_group', columns='economic_phase', values='y', aggfunc='mean', observed=False) * 100

# Ánh xạ nhãn tiếng Việt cho trục cột (giai đoạn vĩ mô) để in ấn đẹp mắt
crosstab_rate.columns = crosstab_rate.columns.map(phase_names)

print("BẢNG TRUNG TÂM RQ3: TỶ LỆ CHUYỂN ĐỔI THEO GIAI ĐOẠN VÀ NHÓM KHÁCH HÀNG (%)")
print("-" * 75)
print(crosstab_rate.round(2).to_string())
print("-" * 75)

# 2. Trực quan hóa Ma trận tỷ lệ chuyển đổi bằng Heatmap sắc nét
plt.figure(figsize=(11, 5.5))
# Dùng bộ màu "YlGnBu" chuẩn phân tích mật độ
sns.heatmap(crosstab_rate / 100, annot=True, fmt=".2%", cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Ma trận Hiệu quả Tái tiếp cận qua các Chu kỳ Vĩ mô (Pivot Summary)", fontsize=12, fontweight='bold')
plt.ylabel("Phân nhóm Khách hàng (Lịch sử tương tác)", fontsize=10)
plt.xlabel("Bối cảnh Chu kỳ Kinh tế", fontsize=10)
plt.tight_layout()

# Tự động xuất ảnh chất lượng cao chèn báo cáo, không gây treo luồng Terminal
plt.savefig('.\RQ3_step32_heatmap.png', dpi=300)
plt.close()
print("\n-> [HỆ THỐNG] Đã xuất file biểu đồ Ma trận nhiệt thành công: '.\RQ3_step32_heatmap.png'")

# 3. KIỂM ĐỊNH CHI-SQUARE TỪNG GIAI ĐOẠN (SO SÁNH ĐỐI KHÁNG: LẦN ĐẦU VS TÁI TIẾP CẬN)
print("\n" + "="*75)
print("KẾT QUẢ KIỂM ĐỊNH CHI-SQUARE CHO TỪNG BỐI CẢNH VĨ MÔ (Khách Mới vs Tái Tiếp Cận)")
print("="*75)

# Danh sách mã số các giai đoạn kinh tế theo cấu trúc dữ liệu sạch
phases_list = [1, 2, 3]

for phase_code in phases_list:
    # Lọc dữ liệu: Chỉ lấy giai đoạn hiện tại VÀ chỉ bốc 2 nhóm đối kháng (Lần đầu & Tái tiếp cận)
    phase_df = df[(df['economic_phase'] == phase_code) & 
                  (df['reapproach_group'].isin(['1. Nhóm lần đầu (Nonexistent)', '2. Nhóm tái tiếp cận (Failure)']))]
    
    # Tạo bảng tần suất (Contingency Table) dựa trên số lượng tuyệt đối để phục vụ kiểm định
    contingency_matrix = pd.crosstab(phase_df['reapproach_group'], phase_df['y'])
    
    print(f"▶ Bối cảnh: {phase_names[phase_code].upper()}")
    
    # Đảm bảo bảng liên hợp hợp lệ (đủ 2 hàng, 2 cột) để tránh lỗi thuật toán
    if contingency_matrix.shape[0] == 2 and contingency_matrix.shape[1] == 2:
        chi2, p_val, dof, expected = chi2_contingency(contingency_matrix)
        
        print(f"  - Chi-square Statistic: {chi2:.4f}")
        print(f"  - p-value: {p_val:.4e}")
        
        # Biện luận kết quả học thuật sắc bén dựa trên mốc alpha = 0.05
        if p_val < 0.05:
            print(f"  => Kết luận: Có sự khác biệt mang Ý NGHĨA THỐNG KÊ (p < 0.05) về tỷ lệ chuyển đổi.")
            print(f"               Bối cảnh vĩ mô này tác động mạnh làm phân hóa rõ rệt hiệu quả giữa khách mới và khách cũ.")
        else:
            print(f"  => Kết luận: Không có sự khác biệt mang ý nghĩa thống kê (p >= 0.05).")
            print(f"               Hiệu quả tiếp cận giữa hai nhóm khách hàng này là tương đương nhau dưới góc độ thống kê.")
    else:
        print("  - [CẢNH BÁO] Không đủ cấu trúc số lượng mẫu đối kháng để thực hiện kiểm định Chi-square.")
    print("-" * 75)



# =========================================================================
# BƯỚC 3.3 · SO SÁNH TỶ LỆ CHUYỂN ĐỔI GIỮA CÁC NHÓM THEO GIAI ĐOẠN
# =========================================================================
import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal

# =========================================================================
# 1. ĐỌC DỮ LIỆU & TIỀN XỬ LÝ ĐỒNG BỘ
# =========================================================================
file_name = 'Data_bank_encoded.csv'
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Không tìm thấy file '{file_name}'.")

df_rq3 = pd.read_csv(file_name)
df_pdays = df_rq3[(df_rq3['poutcome'] == 0) & (df_rq3['pdays'] != 999)].copy()

phase_map = {1: 'Giai đoạn Tăng trưởng', 2: 'Giai đoạn Suy thoái', 3: 'Giai đoạn chạm đáy'}
df_pdays['phase_name'] = df_pdays['economic_phase'].map(phase_map)

def segment_pdays(days):
    if days <= 7:   return '1. Dưới 1 tuần (0-7 ngày)'
    elif days <= 14: return '2. 1-2 tuần (8-14 ngày)'
    else:            return '3. Trên 2 tuần (>14 ngày)'

df_pdays['pdays_group'] = df_pdays['pdays'].apply(segment_pdays)
pdays_order = ['1. Dưới 1 tuần (0-7 ngày)', '2. 1-2 tuần (8-14 ngày)', '3. Trên 2 tuần (>14 ngày)']
df_pdays['pdays_group'] = pd.Categorical(df_pdays['pdays_group'], categories=pdays_order, ordered=True)

# =========================================================================
# 2. XUẤT MA TRẬN BẢNG CHÉO (DẠNG TEXT THUẦN CHO TERMINAL / FILE .PY)
# =========================================================================
pivot_pdays = df_pdays.pivot_table(index='phase_name', columns='pdays_group', values='y', aggfunc='mean', observed=False) * 100

print("="*85)
print("📊 BẢNG CHÉO BƯỚC 3.3: TỶ LỆ CHUYỂN ĐỔI THEO KHOẢNG NGÀY CHỜ VÀ GIAI ĐOẠN (%)")
print("="*85)
# Sử dụng .to_string() để in bảng ngay ngắn ra màn hình đen Terminal
print(pivot_pdays.round(2).to_string())
print("\n" + "="*85)

# =========================================================================
# 3. KIỂM ĐỊNH KRUSKAL-WALLIS (IN CHỮ TRỰC TIẾP KHÔNG CẦN DISPLAY)
# =========================================================================
print(" KẾT QUẢ KIỂM ĐỊNH PHI THAM SỐ KRUSKAL-WALLIS (BIẾN PDAYS)")
print("="*85)

active_phases = ['Giai đoạn Suy thoái', 'Giai đoạn chạm đáy']
for phase in active_phases:
    phase_data = df_pdays[df_pdays['phase_name'] == phase]
    pdays_yes = phase_data[phase_data['y'] == 1]['pdays'].values
    pdays_no = phase_data[phase_data['y'] == 0]['pdays'].values
    
    if len(pdays_yes) > 0 and len(pdays_no) > 0:
        stat, p_val = kruskal(pdays_yes, pdays_no)
        is_significant = "Có ý nghĩa (p < 0.05)" if p_val < 0.05 else "Không ý nghĩa (p >= 0.05)"
        
        print(f"Bối cảnh: {phase}")
        print(f"  - H-statistic: {stat:.4f}")
        print(f"  - p-value: {p_val:.4e} ({is_significant})")
        if p_val < 0.05:
            print(f"  -> Hệ quả: Khoảng cách ngày chờ có sự phân hóa hành vi rõ rệt giữa 2 nhóm kết quả.")
        else:
            print(f"  -> Hệ quả: Tỷ lệ chuyển đổi đồng đều, số ngày chờ không làm lệch phân phối.")
        print("-" * 85)



# Cấu hình đọc file đồng bộ từ bước phân cụm vĩ mô
INPUT_FILE = './Data_bank_encoded.csv'

def check_file_exists():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file mã hóa '{INPUT_FILE}'. Vui lòng chạy bước Phân cụm trước!")

# =====================================================================
# BƯỚC 3.4: LOGISTIC REGRESSION VỚI BIẾN TƯƠNG TÁC (GIỮ NGUYÊN 3 NHÓM)
# =====================================================================

# --- BƯỚC 1: ĐỌC DỮ LIỆU & KIỂM TRA ĐỒNG BỘ ---
check_file_exists()
df = pd.read_csv(INPUT_FILE)
print("Shape dữ liệu ban đầu:", df.shape)

# Tạo nhãn text dựa trên mã số poutcome để in báo cáo kiểm tra
poutcome_map = {
    1: '1. Nhóm lần đầu',
    0: '2. Nhóm tái tiếp cận',
    2: '3. Nhóm kiểm soát dương'
}
df['reapproach_group'] = df['poutcome'].map(poutcome_map)

print("="*70)
print("SỐ LƯỢNG MẪU THEO NHÓM (GIỮ TRỌN VẸN CẢ 3 NHÓM)")
print("="*70)
print(df['reapproach_group'].value_counts())
print(f"Tỷ lệ chuyển đổi baseline tổng thể: {df['y'].mean():.2%}\n")

# Kiểm tra biến chu kỳ kinh tế từ bước phân cụm
if 'economic_phase' not in df.columns:
    print("⚠️ CẢNH BÁO: Biến 'economic_phase' không tồn tại!")
else:
    print("✓ Biến 'economic_phase' tồn tại chuẩn xác:")
    print(df['economic_phase'].value_counts().sort_index())


# --- BƯỚC 2: KIỂM TRA ĐỊNH DẠNG CÁC BIẾN KIỂM SOÁT ---
control_vars = ['contact', 'campaign', 'duration', 'age', 'job']
print("\n" + "="*70)
print("KIỂM TRA CÁC BIẾN KIỂM SOÁT")
print("="*70)
for var in control_vars:
    if var not in df.columns:
        print(f"❌ Biến '{var}' KHÔNG tồn tại!")
    else:
        print(f"✓ Biến '{var}' tồn tại - Type: {df[var].dtype}")

print("\nKiểm tra giá trị thiếu (Missing values):")
print(df[control_vars + ['economic_phase', 'y']].isnull().sum())


# --- BƯỚC 3: TẠO BIẾN TƯƠNG TÁC VÀ CHUẨN BỊ MA TRẬN DỮ LIỆU ---
df_model = df.copy()

# Ép kiểu dữ liệu của job thành string để tránh lỗi One-hot Encoding
df_model['job'] = df_model['job'].astype(str)
job_dummies = pd.get_dummies(df_model['job'], prefix='job', drop_first=True)
df_model = pd.concat([df_model, job_dummies], axis=1)
job_features = job_dummies.columns.tolist()

# Chuẩn hóa Z-score các biến định lượng để đưa về cùng thang đo mẫu
scaler = StandardScaler()
numeric_features = ['contact', 'campaign', 'duration', 'age']
df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])

# Định nghĩa nhãn mục tiêu y
y = df_model['y'].values

# TẠO BIẾN TƯƠNG TÁC CHÍNH THỨC TRÊN TẬP TỔNG THỂ (Không bị triệt tiêu về 0)
df_model['poutcome_x_economic_phase'] = df_model['poutcome'] * df_model['economic_phase']

# Gom tập hợp các biến vào ma trận X đặc trưng
X_features = numeric_features + job_features + ['economic_phase', 'poutcome_x_economic_phase']
X = df_model[X_features].values

print("\n" + "="*70)
print("MA TRẬN DỮ LIỆU ĐƯA VÀO MÔ HÌNH HỒI QUY")
print("="*70)
print(f"Số lượng mẫu dữ liệu: {X.shape[0]}")
print(f"Số lượng đặc trưng đầu vào: {X.shape[1]}")
print(f"Cấu trúc biến phụ thuộc mục tiêu (y):")
print(f"  - Lớp 0 (Từ chối): {(y == 0).sum()} mẫu ({(y == 0).sum()/len(y):.2%})")
print(f"  - Lớp 1 (Chuyển đổi): {(y == 1).sum()} mẫu ({(y == 1).sum()/len(y):.2%})")


# --- BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH VỚI STRATIFIED K-FOLD CROSS-VALIDATION ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
auc_scores = []
f1_scores = []

print("\n" + "="*70)
print("STRATIFIED K-FOLD CROSS-VALIDATION (5 Folds)")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Huấn luyện mô hình Logistic Regression trên từng fold huấn luyện riêng biệt
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Dự đoán xác suất và nhãn lớp trên tập kiểm thử chéo
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    auc_score = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc_score)
    
    f1 = f1_score(y_val, y_pred)
    f1_scores.append(f1)
    
    fold_results.append({
        'Fold': fold,
        'AUC-ROC': auc_score,
        'F1-Score': f1,
        'Train_Size': len(train_idx),
        'Val_Size': len(val_idx)
    })
    print(f"Fold {fold}: AUC-ROC = {auc_score:.4f}, F1-Score = {f1:.4f}")

# Tính toán các giá trị trung bình tổng kết chéo
cv_results = pd.DataFrame(fold_results)


# --- BƯỚC 5: XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION CUỐI CÙNG ---
log_reg_final = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
log_reg_final.fit(X, y)

print("\n" + "="*70)
print("HỆ SỐ CỦA MÔ HÌNH LOGISTIC REGRESSION CUỐI CÙNG")
print("="*70)
coefficients = pd.DataFrame({
    'Feature': X_features,
    'Coefficient': log_reg_final.coef_[0],
    'Odds_Ratio': np.exp(log_reg_final.coef_[0])
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)
print(coefficients.to_string(index=False))
print(f"\n{'-'*70}")
print(f"INTERCEPT (Hệ số tự do): {log_reg_final.intercept_[0]:.6f}")
print(f"{'='*70}")

# Tiền xử lý lấy vị trí chỉ mục biến tương tác để làm báo cáo nhanh
interaction_idx = X_features.index('poutcome_x_economic_phase')
interaction_coef = log_reg_final.coef_[0][interaction_idx]
interaction_odds = np.exp(interaction_coef)

print(f"\n🎯 BIẾN TƯƠNG TÁC (poutcome × economic_phase):")
print(f"   Hệ số hồi quy: {interaction_coef:.6f}")
print(f"   Chỉ số Odds Ratio: {interaction_odds:.4f}")
print(f"   Diễn giải: Khi giai đoạn kinh tế tăng thêm 1 đơn vị,")
print(f"   khả năng chuyển đổi thay đổi tương quan là {(interaction_odds-1)*100:.2f}%")


# --- BƯỚC 6: TÍNH TOÁN BẢNG ODDS RATIO VÀ DIỄN GIẢI CHI TIẾT ---
print("\n" + "="*70)
print("BẢNG ODDS RATIO VÀ HƯỚNG TÁC ĐỘNG CHI TIẾT CHÍNH XÁC")
print("="*70)
odds_ratio_df = pd.DataFrame({
    'Biến': X_features,
    'Hệ số': log_reg_final.coef_[0],
    'Odds Ratio': np.exp(log_reg_final.coef_[0]),
    'Thay đổi %': (np.exp(log_reg_final.coef_[0]) - 1) * 100
})
odds_ratio_df = odds_ratio_df.sort_values('Odds Ratio', ascending=False)
print(odds_ratio_df.to_string(index=False))
print("")

for idx, row in odds_ratio_df.iterrows():
    var_name = row['Biến']
    odds_ratio = row['Odds Ratio']
    change_pct = row['Thay đổi %']
    interpretation = f"Tăng khả năng chuyển đổi {change_pct:.2f}%" if odds_ratio > 1 else f"Giảm khả năng chuyển đổi {abs(change_pct):.2f}%"
    print(f"   {var_name:30s}: OR = {odds_ratio:.4f} → {interpretation}")

# Làm nổi bật kết quả biến tương tác quan trọng nhất
print(f"\n{'*'*70}")
print("⭐ TỔNG HỢP BIẾN TƯƠNG TÁC (poutcome × economic_phase):")
interaction_row = odds_ratio_df[odds_ratio_df['Biến'] == 'poutcome_x_economic_phase'].iloc[0]
print(f"   Odds Ratio thu được: {interaction_row['Odds Ratio']:.4f}")
print(f"   Mức độ thay đổi tương quan: {interaction_row['Thay đổi %']:.2f}%")
print(f"{'*'*70}")


# --- BƯỚC 7: BÁO CÁO FINAL & TỔNG KẾT KHOA HỌC CHO RQ3 ---
y_pred = log_reg_final.predict(X)
y_pred_proba = log_reg_final.predict_proba(X)[:, 1]
from sklearn.metrics import auc as auc_func
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc_score_val = auc_func(fpr, tpr)

print("\n" + "="*70)
print("FINAL CLASSIFICATION REPORT (Toàn bộ dữ liệu tổng hợp)")
print("="*70)
print(classification_report(y, y_pred, target_names=['Không chuyển đổi', 'Chuyển đổi']))

print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
cm = confusion_matrix(y, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0, 0]} | False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]} | True Positives: {cm[1, 1]}")

print("\n" + "="*70)
print("ĐỘ ĐO HIỆU SUẤT CHÍNH CỦA MÔ HÌNH")
print("="*70)
print(f"AUC-ROC (Toàn bộ bộ dữ liệu): {roc_auc_score_val:.4f}")
print(f"F1-Score (Toàn bộ bộ dữ liệu): {f1_score(y, y_pred):.4f}")
print(f"Cross-Val AUC-ROC (Trung bình ± Độ lệch): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Cross-Val F1-Score (Trung bình ± Độ lệch):  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

print("\n" + "="*70)
print("KẾT LUẬN CUỐI CÙNG CHO CÂU HỎI RQ3")
print("="*70)
print(f"""✓ Mô hình Logistic Regression với biến tương tác (poutcome × economic_phase) đã thực thi thành công.
📊 Hiệu suất mô hình:
   - AUC-ROC (Cross-Val): {np.mean(auc_scores):.4f}
   - F1-Score (Cross-Val):  {np.mean(f1_scores):.4f}
🎯 Biến tương tác (poutcome × economic_phase):
   - Odds Ratio: {interaction_row['Odds Ratio']:.4f}
   - Ảnh hưởng: {interaction_row['Thay đổi %']:.2f}%
   
   Diễn giải: Dưới sự tác động điều tiết của chu kỳ vĩ mô, mức độ đóng góp của lịch sử tương tác 
   đối với khả năng gửi tiền thành công của khách hàng đã {('TĂNG LÊN' if interaction_row['Thay đổi %'] > 0 else 'GIẢM ĐI')} {abs(interaction_row['Thay đổi %']):.2f}% 
   khi nền kinh tế tịnh tiến dịch chuyển qua các giai đoạn vĩ mô khác nhau.
🔍 Các biến kiểm soát đã được đưa vào mô hình ổn định:
   - contact, campaign, duration, age, job
✅ Phân tích hồi quy RQ3 đã hoàn thành đồng bộ thành công!""")


# --- BƯỚC 8: TRỰC QUAN HÓA KẾT QUẢ PHÂN TÍCH (VISUALIZATION) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Đồ thị 1: ROC Curve đường cong hiệu năng mô hình
ax = axes[0, 0]
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_score_val:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve - Logistic Regression\n(Toàn bộ tập dữ liệu tích hợp)', fontsize=11, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

# Đồ thị 2: Thanh ngang thể hiện giá trị Odds Ratio của các thuộc tính
ax = axes[0, 1]
importance_df = odds_ratio_df.sort_values('Odds Ratio')
colors = ['#e74c3c' if x < 1 else '#2ecc71' for x in importance_df['Odds Ratio']]
ax.barh(importance_df['Biến'], importance_df['Odds Ratio'], color=colors, alpha=0.7)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Odds Ratio', fontsize=10)
ax.set_title('Odds Ratio của các Biến số hệ thống\n(> 1: Tăng chuyển đổi | < 1: Giảm chuyển đổi)', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Đồ thị 3: Điểm số Cross-Validation đính kèm thanh Error bar
ax = axes[1, 0]
metrics_data = {
    'AUC-ROC': [np.mean(auc_scores), np.std(auc_scores)],
    'F1-Score': [np.mean(f1_scores), np.std(f1_scores)]
}
x_pos = np.arange(len(metrics_data))
means = [v[0] for v in metrics_data.values()]
stds = [v[1] for v in metrics_data.values()]
bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=['skyblue', 'lightcoral'])
ax.set_ylabel('Score', fontsize=10)
ax.set_title('Kết quả Cross-Validation (5-Fold)\nVới thanh Error Bar kiểm định', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_data.keys())
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.02, f'{mean:.4f}', ha='center', fontweight='bold')

# Đồ thị 4: Bản đồ nhiệt trực quan hóa giá trị hệ số hồi quy Beta trọng số
ax = axes[1, 1]
coef_values = log_reg_final.coef_[0].reshape(-1, 1)
im = ax.imshow(coef_values, cmap='RdBu_r', aspect='auto', vmin=-np.max(np.abs(coef_values)), vmax=np.max(np.abs(coef_values)))
ax.set_yticks(range(len(X_features)))
ax.set_yticklabels(X_features, fontsize=9)
ax.set_xticks([0])
ax.set_xticklabels(['Coefficient'], fontsize=9)
ax.set_title('Hệ số Trọng số Mô hình Beta\n(Đỏ: Âm, Xanh: Dương)', fontsize=11, fontweight='bold')
for i, coef in enumerate(coef_values.flatten()):
    ax.text(0, i, f'{coef:.3f}', ha='center', va='center', 
            color='white' if abs(coef) > np.max(np.abs(coef_values))/2 else 'black', fontweight='bold')
plt.colorbar(im, ax=ax, label='Coefficient Value')

plt.tight_layout()
plt.show()
print("✓ Toàn bộ đồ thị trực quan hóa phân tích của nhóm đã khởi tạo thành công!")




# =====================================================================
# BƯỚC 3.5: MA TRẬN QUYẾT ĐỊNH & KHUYẾN NGHỊ TÁI TIẾP CẬN ĐỒNG BỘ VĨ MÔ
# =====================================================================

# --- 1. ĐỌC DỮ LIỆU TRUNG TÂM ---
df = pd.read_csv('./Data_bank_encoded.csv')

# Áp dụng bản đồ ánh xạ giai đoạn chuẩn xác (Giai đoạn 3 sửa thành Chạm đáy)
phase_names = {1: 'Giai đoạn 1 (Tăng trưởng)', 2: 'Giai đoạn 2 (Suy thoái)', 3: 'Giai đoạn 3 (Chạm đáy)'}
df['phase_name'] = df['economic_phase'].map(phase_names)

# Tách phân lớp dữ liệu theo lịch sử tương tác poutcome gốc
df_recontact = df[df['poutcome'] == 0].copy()  # Nhóm tái tiếp cận
df_first = df[df['poutcome'] == 1].copy()      # Nhóm liên hệ lần đầu

print('✓ Bộ dữ liệu tích hợp đã tải thành công!')
print(f'  - Tổng quy mô mẫu quan sát: {len(df)}')
print(f'  - Quy mô phân lớp Tái tiếp cận: {len(df_recontact)}')
print(f'  - Quy mô phân lớp Gọi lần đầu:  {len(df_first)}\n')


# --- 2. TÍNH TOÁN TỶ LỆ CHUYỂN ĐỔI THEO TỪNG GIAI ĐOẠN VĨ MÔ ---
recontact_by_phase = df_recontact.groupby('economic_phase')['y'].agg(['sum', 'count', 'mean']).reset_index()
recontact_by_phase.columns = ['Giai đoạn', 'Chuyển đổi', 'Tổng', 'Tỷ lệ']
recontact_by_phase['Tỷ lệ %'] = (recontact_by_phase['Tỷ lệ'] * 100).round(2)
recontact_by_phase['Tên Giai Đoạn'] = recontact_by_phase['Giai đoạn'].map(phase_names)

first_by_phase = df_first.groupby('economic_phase')['y'].agg(['sum', 'count', 'mean']).reset_index()
first_by_phase.columns = ['Giai đoạn', 'Chuyển đổi', 'Tổng', 'Tỷ lệ']
first_by_phase['Tỷ lệ %'] = (first_by_phase['Tỷ lệ'] * 100).round(2)

print('='*70)
print('TỶ LỆ CHUYỂN ĐỔI - NHÓM TÁI TIẾP CẬN THEO CHU KỲ KINH TẾ VĨ MÔ')
print('='*70)
print(recontact_by_phase[['Tên Giai Đoạn', 'Tỷ lệ %', 'Tổng']].to_string(index=False))

print('\n' + '='*70)
print('TỶ LỆ CHUYỂN ĐỔI - NHÓM LIÊN HỆ LẦN ĐẦU THEO CHU KỲ KINH TẾ VĨ MÔ')
print('='*70)
first_by_phase['Tên Giai Đoạn'] = first_by_phase['Giai đoạn'].map(phase_names)
print(first_by_phase[['Tên Giai Đoạn', 'Tỷ lệ %', 'Tổng']].to_string(index=False))

# Lập bảng so sánh biên độ chênh lệch hiệu quả chiến thuật
comparison = pd.DataFrame({
    'Giai đoạn': recontact_by_phase['Giai đoạn'],
    'Tên Giai Đoạn': recontact_by_phase['Tên Giai Đoạn'],
    'Tái tiếp cận %': recontact_by_phase['Tỷ lệ %'],
    'Lần đầu %': first_by_phase['Tỷ lệ %'],
    'Chênh lệch %': (recontact_by_phase['Tỷ lệ %'] - first_by_phase['Tỷ lệ %']).round(2)
})
print('\n' + '='*70)
print('BẢNG SO SÁNH HIỆU SUẤT ĐỘT PHÁ: TÁI TIẾP CẬN vs LIÊN HỆ LẦN ĐẦU')
print('='*70)
print(comparison[['Tên Giai Đoạn', 'Tái tiếp cận %', 'Lần đầu %', 'Chênh lệch %']].to_string(index=False))


# --- 3. ĐÁNH GIÁ TỶ LỆ CHUYỂN ĐỔI THEO KHOẢNG THỜI GIAN CHỜ (PDAYS) ---
df_pdays = df_recontact[df_recontact['pdays'] != 999].copy()

def segment_pdays(days):
    if days <= 7:
        return '1. Dưới 1 tuần (0-7 ngày)'
    elif days <= 14:
        return '2. Từ 1-2 tuần (8-14 ngày)'
    else:
        return '3. Trên 2 tuần (>14 ngày)'

df_pdays['pdays_group'] = df_pdays['pdays'].apply(segment_pdays)

pdays_rate = df_pdays.groupby('pdays_group')['y'].agg(['sum', 'count', 'mean']).reset_index()
pdays_rate.columns = ['Khoảng thời gian chờ', 'Chuyển đổi', 'Tổng', 'Tỷ lệ']
pdays_rate['Tỷ lệ %'] = (pdays_rate['Tỷ lệ'] * 100).round(2)

print('\n' + '='*70)
print('TỶ LỆ CHUYỂN ĐỔI CHI TIẾT THEO KHOẢNG THỜI GIAN TRÌ HOÃN (pdays)')
print('='*70)
print(pdays_rate[['Khoảng thời gian chờ', 'Tỷ lệ %', 'Tổng']].to_string(index=False))

best_pdays = pdays_rate.loc[pdays_rate['Tỷ lệ %'].idxmax()]
print(f"\n✓ Điểm rơi thời gian tối ưu nhất: {best_pdays['Khoảng thời gian chờ']} đạt tỷ lệ ({best_pdays['Tỷ lệ %']:.2f}%)")


# --- 4. MA TRẬN TƯƠNG TÁC ĐA CHIỀU: GIAI ĐOẠN KINH TẾ × KHOẢNG THỜI GIAN ---
conversion_by_phase_pdays = df_pdays.pivot_table(
    index='economic_phase',
    columns='pdays_group',
    values='y',
    aggfunc='mean'
) * 100
conversion_by_phase_pdays.index = conversion_by_phase_pdays.index.map(phase_names)

print('\n' + '='*85)
print('MA TRẬN XOAY TƯƠNG TÁC: TỶ LỆ CHUYỂN ĐỔI TÁI TIẾP CẬN (%) DIỄN TIẾN THEO BỐI CẢNH')
print('='*85)
print(conversion_by_phase_pdays.round(2).to_string())

best_combination = conversion_by_phase_pdays.stack().idxmax()
best_rate = conversion_by_phase_pdays.stack().max()
print(f"\n✓ Cấu hình phối hợp tối ưu nhất: {best_combination[0]} kết hợp {best_combination[1]} ➔ Tỷ lệ kỳ vọng: {best_rate:.2f}%")


# --- 5. XÂY DỰNG MA TRẬN QUYẾT ĐỊNH TÁI TIẾP CẬN HÀNH ĐỘNG ---
decision_matrix = []
summary_recommendations = []

for phase_code in [1, 2, 3]:
    phase_text = phase_names[phase_code]
    
    recontact_row = recontact_by_phase[recontact_by_phase['Giai đoạn'] == phase_code]
    first_row = first_by_phase[first_by_phase['Giai đoạn'] == phase_code]
    
    recontact_rate = recontact_row['Tỷ lệ %'].values[0] if len(recontact_row) > 0 else 0
    first_rate = first_row['Tỷ lệ %'].values[0] if len(first_row) > 0 else 0
    diff_rate = recontact_rate - first_rate
    
    if phase_text in conversion_by_phase_pdays.index:
        phase_pdays_rates = conversion_by_phase_pdays.loc[phase_text]
        best_pdays_range = phase_pdays_rates.idxmax()
        best_rate_phase = phase_pdays_rates.max()
    else:
        best_pdays_range = 'N/A'
        best_rate_phase = 0
        
    # Logic thiết lập Ma trận Quyết định (Decision Rule)
    if diff_rate > 0 and recontact_rate >= 12:
        recommendation = '✓ CÓ (ƯU TIÊN CHIẾN LƯỢC)'
        reason = f'Ưu thế vượt trội so với gọi mới (+{diff_rate:.2f}%)'
    elif recontact_rate >= 8:
        recommendation = '△ CÓ NHƯNG CHỌN LỌC'
        reason = f'Thị trường áp lực, tập trung vào khung thời gian tối ưu'
    else:
        recommendation = '✗ HẠN CHẾ / TẠM DỪNG'
        reason = 'Tỷ lệ chuyển đổi danh mục quá thấp'
        
    decision_matrix.append({
        'Giai đoạn vĩ mô': phase_text,
        'Tỷ lệ Tái tiếp cận': f"{recontact_rate:.2f}%",
        'Tỷ lệ Gọi lần đầu': f"{first_rate:.2f}%",
        'Biên độ chênh lệch': f"{diff_rate:+.2f}%",
        'Khoảng tối ưu (pdays)': best_pdays_range,
        'Tỷ lệ kỳ vọng': f"{best_rate_phase:.2f}%",
        'Định hướng Hành động': recommendation
    })
    
    summary_recommendations.append({
        'Giai đoạn kinh tế': phase_text,
        'Tỷ lệ chuyển đổi (%)': f"{recontact_rate:.2f}",
        'So với nhóm lần đầu (%)': f"+{diff_rate:.2f}" if diff_rate > 0 else f"{diff_rate:.2f}",
        'Khoảng pdays tối ưu': best_pdays_range,
        'Tỷ lệ kỳ vọng (%)': f"{best_rate_phase:.2f}"
    })

decision_df = pd.DataFrame(decision_matrix)
summary_df = pd.DataFrame(summary_recommendations)

print("\n" + "="*140)
print("BẢNG 3.5: MA TRẬN QUYẾT ĐỊNH TÁI TIẾP CẬN KINH DOANH CHUẨN XÁC")
print("="*140)
print(decision_df.to_string(index=False))
print("="*140)
print('GHI CHÚ: Bảng trên là kết quả đầu ra ứng dụng thực tiễn cao nhất của câu hỏi nghiên cứu RQ3.')


# --- 6. KHUYẾN NGHỊ HÀNH ĐỘNG CHI TIẾT (ACTION PLAN BÁO CÁO) ---
print('\n' + '='*100)
print('BẢN KẾ HOẠCH HÀNH ĐỘNG QUẢN TRỊ CHI TIẾT (ACTION PLAN)')
print('='*100)

tg_data = decision_matrix[0]
st_data = decision_matrix[1]
ph_data = decision_matrix[2]

avg_expected = (float(tg_data['Tỷ lệ kỳ vọng'].replace('%','')) + 
                float(st_data['Tỷ lệ kỳ vọng'].replace('%','')) + 
                float(ph_data['Tỷ lệ kỳ vọng'].replace('%',''))) / 3

print(f"""DỰA TRÊN KẾT QUẢ ĐỊNH LƯỢNG MÔ HÌNH RQ3, ĐỀ XUẤT PHÂN BỔ TÀI NGUYÊN NHƯ SAU:

┌─ GIAI ĐOẠN 1: TĂNG TRƯỞNG (Economic Growth) ────────────────────────────────┐
│ ➜ ĐỊNH HƯỚNG: {tg_data['Định hướng Hành động']}                                 │
│ • Hiệu năng Tái tiếp cận: {tg_data['Tỷ lệ Tái tiếp cận']} | Biên độ lợi thế: {tg_data['Biên độ chênh lệch']}       │
│ • Chiến thuật triển khai: Tập trung kích hoạt danh sách gọi lại chiến dịch cũ. │
│ • Khung thời gian vàng: Thúc đẩy gọi trong {tg_data['Khoảng tối ưu (pdays)']}         │
│ ➔ Tỷ lệ chốt hợp đồng mục tiêu: {tg_data['Tỷ lệ kỳ vọng']}                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ GIAI ĐOẠN 2: SUY THOÁI (Economic Recession) ───────────────────────────────┐
│ ➜ ĐỊNH HƯỚNG: {st_data['Định hướng Hành động']}                                 │
│ • Hiệu năng Tái tiếp cận: {st_data['Tỷ lệ Tái tiếp cận']} | Biên độ lợi thế: {st_data['Biên độ chênh lệch']}       │
│ • Chiến thuật triển khai: Tiết kiệm chi phí vận hành. Chỉ gọi danh sách lọc kỹ.│
│ • Khung thời gian vàng: Bắt buộc tối ưu theo {st_data['Khoảng tối ưu (pdays)']}        │
│ ➔ Tỷ lệ chốt hợp đồng mục tiêu: {st_data['Tỷ lệ kỳ vọng']}                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ GIAI ĐOẠN 3: CHẠM ĐÁY (Economic Trough) ───────────────────────────────────┐
│ ➜ ĐỊNH HƯỚNG: {ph_data['Định hướng Hành động']}                                 │
│ • Hiệu năng Tái tiếp cận: {ph_data['Tỷ lệ Tái tiếp cận']} | Biên độ lợi thế: {ph_data['Biên độ chênh lệch']}       │
│ • Chiến thuật triển khai: ĐÂY LÀ THỜI ĐIỂM SÀNG LỌC. Đổ ngân sách tái khai thác.│
│ • Khung thời gian vàng: Tiếp cận tối ưu trong {ph_data['Khoảng tối ưu (pdays)']}         │
│ ➔ Tỷ lệ chốt hợp đồng mục tiêu: {ph_data['Tỷ lệ kỳ vọng']}                         │
└─────────────────────────────────────────────────────────────────────────────┘

TỔNG KẾT TỶ LỆ CHUYỂN ĐỔI KỲ VỌNG TRUNG BÌNH TOÀN CHIẾN DỊCH: {avg_expected:.2f}%
""")
print('='*100)
print('✅ TOÀN BỘ TIẾN TRÌNH XỬ LÝ RQ3 ĐÃ ĐƯỢC ĐỒNG BỘ THÀNH CÔNG VỚI BƯỚC 2')
print('='*100)


# --- 7. ĐỒ THỊ TRỰC QUAN HÓA KẾT QUẢ ĐỒNG BỘ (VISUALIZATION) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Biểu đồ 1: So sánh Tái tiếp cận vs Gọi lần đầu
ax = axes[0, 0]
x_axis = np.arange(len(comparison))
bar_width = 0.35
ax.bar(x_axis - bar_width/2, comparison['Tái tiếp cận %'], bar_width, label='Tái tiếp cận', color='#3498db', alpha=0.85, edgecolor='black')
ax.bar(x_axis + bar_width/2, comparison['Lần đầu %'], bar_width, label='Gọi lần đầu', color='#e74c3c', alpha=0.85, edgecolor='black')
ax.set_xlabel('Phân cụm chu kỳ kinh tế vĩ mô', fontsize=10, fontweight='bold')
ax.set_ylabel('Tỷ lệ chuyển đổi thành công (%)', fontsize=10, fontweight='bold')
ax.set_title('Hiệu suất chuyển đổi danh mục qua các Giai đoạn', fontsize=11, fontweight='bold')
ax.set_xticks(x_axis)
ax.set_xticklabels(['Giai đoạn 1\n(Tăng trưởng)', 'Giai đoạn 2\n(Suy thoái)', 'Giai đoạn 3\n(Chạm đáy)'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Biểu đồ 2: Phân phối hiệu năng theo pdays
ax = axes[0, 1]
colors_pdays = ['#2ecc71', '#f1c40f', '#e67e22']
ax.bar(range(len(pdays_rate)), pdays_rate['Tỷ lệ %'], color=colors_pdays, alpha=0.85, edgecolor='black')
ax.set_ylabel('Tỷ lệ chuyển đổi (%)', fontsize=10, fontweight='bold')
ax.set_title('Tỷ lệ Chuyển đổi Khách hàng theo Khoảng Thời gian Chờ (pdays)', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(pdays_rate)))
ax.set_xticklabels([lbl.replace(' (', '\n(') for lbl in pdays_rate['Khoảng thời gian chờ']], fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Biểu đồ 3: Heatmap ma trận tương tác Giai đoạn × pdays
ax = axes[1, 0]
sns.heatmap(conversion_by_phase_pdays, annot=True, fmt='.2f', cmap='YlGnBu', 
            cbar_kws={'label': 'Tỷ lệ chuyển đổi mục tiêu (%)'}, ax=ax, linewidths=1.5, linecolor='white')
ax.set_title('Bản đồ nhiệt Tương tác: Giai đoạn vĩ mô × Thời gian chờ', fontsize=11, fontweight='bold')
ax.set_xlabel('Khoảng thời gian trì hoãn gọi lại (pdays)', fontsize=10, fontweight='bold')
ax.set_ylabel('Bối cảnh cấu trúc vĩ mô', fontsize=10, fontweight='bold')

# Biểu đồ 4: Biên độ lợi thế
ax = axes[1, 1]
colors_diff = ['#27ae60' if v > 0 else '#c0392b' for v in comparison['Chênh lệch %']]
ax.bar(range(len(comparison)), comparison['Chênh lệch %'], color=colors_diff, alpha=0.85, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
ax.set_ylabel('Biên độ ưu thế vượt trội (%)', fontsize=10, fontweight='bold')
ax.set_title('Biên độ chênh lệch: Tái tiếp cận so với Nhóm gọi mới', fontsize=11, fontweight='bold')
ax.set_xticks(range(len(comparison)))
ax.set_xticklabels(['Giai đoạn 1\n(Tăng trưởng)', 'Giai đoạn 2\n(Suy thoái)', 'Giai đoạn 3\n(Chạm đáy)'])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print('✓ Toàn bộ hệ thống bảng biểu trực quan hóa báo cáo khoa học đã xuất bản thành công!')
