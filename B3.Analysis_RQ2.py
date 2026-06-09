import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
print("==========================================================")
print("     TIẾN HÀNH RQ2: PHÂN TÍCH CHIẾN DỊCH TELEMARKETING")
print("==========================================================")

# 1. ĐỌC DỮ LIỆU ĐÃ MÃ HÓA (Kế thừa từ file Data_bank_encoded.csv sạch)
input_file = '.\Data_bank_encoded.csv'

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Không tìm thấy file '{input_file}'. Vui lòng chạy file Data_preparation.py trước!")

df = pd.read_csv(input_file)
print(f"-> Tải dữ liệu mã hóa thành công. Quy mô phân tích: {df.shape[0]} dòng, {df.shape[1]} cột.")


# =========================================================================
# BƯỚC 2.1 · TỶ LỆ CHUYỂN ĐỔI BASELINE THEO GIAI ĐOẠN KINH TẾ
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 2.1: TỶ LỆ CHUYỂN ĐỔI BASELINE THEO GIAI ĐOẠN")
print("="*50)

# Đảm bảo các cột định dạng số nguyên chuẩn để không bị lỗi tính toán
df['economic_phase'] = df['economic_phase'].astype(int)
df['y'] = df['y'].astype(int)

# Tạo từ điển tên tiếng Việt phục vụ hiển thị in ấn kết quả cho đẹp
phase_names = {1: '1. Tăng trưởng', 2: '2. Suy thoái', 3: '3. Chạm đáy'}

# Tính tỷ lệ chuyển đổi baseline (giá trị trung bình của y) theo giai đoạn
conversion_baseline = df.groupby('economic_phase')['y'].mean()
counts_baseline = df.groupby('economic_phase')['y'].agg(Số_đơn_chốt='sum', Tổng_cuộc_gọi='count')

print("Tỷ lệ chuyển đổi Baseline thực tế từng giai đoạn:")
for phase_code, rate in conversion_baseline.items():
    print(f" - {phase_names[phase_code]}: {rate*100:.2f}%")

print("\nBảng tổng hợp số lượng mẫu:")
counts_baseline.index = counts_baseline.index.map(phase_names)
print(counts_baseline)

# Tạo bảng liên hợp (Contingency Table) phục vụ kiểm định Chi-square toàn diện
contingency_baseline = pd.crosstab(df['economic_phase'], df['y'])
contingency_baseline.index = contingency_baseline.index.map(phase_names)
contingency_baseline.columns = ['Từ chối (y=0)', 'Chốt đơn (y=1)']

print('\nBảng liên hợp thực tế phục vụ kiểm định Chi-square:')
print(contingency_baseline)

# Thực hiện kiểm định Chi-square toàn cục
chi2, p_val, dof, expected = chi2_contingency(contingency_baseline)
print(f'\nKết quả kiểm định Chi-square toàn cục:')
print(f' - Hệ số Chi2: {chi2:.4f}')
print(f' - Hệ số p-value: {p_val:.4e} (tức là: {p_val:.4f})')
print(f' - Bậc tự do (dof): {dof}')

# Biện luận nhanh kết quả Bước 2.1
if p_val < 0.05:
    print("=> KẾT LUẬN: Bác bỏ H0. Sự khác biệt tỷ lệ chuyển đổi giữa các giai đoạn kinh tế CÓ ý nghĩa thống kê.")
    print("   Bối cảnh vĩ mô thực sự ảnh hưởng mạnh mẽ tới xác suất chốt đơn của khách hàng.")
else:
    print("=> KẾT LUẬN: Chưa có cơ sở bác bỏ H0 (Không có sự khác biệt mang ý nghĩa thống kê).")


# =========================================================================
# BƯỚC 2.2 · PHÂN TÍCH KÊNH TIẾP CẬN (CONTACT) QUA TỪNG GIAI ĐOẠN
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 2.2: PHÂN TÍCH KÊNH TIẾP CẬN (CONTACT) 3 CHIỀU")
print("="*50)

# Đảm bảo cột contact là số nguyên (1: Cellular, 0: Telephone)
df['contact'] = df['contact'].astype(int)
contact_names = {1: 'Di động (Cellular)', 0: 'Điện thoại bàn (Telephone)'}

# Lặp qua từng giai đoạn kinh tế để phân tích sâu nội bộ (Crosstab 3 chiều tách biệt)
for phase_code in [1, 2, 3]:
    print(f"\n▶ TRỰC QUAN HÓA CHO GIAI ĐOẠN: {phase_names[phase_code].upper()}")
    
    # Lọc dữ liệu riêng cho giai đoạn hiện tại
    df_phase = df[df['economic_phase'] == phase_code]
    
    # Tính tỷ lệ chuyển đổi cụ thể theo từng kênh trong giai đoạn này
    contact_conversion = df_phase.groupby('contact')['y'].mean()
    contact_counts = df_phase.groupby('contact')['y'].agg(Chốt_đơn='sum', Tổng_gọi='count')
    
    print(f" Tỷ lệ chuyển đổi theo kênh:")
    for contact_code, rate in contact_conversion.items():
        print(f"   + {contact_names[contact_code]}: {rate*100:.2f}% (Chốt {contact_counts.loc[contact_code, 'Chốt_đơn']}/{contact_counts.loc[contact_code, 'Tổng_gọi']} cuộc gọi)")
        
    # Tạo bảng liên hợp cục bộ giữa kênh tiếp cận (contact) và biến mục tiêu (y)
    contingency_contact = pd.crosstab(df_phase['contact'], df_phase['y'])
    contingency_contact.index = contingency_contact.index.map(contact_names)
    contingency_contact.columns = ['Từ chối (y=0)', 'Chốt đơn (y=1)']
    
    # Thực hiện kiểm định Chi-square cho riêng giai đoạn này
    chi2_c, p_val_c, dof_c, expected_c = chi2_contingency(contingency_contact)
    print(f" Kết quả kiểm định Chi-square nội bộ Phase {phase_code}:")
    print(f"   + Chi2 = {chi2_c:.4f} | p-value = {p_val_c:.4e}")
    
    # Kết luận nhanh cho từng giai đoạn
    if p_val_c < 0.05:
        print(f"   => KẾT LUẬN: Trong giai đoạn này, sự chênh lệch hiệu quả giữa Di động và Điện thoại bàn CÓ ý nghĩa thống kê rõ rệt.")
    else:
        print(f"   => KẾT LUẬN: Không có sự khác biệt mang ý nghĩa thống kê giữa 2 kênh tiếp cận trong giai đoạn này.")



# Cấu hình hiển thị tiếng Việt font cơ bản cho matplotlib nếu cần
plt.rcParams['axes.unicode_minus'] = False

# Định nghĩa lại bộ từ điển nhãn hiển thị từ file trước để đồng bộ hệ thống in ấn
phase_names = {1: '1. Tăng trưởng', 2: '2. Suy thoái', 3: '3. Chạm đáy'}
month_labels = {3: 'Th3', 4: 'Th4', 5: 'Th5', 6: 'Th6', 7: 'Th7', 8: 'Th8', 9: 'Th9', 10: 'Th10', 11: 'Th11', 12: 'Th12'}
day_labels = {1: 'Thứ 2', 2: 'Thứ 3', 3: 'Thứ 4', 4: 'Thứ 5', 5: 'Thứ 6'}
duration_labels = {1: 'Dưới 2 phút', 2: '2-5 phút', 3: '5-10 phút', 4: 'Trên 10 phút'}

# =========================================================================
# BƯỚC 2.3 · PHÂN TÍCH THỜI ĐIỂM TIẾP CẬN (MONTH & DAY_OF_WEEK)
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 2.3: PHÂN TÍCH THỜI ĐIỂM TIẾP CẬN (HEATMAP)")
print("="*50)

# Tạo bảng pivot tính tỷ lệ chuyển đổi (mean của y) cho Tháng và Thứ
pivot_month = df.pivot_table(index='economic_phase', columns='month', values='y', aggfunc='mean', observed=False)
pivot_day = df.pivot_table(index='economic_phase', columns='day_of_week', values='y', aggfunc='mean', observed=False)

# Ánh xạ nhãn tiếng Việt cho các trục tọa độ của bảng Pivot trước khi vẽ biểu đồ
pivot_month.index = pivot_month.index.map(phase_names)
pivot_month.columns = pivot_month.columns.map(month_labels)
pivot_day.index = pivot_day.index.map(phase_names)
pivot_day.columns = pivot_day.columns.map(day_labels)

# Vẽ biểu đồ Heatmap đôi
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(pivot_month, annot=True, fmt=".2%", cmap="YlGnBu", cbar=True, linewidths=0.5, ax=axes[0])
axes[0].set_title("Tỷ lệ Chuyển đổi theo Tháng & Giai đoạn Kinh tế", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Giai đoạn Kinh tế", fontsize=10)
axes[0].set_xlabel("Tháng tiếp cận", fontsize=10)

sns.heatmap(pivot_day, annot=True, fmt=".2%", cmap="Oranges", cbar=True, linewidths=0.5, ax=axes[1])
axes[1].set_title("Tỷ lệ Chuyển đổi theo Thứ & Giai đoạn Kinh tế", fontsize=12, fontweight='bold')
axes[1].set_ylabel("")
axes[1].set_xlabel("Thứ trong tuần", fontsize=10)

plt.tight_layout()
plt.show()

# Kiểm định Kruskal-Wallis cho Tháng và Thứ qua từng giai đoạn
for phase_code in [1, 2, 3]:
    df_phase = df[df['economic_phase'] == phase_code]
    print(f"\n▶ GIAI ĐOẠN: {phase_names[phase_code].upper()}")
    
    # 1. Kiểm định Kruskal-Wallis cho biến Month
    month_groups = [group['y'].values for name, group in df_phase.groupby('month')]
    month_groups = [g for g in month_groups if len(g) > 0]
    if len(month_groups) > 1:
        stat_m, p_val_m = kruskal(*month_groups)
        print(f" - Kiểm định Tháng: H-stat = {stat_m:.4f} | p-value = {p_val_m:.4e}")
        print(f"   => Ý nghĩa thống kê theo tháng? {'CÓ (p < 0.05)' if p_val_m < 0.05 else 'KHÔNG'}")
        
    # 2. Kiểm định Kruskal-Wallis cho biến Day_of_week
    day_groups = [group['y'].values for name, group in df_phase.groupby('day_of_week')]
    day_groups = [g for g in day_groups if len(g) > 0]
    if len(day_groups) > 1:
        stat_d, p_val_d = kruskal(*day_groups)
        print(f" - Kiểm định Thứ:  H-stat = {stat_d:.4f} | p-value = {p_val_d:.4e}")
        print(f"   => Ý nghĩa thống kê theo thứ?  {'CÓ (p < 0.05)' if p_val_d < 0.05 else 'KHÔNG'}")


# =========================================================================
# BƯỚC 2.4 · PHÂN TÍCH SỐ LẦN GỌI (CAMPAIGN) VÀ ĐƯỜNG CONG CHIẾN DỊCH
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 2.4: PHÂN TÍCH ĐƯỜNG CONG TỶ LỆ THEO SỐ LẦN GỌI")
print("="*50)

# 1. Gom nhóm tính tỷ lệ chuyển đổi trực tiếp theo campaign_group sạch từ file csv
campaign_pivot = df.groupby(['economic_phase', 'campaign_group'], observed=False)['y'].mean().reset_index()
campaign_pivot['economic_phase_name'] = campaign_pivot['economic_phase'].map(phase_names)

# 2. Thiết lập thứ tự trục X cố định để biểu đồ chạy mượt mà từ 1 đến 5+ (tránh lỗi sắp xếp theo chữ)
campaign_order = ['1', '2', '3', '4', '5+']
campaign_pivot['campaign_group'] = pd.Categorical(campaign_pivot['campaign_group'], categories=campaign_order, ordered=True)

# 3. Trực quan hóa đường cong chiến dịch
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=campaign_pivot, 
    x='campaign_group',           # Sử dụng trực tiếp trường chứa dữ liệu '5+' đồng bộ
    y='y', 
    hue='economic_phase_name', 
    marker='o', 
    linewidth=2.5,
    markersize=8
)

plt.title("Đường cong Tỷ lệ Chuyển đổi theo Số lần gọi qua các Giai đoạn", fontsize=12, fontweight='bold')
plt.xlabel("Số lần tiếp cận trong chiến dịch hiện tại (Campaign Group)", fontsize=10)
plt.ylabel("Tỷ lệ Chuyển đổi (Conversion Rate)", fontsize=10)

# Định dạng trục Y hiển thị theo % cho trực quan
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Bối cảnh vĩ mô", fontsize=9)
plt.tight_layout()

# Bạn có thể chọn 1 trong 2 dòng dưới đây tùy nhu cầu:
plt.show()  # Xem trực tiếp popup (Nhớ bấm dấu X tắt cửa sổ để code chạy tiếp bước 2.5)
# plt.savefig('.\duong_cong_campaign.png', dpi=300); plt.close() # Hoặc tự lưu thành file ảnh tránh đứng máy

# 4. Kiểm định phi tham số Mann-Whitney U so sánh phân bổ chiến dịch gốc (chưa nén) giữa nhóm Yes và No
print("\nKết quả kiểm định Mann-Whitney U cho biến Campaign gốc:")
for phase_code in sorted(df['economic_phase'].unique()):
    df_phase = df[df['economic_phase'] == phase_code]
    campaign_yes = df_phase[df_phase['y'] == 1]['campaign'].values
    campaign_no = df_phase[df_phase['y'] == 0]['campaign'].values
    
    if len(campaign_yes) > 0 and len(campaign_no) > 0:
        stat, p_val = mannwhitneyu(campaign_yes, campaign_no, alternative='two-sided')
        print(f" - Giai đoạn [{phase_names.get(phase_code, str(phase_code))}]:")
        print(f"   + U-statistic: {stat:.2f} | p-value = {p_val:.4e}")
        print(f"   => Ý nghĩa thống kê? {'CÓ sự tác động rõ rệt (p < 0.05)' if p_val < 0.05 else 'KHÔNG có tác động ý nghĩa'}\n")


# =========================================================================
# BƯỚC 2.5 · PHÂN TÍCH THỜI LƯỢNG TƯ VẤN (DURATION) - BIẾN POST-HOC
# =========================================================================
print("\n" + "="*50)
print(" BƯỚC 2.5: PHÂN TÍCH THỜI LƯỢNG TƯ VẤN (POST-HOC)")
print("="*50)

# Tạo bảng tỷ lệ chuyển đổi dựa trên trường duration_group đã chia nhóm từ bước prep
pivot_duration = df.pivot_table(index='economic_phase', columns='duration_group', values='y', aggfunc='mean', observed=False)
pivot_duration.index = pivot_duration.index.map(phase_names)
pivot_duration.columns = pivot_duration.columns.map(duration_labels)

# Trực quan hóa Heatmap phân khúc thời lượng cuộc gọi
plt.figure(figsize=(10, 5))
sns.heatmap(pivot_duration, annot=True, fmt=".2%", cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Tỷ lệ Chuyển đổi theo Phân khúc Thời lượng & Chu kỳ Vĩ mô", fontsize=12, fontweight='bold')
plt.ylabel("Giai đoạn Kinh tế", fontsize=10)
plt.xlabel("Phân khúc thời lượng tư vấn cuộc gọi (Post-hoc Variable)", fontsize=10)
plt.tight_layout()
plt.show()

# Hàm tính kích thước hiệu ứng Cohen's d để đo cường độ ảnh hưởng thực tế
def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0: return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_se if pooled_se != 0 else 0

# Thực hiện kiểm định Kruskal-Wallis so sánh thời lượng trung bình và tính toán Cohen's d
print("\nKiểm định Kruskal-Wallis & Đo lường kích thước hiệu ứng (Cohen's d):")
for phase_code in [1, 2, 3]:
    df_phase = df[df['economic_phase'] == phase_code]
    duration_yes = df_phase[df_phase['y'] == 1]['duration'].values
    duration_no = df_phase[df_phase['y'] == 0]['duration'].values
    
    if len(duration_yes) > 0 and len(duration_no) > 0:
        stat, p_val = kruskal(duration_yes, duration_no)
        d_val = calculate_cohens_d(duration_yes, duration_no)
        
        # Đánh giá độ mạnh của Cohen's d theo chuẩn học thuật
        if abs(d_val) < 0.2: effect_desc = "Rất nhỏ (Negligible)"
        elif abs(d_val) < 0.5: effect_desc = "Nhỏ (Small)"
        elif abs(d_val) < 0.8: effect_desc = "Trung bình (Medium)"
        else: effect_desc = "Lớn (Large)"
        
        print(f" - Giai đoạn [{phase_names[phase_code]}]:")
        print(f"   + Kruskal H-stat: {stat:.2f} | p-value = {p_val:.4e}")
        print(f"   + Kích thước hiệu ứng Cohen's d: {d_val:.4f} -> Cường độ ảnh hưởng: {effect_desc}")
        print(f"   => Ý nghĩa thống kê? {'CÓ' if p_val < 0.05 else 'KHÔNG'}\n")

print("\n==========================================================")
print(" 🎉 HOÀN THÀNH TOÀN BỘ PHÂN TÍCH RQ2 (TỪ BƯỚC 2.1 ĐẾN 2.5)")
print("==========================================================")


# =========================================================================
# BƯỚC 2.6 ·CÂY QUYẾT ĐỊNH
# =========================================================================


print("\n" + "="*80)
print("  BƯỚC 2.6: DECISION TREE THEO TỪNG GIAI ĐOẠN KINH TẾ (ĐỒNG BỘ ĐẦY ĐỦ)")
print("="*80)

# ============================================================
# LOAD DỮ LIỆU ĐÃ MÃ HÓA ĐỒNG BỘ TỪ FILE CSV SẠCH
# ============================================================
input_file = '.\Data_bank_encoded.csv'
if not os.path.exists(input_file):
    # Nếu chưa chạy prep, dự phòng đọc file kết quả tổng cục
    input_file = '.\Data_bank_final_results.csv'

df_source = pd.read_csv(input_file)

# Bản sao xử lý mô hình hóa
df_encoded = df_source.copy()

# Đồng bộ loại bỏ các cuộc gọi nhỡ (duration = 0) để giảm nhiễu mô hình
if 'duration' in df_encoded.columns:
    df_encoded = df_encoded[df_encoded['duration'] > 0].reset_index(drop=True)

# ============================================================
# ÉP ĐỒNG BỘ CÁC BIẾN SANG ĐỊNH DẠNG SỐ HỌC (ORDINAL ENCODING)
# ============================================================

# Biến mục tiêu y (yes -> 1, no -> 0)
if df_encoded['y'].dtype == 'object':
    df_encoded['y_enc'] = (df_encoded['y'].str.strip().str.lower() == 'yes').astype(int)
else:
    df_encoded['y_enc'] = df_encoded['y'].astype(int)

# contact (cellular -> 1, telephone -> 0)
if df_encoded['contact'].dtype == 'object':
    df_encoded['contact_enc'] = (df_encoded['contact'].str.strip().str.lower() == 'cellular').astype(int)
else:
    df_encoded['contact_enc'] = df_encoded['contact'].astype(int)

# month — Đồng bộ theo thứ tự thời gian tuyến tính
month_order = {
    'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12,
    'Th3':3, 'Th4':4, 'Th5':5, 'Th6':6, 'Th7':7, 'Th8':8, 'Th9':9, 'Th10':10, 'Th11':11, 'Th12':12
}
if df_encoded['month'].dtype == 'object':
    df_encoded['month_enc'] = df_encoded['month'].map(month_order).fillna(5) # Mặc định tháng 5 nếu lệch nhãn
else:
    df_encoded['month_enc'] = df_encoded['month'].astype(int)

# day_of_week — Đồng bộ thứ tự các ngày trong tuần
day_order = {
    'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5,
    'Thứ 2':1, 'Thứ 3':2, 'Thứ 4':3, 'Thứ 5':4, 'Thứ 6':5
}
if df_encoded['day_of_week'].dtype == 'object':
    df_encoded['day_enc'] = df_encoded['day_of_week'].map(day_order).fillna(1)
else:
    df_encoded['day_enc'] = df_encoded['day_of_week'].astype(int)

# Chuyển đổi chiến dịch campaign_group '5+' dạng chuỗi (nếu có) sang số thứ tự để thuật toán phân nhánh
if df_encoded['campaign_group'].dtype == 'object' or isinstance(df_encoded['campaign_group'].dtype, pd.CategoricalDtype):
    df_encoded['campaign_num_group'] = df_encoded['campaign_group'].map({'1':1, '2':2, '3':3, '4':4, '5+':5}).fillna(5).astype(int)
else:
    df_encoded['campaign_num_group'] = df_encoded['campaign_group'].astype(int)

# Bảo toàn định dạng mã hóa số của duration_group
df_encoded['duration_num_group'] = df_encoded['duration_group'].astype(int)

# ============================================================
# ĐỊNH NGHĨA DANH SÁCH FEATURES VÀ ĐỒNG BỘ TÊN GIAI ĐOẠN
# ============================================================
features       = ['contact_enc', 'month_enc', 'day_enc', 'campaign_num_group', 'duration_num_group', 'previous']
feature_labels = ['Kênh liên lạc', 'Tháng tiếp cận', 'Ngày trong tuần', 'Tần suất gọi (nhóm)', 'Thời lượng gọi (nhóm)', 'Lịch sử tiếp cận trước']
target         = 'y_enc'

# Từ điển ánh xạ nhãn hiển thị trực quan quy tắc cây
feature_mapping_rules = dict(zip(features, feature_labels))

# Chuẩn hóa danh sách 3 giai đoạn kinh tế viết hoa đúng lõi tệp dữ liệu CSV của nhóm
phases = ['Giai đoạn Tăng trưởng', 'Giai đoạn Suy thoái', 'Giai đoạn Chạm đáy']

# Kiểm tra thực tế cột economic_phase để tránh lỗi rỗng bảng
if 'economic_phase' in df_encoded.columns:
    # Nếu cột đang lưu dạng số (1, 2, 3), ta map về chuỗi để đồng bộ vòng lặp
    if df_encoded['economic_phase'].dtype in [np.int64, np.int32, int]:
        phase_names_map = {1: 'Giai đoạn Tăng trưởng', 2: 'Giai đoạn Suy thoái', 3: 'Giai đoạn Chạm đáy'}
        df_encoded['economic_phase_str'] = df_encoded['economic_phase'].map(phase_names_map)
    else:
        df_encoded['economic_phase_str'] = df_encoded['economic_phase'].str.strip()
else:
    raise KeyError("Không tìm thấy trường dữ liệu bối cảnh 'economic_phase' trong file.")

# ============================================================
# VÒNG LẶP CHÍNH HUẤN LUYỆN 3 CÂY CHO 3 GIAI ĐOẠN
# ============================================================
results_summary = []

for phase in phases:
    sep = "*" * 75
    print(f"\n{sep}")
    print(f"  🌲 MÔ HÌNH CÂY QUYẾT ĐỊNH — {phase.upper()}")
    print(f"{sep}")

    # Lọc dữ liệu chuẩn xác theo chuỗi giai đoạn tương ứng
    df_phase = df_encoded[df_encoded['economic_phase_str'] == phase].copy()

    X = df_phase[features].values
    y = df_phase[target].values

    n_total = len(y)
    n_yes   = int(y.sum())
    n_no    = n_total - n_yes
    
    print(f"  [THÔNG SỐ MẪU THỰC TẾ]:")
    print(f"   + Tổng số cuộc gọi phân tích : {n_total:,}")
    print(f"   + Chốt đơn thành công (yes)  : {n_yes:,} ({n_yes/n_total:.2%})")
    print(f"   + Từ chối dịch vụ (no)       : {n_no:,} ({n_no/n_total:.2%})")

    if n_yes < 10:
        print("  => [BỎ QUA GIAI ĐOẠN] Số mẫu thành công quá nhỏ để chạy Cross-Validation học thuật.")
        continue

    # --- Đánh giá chéo Stratified 5-Fold K-Fold ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores_cv, auc_scores_cv = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = DecisionTreeClassifier(
            max_depth=4,
            class_weight='balanced',  # Tự động bù trừ trọng số cứu tập dữ liệu lệch nặng (~89% no)
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        f1_scores_cv.append(f1_score(y_test, y_pred, zero_division=0))
        auc_scores_cv.append(roc_auc_score(y_test, y_proba))

    mean_f1  = np.mean(f1_scores_cv)
    mean_auc = np.mean(auc_scores_cv)

    print(f"\n  [KẾT QUẢ KIỂM ĐỊNH ĐÁNH GIÁ CHÉO 5-FOLD CHỐNG QUÁ KHỚP]:")
    print(f"   - F1-Score trung bình kiểm định : {mean_f1:.4f}")
    print(f"   - AUC-ROC trung bình tổng quát  : {mean_auc:.4f}")

    # --- Huấn luyện cây quyết định chính thức cho toàn bộ giai đoạn ---
    final_tree = DecisionTreeClassifier(
        max_depth=4,
        class_weight='balanced',
        random_state=42
    )
    final_tree.fit(X, y)

    # Trích xuất tầm quan trọng của các đặc trưng (Feature Importance)
    print(f"\n  [ĐỘ QUAN TRỌNG CỦA CÁC BIẾN TÁC ĐỘNG]:")
    importances = final_tree.feature_importances_
    importance_pairs = sorted(
        zip(feature_labels, importances),
        key=lambda x: x[1],
        reverse=True
    )
    for label, imp in importance_pairs:
        bar = "█" * int(imp * 40)
        print(f"   + {label:<25}: {imp:.4f} ({imp*100:.1f}%) {bar}")

    # Trích xuất quy tắc If-Then của cây quyết định
    print(f"\n  [QUY TẮC PHÂN NHÁNH THỰC TIỄN - IF-THEN RULES]:")
    print("-" * 60)
    # Ghi chú học thuật: export_text bắt buộc nhận ký tự không dấu hoặc mã biến gốc để tránh lỗi font thư viện
    rules = export_text(final_tree, feature_names=features)
    print(rules)
    print("  * Ghi chú đọc điều kiện phân nhánh:")
    for key, val in feature_mapping_rules.items():
        print(f"    - {key} đại diện cho biến: [{val}]")
    print("-" * 60)

    results_summary.append({
        'Bối cảnh Vĩ mô': phase,
        'Tổng cuộc gọi': f"{n_total:,}",
        'Tỷ lệ Conversion': f"{n_yes/n_total:.2%}",
        'F1-Score': f"{mean_f1:.4f}",
        'AUC-ROC': f"{mean_auc:.4f}",
        'Biến tác động số 1': importance_pairs[0][0],
        'Biến tác động số 2': importance_pairs[1][0]
    })

# ============================================================
# BƯỚC 5: XUẤT BẢNG TÓM TẮT MA TRẬN SO SÁNH CUỐI CÙNG
# ============================================================
print("\n" + "="*85)
print("  BẢNG MA TRẬN TỔNG HỢP SO SÁNH ĐA GIAI ĐOẠN ĐỂ ĐƯA VÀO BÁO CÁO KẾT LUẬN")
print("="*85)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

print("\n 🎉 HOÀN THÀNH TÍCH HỢP ĐỒNG BỘ TOÀN DIỆN BƯỚC 2.6 CHO NHÓM!")
print("="*85)