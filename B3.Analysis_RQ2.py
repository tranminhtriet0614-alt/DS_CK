import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

# =========================================================================
# CẤU HÌNH HỆ THỐNG
# =========================================================================
plt.rcParams['axes.unicode_minus'] = False

INPUT_FILE = './Data_bank_encoded.csv'

# Bộ từ điển nhãn hiển thị dùng chung toàn file
PHASE_NAMES    = {1: '1. Tăng trưởng', 2: '2. Suy thoái', 3: '3. Chạm đáy'}
CONTACT_NAMES  = {1: 'Di động (Cellular)', 0: 'Điện thoại bàn (Telephone)'}
MONTH_LABELS   = {3:'Th3', 4:'Th4', 5:'Th5', 6:'Th6', 7:'Th7', 8:'Th8',
                  9:'Th9', 10:'Th10', 11:'Th11', 12:'Th12'}
DAY_LABELS     = {1:'Thứ 2', 2:'Thứ 3', 3:'Thứ 4', 4:'Thứ 5', 5:'Thứ 6'}
DURATION_LABELS = {1:'Dưới 2 phút', 2:'2-5 phút', 3:'5-10 phút', 4:'Trên 10 phút'}

print("==========================================================")
print("     TIẾN HÀNH RQ2: PHÂN TÍCH CHIẾN DỊCH TELEMARKETING")
print("==========================================================")

# =========================================================================
# ĐỌC DỮ LIỆU
# =========================================================================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Không tìm thấy file '{INPUT_FILE}'. Vui lòng chạy Data_preparation.py trước!")

df = pd.read_csv(INPUT_FILE)
print(f"-> Tải dữ liệu mã hóa thành công. Quy mô: {df.shape[0]:,} dòng, {df.shape[1]} cột.")

df['economic_phase'] = df['economic_phase'].astype(int)
df['y']              = df['y'].astype(int)
df['contact']        = df['contact'].astype(int)

# =========================================================================
# BƯỚC 2.1: TỶ LỆ CHUYỂN ĐỔI BASELINE THEO GIAI ĐOẠN KINH TẾ
# =========================================================================
print("\n" + "=" * 50)
print(" BƯỚC 2.1: TỶ LỆ CHUYỂN ĐỔI BASELINE THEO GIAI ĐOẠN")
print("=" * 50)

conversion_baseline = df.groupby('economic_phase')['y'].mean()
counts_baseline     = df.groupby('economic_phase')['y'].agg(Số_đơn_chốt='sum', Tổng_cuộc_gọi='count')

print("Tỷ lệ chuyển đổi Baseline thực tế từng giai đoạn:")
for phase_code, rate in conversion_baseline.items():
    print(f" - {PHASE_NAMES[phase_code]}: {rate*100:.2f}%")

print("\nBảng tổng hợp số lượng mẫu:")
counts_baseline.index = counts_baseline.index.map(PHASE_NAMES)
print(counts_baseline)

contingency_baseline = pd.crosstab(df['economic_phase'], df['y'])
contingency_baseline.index   = contingency_baseline.index.map(PHASE_NAMES)
contingency_baseline.columns = ['Từ chối (y=0)', 'Chốt đơn (y=1)']
print('\nBảng liên hợp thực tế phục vụ kiểm định Chi-square:')
print(contingency_baseline)

chi2, p_val, dof, _ = chi2_contingency(contingency_baseline)
print(f'\nKết quả kiểm định Chi-square toàn cục:')
print(f' - Hệ số Chi2 : {chi2:.4f}')
print(f' - p-value    : {p_val:.4e}')
print(f' - Bậc tự do  : {dof}')

if p_val < 0.05:
    print("=> KẾT LUẬN: Bác bỏ H0. Sự khác biệt tỷ lệ chuyển đổi giữa các giai đoạn CÓ ý nghĩa thống kê.")
    print("   Bối cảnh vĩ mô thực sự ảnh hưởng mạnh tới xác suất chốt đơn của khách hàng.")
else:
    print("=> KẾT LUẬN: Chưa có cơ sở bác bỏ H0 (Không có sự khác biệt mang ý nghĩa thống kê).")

# =========================================================================
# BƯỚC 2.2: PHÂN TÍCH KÊNH TIẾP CẬN (CONTACT) QUA TỪNG GIAI ĐOẠN
# =========================================================================
print("\n" + "=" * 50)
print(" BƯỚC 2.2: PHÂN TÍCH KÊNH TIẾP CẬN (CONTACT) 3 CHIỀU")
print("=" * 50)

for phase_code in [1, 2, 3]:
    print(f"\n▶ TRỰC QUAN HÓA CHO GIAI ĐOẠN: {PHASE_NAMES[phase_code].upper()}")

    df_phase           = df[df['economic_phase'] == phase_code]
    contact_conversion = df_phase.groupby('contact')['y'].mean()
    contact_counts     = df_phase.groupby('contact')['y'].agg(Chốt_đơn='sum', Tổng_gọi='count')

    print(" Tỷ lệ chuyển đổi theo kênh:")
    for contact_code, rate in contact_conversion.items():
        print(f"   + {CONTACT_NAMES[contact_code]}: {rate*100:.2f}%"
              f" (Chốt {contact_counts.loc[contact_code, 'Chốt_đơn']}/{contact_counts.loc[contact_code, 'Tổng_gọi']} cuộc gọi)")

    contingency_contact = pd.crosstab(df_phase['contact'], df_phase['y'])
    contingency_contact.index   = contingency_contact.index.map(CONTACT_NAMES)
    contingency_contact.columns = ['Từ chối (y=0)', 'Chốt đơn (y=1)']

    chi2_c, p_val_c, _, _ = chi2_contingency(contingency_contact)
    print(f" Kết quả kiểm định Chi-square nội bộ Phase {phase_code}:")
    print(f"   + Chi2 = {chi2_c:.4f} | p-value = {p_val_c:.4e}")

    if p_val_c < 0.05:
        print("   => KẾT LUẬN: Sự chênh lệch hiệu quả giữa Di động và Điện thoại bàn CÓ ý nghĩa thống kê.")
    else:
        print("   => KẾT LUẬN: Không có sự khác biệt mang ý nghĩa thống kê giữa 2 kênh trong giai đoạn này.")

    contact_plot = contact_conversion.reset_index()
    contact_plot.columns = ['Kênh tiếp cận', 'Tỷ lệ chuyển đổi']
    contact_plot['Kênh tiếp cận'] = contact_plot['Kênh tiếp cận'].map(CONTACT_NAMES)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=contact_plot, x='Kênh tiếp cận', y='Tỷ lệ chuyển đổi',
                hue='Kênh tiếp cận', palette='Set2', legend=False)
    plt.title(f'Tỷ lệ chuyển đổi theo kênh tiếp cận — {PHASE_NAMES[phase_code]}',
              fontsize=11, fontweight='bold')
    plt.ylabel('Tỷ lệ chuyển đổi', fontsize=10)
    plt.xlabel('Kênh tiếp cận', fontsize=10)
    plt.ylim(0, max(contact_plot['Tỷ lệ chuyển đổi'].max() * 1.15, 0.05))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    for i, v in enumerate(contact_plot['Tỷ lệ chuyển đổi']):
        plt.text(i, v + 0.005, f'{v:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'./RQ2_contact_phase_{phase_code}.png', dpi=300)
    plt.close()
    print(f"   -> Đã xuất biểu đồ: './RQ2_contact_phase_{phase_code}.png'")

# =========================================================================
# BƯỚC 2.3: PHÂN TÍCH THỜI ĐIỂM TIẾP CẬN (MONTH & DAY_OF_WEEK)
# =========================================================================
print("\n" + "=" * 50)
print(" BƯỚC 2.3: PHÂN TÍCH THỜI ĐIỂM TIẾP CẬN (HEATMAP)")
print("=" * 50)

pivot_month = df.pivot_table(index='economic_phase', columns='month',       values='y', aggfunc='mean', observed=False)
pivot_day   = df.pivot_table(index='economic_phase', columns='day_of_week', values='y', aggfunc='mean', observed=False)

pivot_month.index   = pivot_month.index.map(PHASE_NAMES)
pivot_month.columns = pivot_month.columns.map(MONTH_LABELS)
pivot_day.index     = pivot_day.index.map(PHASE_NAMES)
pivot_day.columns   = pivot_day.columns.map(DAY_LABELS)

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

for phase_code in [1, 2, 3]:
    df_phase = df[df['economic_phase'] == phase_code]
    print(f"\n▶ GIAI ĐOẠN: {PHASE_NAMES[phase_code].upper()}")

    month_groups = [g['y'].values for _, g in df_phase.groupby('month') if len(g) > 0]
    if len(month_groups) > 1:
        stat_m, p_val_m = kruskal(*month_groups)
        print(f" - Kiểm định Tháng: H-stat = {stat_m:.4f} | p-value = {p_val_m:.4e}")
        print(f"   => Ý nghĩa thống kê theo tháng? {'CÓ (p < 0.05)' if p_val_m < 0.05 else 'KHÔNG'}")

    day_groups = [g['y'].values for _, g in df_phase.groupby('day_of_week') if len(g) > 0]
    if len(day_groups) > 1:
        stat_d, p_val_d = kruskal(*day_groups)
        print(f" - Kiểm định Thứ:   H-stat = {stat_d:.4f} | p-value = {p_val_d:.4e}")
        print(f"   => Ý nghĩa thống kê theo thứ?  {'CÓ (p < 0.05)' if p_val_d < 0.05 else 'KHÔNG'}")

# =========================================================================
# BƯỚC 2.4: PHÂN TÍCH SỐ LẦN GỌI (CAMPAIGN) VÀ ĐƯỜNG CONG CHIẾN DỊCH
# =========================================================================
print("\n" + "=" * 50)
print(" BƯỚC 2.4: PHÂN TÍCH ĐƯỜNG CONG TỶ LỆ THEO SỐ LẦN GỌI")
print("=" * 50)

campaign_pivot = df.groupby(['economic_phase', 'campaign_group'], observed=False)['y'].mean().reset_index()
campaign_pivot['economic_phase_name'] = campaign_pivot['economic_phase'].map(PHASE_NAMES)

campaign_order = ['1', '2', '3', '4', '5+']
campaign_pivot['campaign_group'] = pd.Categorical(campaign_pivot['campaign_group'],
                                                   categories=campaign_order, ordered=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=campaign_pivot, x='campaign_group', y='y',
             hue='economic_phase_name', marker='o', linewidth=2.5, markersize=8)
plt.title("Đường cong Tỷ lệ Chuyển đổi theo Số lần gọi qua các Giai đoạn",
          fontsize=12, fontweight='bold')
plt.xlabel("Số lần tiếp cận trong chiến dịch hiện tại (Campaign Group)", fontsize=10)
plt.ylabel("Tỷ lệ Chuyển đổi (Conversion Rate)", fontsize=10)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Bối cảnh vĩ mô", fontsize=9)
plt.tight_layout()
plt.show()

print("\nKết quả kiểm định Mann-Whitney U cho biến Campaign gốc:")
for phase_code in sorted(df['economic_phase'].unique()):
    df_phase     = df[df['economic_phase'] == phase_code]
    campaign_yes = df_phase[df_phase['y'] == 1]['campaign'].values
    campaign_no  = df_phase[df_phase['y'] == 0]['campaign'].values

    if len(campaign_yes) > 0 and len(campaign_no) > 0:
        stat, p_val = mannwhitneyu(campaign_yes, campaign_no, alternative='two-sided')
        print(f" - Giai đoạn [{PHASE_NAMES.get(phase_code, str(phase_code))}]:")
        print(f"   + U-statistic: {stat:.2f} | p-value = {p_val:.4e}")
        print(f"   => Ý nghĩa thống kê? {'CÓ (p < 0.05)' if p_val < 0.05 else 'KHÔNG'}\n")

# =========================================================================
# BƯỚC 2.5: PHÂN TÍCH THỜI LƯỢNG TƯ VẤN (DURATION) — BIẾN POST-HOC
# =========================================================================
print("\n" + "=" * 50)
print(" BƯỚC 2.5: PHÂN TÍCH THỜI LƯỢNG TƯ VẤN (POST-HOC)")
print("=" * 50)

pivot_duration = df.pivot_table(index='economic_phase', columns='duration_group',
                                values='y', aggfunc='mean', observed=False)
pivot_duration.index   = pivot_duration.index.map(PHASE_NAMES)
pivot_duration.columns = pivot_duration.columns.map(DURATION_LABELS)

plt.figure(figsize=(10, 5))
sns.heatmap(pivot_duration, annot=True, fmt=".2%", cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Tỷ lệ Chuyển đổi theo Phân khúc Thời lượng & Chu kỳ Vĩ mô",
          fontsize=12, fontweight='bold')
plt.ylabel("Giai đoạn Kinh tế", fontsize=10)
plt.xlabel("Phân khúc thời lượng tư vấn cuộc gọi (Post-hoc Variable)", fontsize=10)
plt.tight_layout()
plt.show()


def calculate_cohens_d(group1, group2):
    """Tính kích thước hiệu ứng Cohen's d để đo cường độ ảnh hưởng thực tế."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se  = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_se if pooled_se != 0 else 0


print("\nKiểm định Kruskal-Wallis & Đo lường kích thước hiệu ứng (Cohen's d):")
for phase_code in [1, 2, 3]:
    df_phase     = df[df['economic_phase'] == phase_code]
    duration_yes = df_phase[df_phase['y'] == 1]['duration'].values
    duration_no  = df_phase[df_phase['y'] == 0]['duration'].values

    if len(duration_yes) > 0 and len(duration_no) > 0:
        stat, p_val = kruskal(duration_yes, duration_no)
        d_val       = calculate_cohens_d(duration_yes, duration_no)

        if   abs(d_val) < 0.2: effect_desc = "Rất nhỏ (Negligible)"
        elif abs(d_val) < 0.5: effect_desc = "Nhỏ (Small)"
        elif abs(d_val) < 0.8: effect_desc = "Trung bình (Medium)"
        else:                   effect_desc = "Lớn (Large)"

        print(f" - Giai đoạn [{PHASE_NAMES[phase_code]}]:")
        print(f"   + Kruskal H-stat: {stat:.2f} | p-value = {p_val:.4e}")
        print(f"   + Cohen's d: {d_val:.4f} -> Cường độ: {effect_desc}")
        print(f"   => Ý nghĩa thống kê? {'CÓ' if p_val < 0.05 else 'KHÔNG'}\n")

print("==========================================================")
print(" HOÀN THÀNH TOÀN BỘ PHÂN TÍCH RQ2 (BƯỚC 2.1 ĐẾN 2.5)")
print("==========================================================")

# =========================================================================
# BƯỚC 2.6: CÂY QUYẾT ĐỊNH THEO TỪNG GIAI ĐOẠN KINH TẾ
# =========================================================================
print("\n" + "=" * 80)
print("  BƯỚC 2.6: DECISION TREE THEO TỪNG GIAI ĐOẠN KINH TẾ")
print("=" * 80)

df_encoded = pd.read_csv(INPUT_FILE)
if 'duration' in df_encoded.columns:
    df_encoded = df_encoded[df_encoded['duration'] > 0].reset_index(drop=True)

# --- Đồng bộ mã hóa các biến ---
df_encoded['y_enc'] = df_encoded['y'].astype(int) if df_encoded['y'].dtype != 'object' \
    else (df_encoded['y'].str.strip().str.lower() == 'yes').astype(int)

df_encoded['contact_enc'] = df_encoded['contact'].astype(int) if df_encoded['contact'].dtype != 'object' \
    else (df_encoded['contact'].str.strip().str.lower() == 'cellular').astype(int)

month_order = {'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
               'Th3':3,'Th4':4,'Th5':5,'Th6':6,'Th7':7,'Th8':8,'Th9':9,'Th10':10,'Th11':11,'Th12':12}
df_encoded['month_enc'] = df_encoded['month'].map(month_order).fillna(5).astype(int) \
    if df_encoded['month'].dtype == 'object' else df_encoded['month'].astype(int)

day_order = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'Thứ 2':1,'Thứ 3':2,'Thứ 4':3,'Thứ 5':4,'Thứ 6':5}
df_encoded['day_enc'] = df_encoded['day_of_week'].map(day_order).fillna(1).astype(int) \
    if df_encoded['day_of_week'].dtype == 'object' else df_encoded['day_of_week'].astype(int)

df_encoded['campaign_num_group'] = df_encoded['campaign_group'].map(
    {'1':1,'2':2,'3':3,'4':4,'5+':5}).fillna(5).astype(int) \
    if df_encoded['campaign_group'].dtype == 'object' or \
       isinstance(df_encoded['campaign_group'].dtype, pd.CategoricalDtype) \
    else df_encoded['campaign_group'].astype(int)

df_encoded['duration_num_group'] = df_encoded['duration_group'].astype(int)

# --- Định nghĩa features ---
FEATURES_DT      = ['contact_enc', 'month_enc', 'day_enc', 'campaign_num_group', 'duration_num_group', 'previous']
FEATURE_LABELS   = ['Kênh liên lạc', 'Tháng tiếp cận', 'Ngày trong tuần',
                    'Tần suất gọi (nhóm)', 'Thời lượng gọi (nhóm)', 'Lịch sử tiếp cận trước']
FEATURE_MAP_RULES = dict(zip(FEATURES_DT, FEATURE_LABELS))
TARGET = 'y_enc'

PHASES = ['Giai đoạn Tăng trưởng', 'Giai đoạn Suy thoái', 'Giai đoạn Chạm đáy']

if 'economic_phase' not in df_encoded.columns:
    raise KeyError("Không tìm thấy trường 'economic_phase' trong file.")

if df_encoded['economic_phase'].dtype in [np.int64, np.int32, int]:
    phase_names_map = {1: 'Giai đoạn Tăng trưởng', 2: 'Giai đoạn Suy thoái', 3: 'Giai đoạn Chạm đáy'}
    df_encoded['economic_phase_str'] = df_encoded['economic_phase'].map(phase_names_map)
else:
    df_encoded['economic_phase_str'] = df_encoded['economic_phase'].str.strip()

# --- Vòng lặp huấn luyện 3 cây ---
results_summary = []

for phase in PHASES:
    sep = "*" * 75
    print(f"\n{sep}")
    print(f"  CÂY QUYẾT ĐỊNH — {phase.upper()}")
    print(f"{sep}")

    df_phase = df_encoded[df_encoded['economic_phase_str'] == phase].copy()
    X        = df_phase[FEATURES_DT].values
    y        = df_phase[TARGET].values

    n_total, n_yes = len(y), int(y.sum())
    n_no = n_total - n_yes

    print(f"  [THÔNG SỐ MẪU]:")
    print(f"   + Tổng cuộc gọi     : {n_total:,}")
    print(f"   + Chốt đơn (yes)    : {n_yes:,} ({n_yes/n_total:.2%})")
    print(f"   + Từ chối (no)      : {n_no:,} ({n_no/n_total:.2%})")

    if n_yes < 10:
        print("  => [BỎ QUA] Số mẫu thành công quá nhỏ để chạy Cross-Validation.")
        continue

    # Đánh giá chéo Stratified 5-Fold
    skf             = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores_cv    = []
    auc_scores_cv   = []

    for train_idx, test_idx in skf.split(X, y):
        clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        y_pred  = clf.predict(X[test_idx])
        y_proba = clf.predict_proba(X[test_idx])[:, 1]
        f1_scores_cv.append(f1_score(y[test_idx], y_pred, zero_division=0))
        auc_scores_cv.append(roc_auc_score(y[test_idx], y_proba))

    mean_f1  = np.mean(f1_scores_cv)
    mean_auc = np.mean(auc_scores_cv)

    print(f"\n  [KẾT QUẢ KIỂM ĐỊNH CHÉO 5-FOLD]:")
    print(f"   - F1-Score trung bình : {mean_f1:.4f}")
    print(f"   - AUC-ROC trung bình  : {mean_auc:.4f}")

    # Huấn luyện cây chính thức trên toàn bộ giai đoạn
    final_tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    final_tree.fit(X, y)

    importances     = final_tree.feature_importances_
    importance_pairs = sorted(zip(FEATURE_LABELS, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  [ĐỘ QUAN TRỌNG CÁC BIẾN]:")
    for label, imp in importance_pairs:
        bar = "█" * int(imp * 40)
        print(f"   + {label:<25}: {imp:.4f} ({imp*100:.1f}%) {bar}")

    print(f"\n  [QUY TẮC PHÂN NHÁNH IF-THEN]:")
    print("-" * 60)
    # Ghi chú: export_text dùng tên biến gốc không dấu để tránh lỗi font thư viện
    print(export_text(final_tree, feature_names=FEATURES_DT))
    print("  * Tên biến đầy đủ:")
    for key, val in FEATURE_MAP_RULES.items():
        print(f"    - {key} = [{val}]")
    print("-" * 60)

    results_summary.append({
        'Bối cảnh Vĩ mô'   : phase,
        'Tổng cuộc gọi'    : f"{n_total:,}",
        'Tỷ lệ Conversion' : f"{n_yes/n_total:.2%}",
        'F1-Score'         : f"{mean_f1:.4f}",
        'AUC-ROC'          : f"{mean_auc:.4f}",
        'Biến tác động #1' : importance_pairs[0][0],
        'Biến tác động #2' : importance_pairs[1][0],
    })

print("\n" + "=" * 85)
print("  BẢNG MA TRẬN TỔNG HỢP SO SÁNH ĐA GIAI ĐOẠN")
print("=" * 85)
print(pd.DataFrame(results_summary).to_string(index=False))

print("\n HOÀN THÀNH TÍCH HỢP BƯỚC 2.6!")
print("=" * 85)
