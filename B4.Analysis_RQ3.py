import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

# =========================================================================
# CẤU HÌNH HỆ THỐNG & HIỂN THỊ
# =========================================================================
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("=" * 60)
print("  TIẾN HÀNH RQ3: HIỆU QUẢ TÁI TIẾP CẬN THEO CHU KỲ VĨ MÔ")
print("=" * 60)


# =========================================================================
# BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# =========================================================================
INPUT_FILE = './Data_bank_encoded.csv'
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Không tìm thấy file '{INPUT_FILE}'. Vui lòng chạy file dữ liệu sạch trước!"
    )

df = pd.read_csv(INPUT_FILE)

# Từ điển nhãn dùng chung cho toàn bộ script
POUTCOME_MAP_FULL = {
    1: '1. Nhóm lần đầu',
    0: '2. Nhóm tái tiếp cận',
    2: '3. Nhóm kiểm soát dương',
}
PHASE_MAP = {
    1: 'Giai đoạn Tăng trưởng',
    2: 'Giai đoạn Suy thoái',
    3: 'Giai đoạn chạm đáy',
}
PHASE_ORDER = ['Giai đoạn Tăng trưởng', 'Giai đoạn Suy thoái', 'Giai đoạn chạm đáy']

df['reapproach_group'] = df['poutcome'].map(POUTCOME_MAP_FULL)
df['phase_name']       = df['economic_phase'].map(PHASE_MAP)

df = df.dropna(subset=['reapproach_group', 'economic_phase', 'y'])
df['economic_phase'] = df['economic_phase'].astype(int)
df['y']              = df['y'].astype(int)


# =========================================================================
# BƯỚC 2: MÔ TẢ ĐẶC TRƯNG & TỶ LỆ CHUYỂN ĐỔI BASELINE
# =========================================================================
print("\n" + "=" * 55)
print(" BƯỚC 2: MÔ TẢ ĐẶC TRƯNG CẤU TRÚC CÁC NHÓM KHÁCH HÀNG")
print("=" * 55)

print("SỐ LƯỢNG MẪU THỰC TẾ MỖI NHÓM TƯƠNG TÁC:")
print(df['reapproach_group'].value_counts().to_string())
print("-" * 60)

conversion_rate = df.groupby('reapproach_group', observed=False)['y'].mean().reset_index()

plt.figure(figsize=(9, 5))
sns.barplot(
    data=conversion_rate, x='reapproach_group', y='y',
    hue='reapproach_group', palette='Set2', legend=False,
)
plt.title(
    'Tỷ lệ Chuyển đổi giữa các Nhóm Khách hàng (RQ3 Baseline Benchmark)',
    fontsize=11, fontweight='bold',
)
plt.xlabel('Phân nhóm theo Lịch sử tương tác', fontsize=10)
plt.ylabel('Tỷ lệ Chuyển đổi (Conversion Rate)', fontsize=10)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

for idx, row in conversion_rate.iterrows():
    plt.text(idx, row['y'] + 0.01, f"{row['y']:.2%}", ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('./RQ3_baseline_conversion.png', dpi=300)
plt.show()
print("-> [HỆ THỐNG] Đã xuất: './RQ3_baseline_conversion.png'\n")


# =========================================================================
# BƯỚC 3: BẢNG CHÉO TỶ LỆ CHUYỂN ĐỔI THEO GIAI ĐOẠN VÀ NHÓM KHÁCH HÀNG
# =========================================================================
crosstab_rate = (
    df.pivot_table(index='reapproach_group', columns='phase_name', values='y', aggfunc='mean')
    * 100
)[PHASE_ORDER]

print("=" * 75)
print("BẢNG TRUNG TÂM RQ3: TỶ LỆ CHUYỂN ĐỔI THEO GIAI ĐOẠN VÀ NHÓM KHÁCH HÀNG (%)")
print("=" * 75)
print(crosstab_rate.round(2).to_string())
print()


# =========================================================================
# BƯỚC 4: PHÂN PHỐI SỐ LƯỢNG & TỶ LỆ CẤU TRÚC MẪU
# =========================================================================
df_counts = pd.crosstab(df['reapproach_group'], df['phase_name'])[PHASE_ORDER]
print("=" * 80)
print("BẢNG 1: SỐ LƯỢNG MẪU THỰC TẾ (COUNT) PHÂN BỔ QUA CÁC GIAI ĐOẠN")
print("=" * 80)
display(df_counts)

df_pct = pd.crosstab(df['reapproach_group'], df['phase_name'], normalize='index')[PHASE_ORDER] * 100
print("\n" + "=" * 80)
print("BẢNG 2: TỶ LỆ % CẤU TRÚC PHÂN PHỐI MẪU THEO TỪNG NHÓM KHÁCH HÀNG")
print("=" * 80)
display(df_pct.round(2))


# =========================================================================
# BƯỚC 5: BIỂU ĐỒ CỘT CHỒNG (STACKED BAR CHART)
# =========================================================================
colors = ['#2b5c8f', '#d95f02', '#7570b3']
ax = df_pct.plot(kind='barh', stacked=True, color=colors, figsize=(10, 5), width=0.6)

plt.title(
    'CẤU TRÚC PHÂN PHỐI CUỘC GỌI VÀO CÁC GIAI ĐOẠN KINH TẾ (%)',
    fontsize=12, fontweight='bold', pad=15,
)
plt.xlabel('Tỷ lệ phần trăm (%)', fontsize=10)
plt.ylabel('Nhóm khách hàng', fontsize=10)
plt.xlim(0, 100)
plt.legend(title='Bối cảnh vĩ mô', bbox_to_anchor=(1.02, 1), loc='upper left')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y_pos = p.get_xy()
    if width > 5:
        ax.text(
            x + width / 2, y_pos + height / 2, f'{width:.1f}%',
            ha='center', va='center', color='white', fontweight='bold', fontsize=9,
        )

plt.tight_layout()
plt.savefig('./distribution_simpson.png', dpi=300)
plt.show()
print("-> [HỆ THỐNG] Đã xuất: './distribution_simpson.png'\n")


# =========================================================================
# BƯỚC 6: LOGISTIC REGRESSION ĐỐI KHÁNG (CHỈ GIỮ NONEXISTENT VÀ FAILURE)
# =========================================================================
print("=" * 70)
print(" BƯỚC 6: LOGISTIC REGRESSION ĐỐI KHÁNG (NONEXISTENT vs FAILURE)")
print("=" * 70)

# --- 6.1: LỌC BỎ NHÓM SUCCESS ---
df_model = df[df['poutcome'].isin([0, 1])].copy()
print(f"Shape sau khi lọc bỏ nhóm Success: {df_model.shape}")

POUTCOME_MAP_BINARY = {
    1: '1. Nhóm lần đầu (Nonexistent)',
    0: '2. Nhóm tái tiếp cận (Failure)',
}
df_model['reapproach_group'] = df_model['poutcome'].map(POUTCOME_MAP_BINARY)

print("\nSỐ LƯỢNG MẪU THEO NHÓM ĐỐI KHÁNG (ĐÃ LOẠI BỎ SUCCESS)")
print("=" * 70)
print(df_model['reapproach_group'].value_counts().to_string())
print(f"Tỷ lệ chuyển đổi baseline trên tập lọc: {df_model['y'].mean():.2%}\n")

# --- 6.2: MÃ HÓA BIẾN, TẠO BIẾN TƯƠNG TÁC ---
df_model = df_model.dropna(
    subset=['contact', 'campaign', 'duration', 'age', 'job', 'economic_phase', 'y']
).copy()

# Biến nhị phân: 1 = Failure, 0 = Nonexistent
df_model['poutcome_Failure'] = np.where(df_model['poutcome'] == 0, 1, 0)

# One-hot encoding cho job
job_dummies  = pd.get_dummies(df_model['job'].astype(str), prefix='job', drop_first=True)
df_model     = pd.concat([df_model, job_dummies], axis=1)
job_features = job_dummies.columns.tolist()

# Chuẩn hóa Z-score các biến định lượng
numeric_features = ['contact', 'campaign', 'duration', 'age']
scaler = StandardScaler()
df_model[numeric_features] = scaler.fit_transform(df_model[numeric_features])

# Biến tương tác lõi
df_model['poutcome_Failure_x_economic_phase'] = (
    df_model['poutcome_Failure'] * df_model['economic_phase']
)

X_features = numeric_features + job_features + [
    'economic_phase', 'poutcome_Failure', 'poutcome_Failure_x_economic_phase'
]
X = df_model[X_features].values
y = df_model['y'].astype(int).values

print("MA TRẬN DỮ LIỆU ĐƯA VÀO MÔ HÌNH HỒI QUY ĐỐI KHÁNG")
print("=" * 70)
print(f"Số lượng mẫu   : {X.shape[0]}")
print(f"Số lượng đặc trưng: {X.shape[1]}")
print(f"Cấu trúc lớp   : Lớp 0 ({np.sum(y == 0)} mẫu) | Lớp 1 ({np.sum(y == 1)} mẫu)\n")

# --- 6.3: CROSS-VALIDATION ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, f1_scores_cv = [], []

for train_idx, val_idx in skf.split(X, y):
    clf = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    clf.fit(X[train_idx], y[train_idx])
    auc_scores.append(roc_auc_score(y[val_idx], clf.predict_proba(X[val_idx])[:, 1]))
    f1_scores_cv.append(f1_score(y[val_idx], clf.predict(X[val_idx])))

# --- 6.4: HUẤN LUYỆN MÔ HÌNH CUỐI & BẢNG HỆ SỐ ---
log_reg_final = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
log_reg_final.fit(X, y)

coef_df = pd.DataFrame({'Biến': X_features, 'Hệ số (Beta)': log_reg_final.coef_[0]})
coef_df = pd.concat([
    coef_df,
    pd.DataFrame({'Biến': ['Intercept'], 'Hệ số (Beta)': log_reg_final.intercept_}),
], ignore_index=True)

coef_df['Odds Ratio'] = np.exp(coef_df['Hệ số (Beta)'])
coef_df['Khả năng chuyển đổi thay đổi (%)'] = (coef_df['Odds Ratio'] - 1) * 100

def clean_feature_name(name: str) -> str:
    """Rút gọn tên biến cho bảng kết quả. Lưu ý: chuỗi dài phải replace trước."""
    replacements = [
        ('poutcome_Failure_x_economic_phase', 'poutcome_x_economic_phase'),
        ('poutcome_Failure', 'poutcome_failure'),
        ('Intercept', 'Hệ số chặn'),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name

coef_df['Biến'] = coef_df['Biến'].apply(clean_feature_name)

# Sắp xếp: giảm dần theo Beta, Hệ số chặn xuống cuối
intercept_row = coef_df[coef_df['Biến'] == 'Hệ số chặn']
coef_df = pd.concat([
    coef_df[coef_df['Biến'] != 'Hệ số chặn'].sort_values('Hệ số (Beta)', ascending=False),
    intercept_row,
]).reset_index(drop=True)

coef_df['Hệ số (Beta)'] = coef_df['Hệ số (Beta)'].round(6)
coef_df['Odds Ratio']   = coef_df['Odds Ratio'].round(6)
coef_df['Khả năng chuyển đổi thay đổi (%)'] = coef_df['Khả năng chuyển đổi thay đổi (%)'].apply(
    lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
)

print("=" * 85)
print(" BẢNG TỔNG HỢP HỆ SỐ HỒI QUY LOGISTIC")
print("=" * 85)
display(coef_df.set_index('Biến'))

# --- 6.5: KẾT LUẬN ---
interaction_row = coef_df[coef_df['Biến'] == 'poutcome_x_economic_phase'].iloc[0]

print("\n" + "=" * 85)
print(" KẾT LUẬN CUỐI CÙNG CHO CÂU HỎI RQ3 (KỊCH BẢN ĐỐI KHÁNG)")
print("=" * 85)
print(f"📊 Hiệu suất mô hình kiểm soát:")
print(f"   - AUC-ROC (Cross-Val): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"   - F1-Score (Cross-Val): {np.mean(f1_scores_cv):.4f} ± {np.std(f1_scores_cv):.4f}")
print(f"\n🎯 BIẾN TƯƠNG TÁC LÕI (poutcome_x_economic_phase):")
print(f"   - Hệ số hồi quy: {interaction_row['Hệ số (Beta)']}")
print(f"   - Odds Ratio   : {interaction_row['Odds Ratio']}")
print(f"   - Mức độ tác động: {interaction_row['Khả năng chuyển đổi thay đổi (%)']}")


# =========================================================================
# BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ ĐÁNH GIÁ
# =========================================================================
print("\n" + "=" * 70)
print(" BƯỚC 7: XUẤT BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH")
print("=" * 70)

mean_auc = np.mean(auc_scores)
std_auc  = np.std(auc_scores)
mean_f1  = np.mean(f1_scores_cv)
std_f1   = np.std(f1_scores_cv)

# --- 7.1: CROSS-VALIDATION BAR CHART ---
fig_cv, ax_cv = plt.subplots(figsize=(8, 6))
labels = ['AUC-ROC', 'F1-Score']
means  = [mean_auc, mean_f1]
errors = [std_auc, std_f1]
colors_cv = ['#A8D8EA', '#E57373']

bars = ax_cv.bar(labels, means, yerr=errors, capsize=6, color=colors_cv,
                 edgecolor='gray', error_kw={'capthick': 1.5, 'elinewidth': 1.5})

ax_cv.set_title('Kết quả Cross-Validation (5-Fold)\nVới thanh Error Bar kiểm định',
                fontsize=13, fontweight='bold', pad=15)
ax_cv.set_ylabel('Score', fontsize=11)
ax_cv.set_ylim(0, 1.05)
ax_cv.grid(axis='y', linestyle='--', alpha=0.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax_cv.text(
        bar.get_x() + bar.get_width() / 2, height + errors[i] + 0.02,
        f'{means[i]:.4f}', ha='center', fontweight='bold', fontsize=10,
    )

plt.tight_layout()
plt.savefig('./RQ3_Cross_Validation_Result.png', dpi=300)
plt.show()
print("-> Đã xuất: './RQ3_Cross_Validation_Result.png'")

# --- 7.2: ROC CURVE ---
y_pred_proba_final = log_reg_final.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_pred_proba_final)
final_auc    = roc_auc_score(y, y_pred_proba_final)

fig_roc, ax_roc = plt.subplots(figsize=(9, 6.5))
ax_roc.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {final_auc:.4f})')
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

ax_roc.set_title('ROC Curve - Logistic Regression\n(Toàn bộ tập dữ liệu tích hợp)',
                 fontsize=14, fontweight='bold', pad=15)
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_xlim([-0.01, 1.0])
ax_roc.set_ylim([0.0, 1.02])
ax_roc.grid(True, linestyle='-', alpha=0.5)
ax_roc.legend(loc='lower right', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig('./RQ3_ROC_Curve_Result.png', dpi=300)
plt.show()
print("-> Đã xuất: './RQ3_ROC_Curve_Result.png'")
