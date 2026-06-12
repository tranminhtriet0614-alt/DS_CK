import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================================================================
# CẤU HÌNH: Ghi đè trực tiếp kết quả vào file encoded để tạo Dataset trung tâm
# =========================================================================
INPUT_FILE  = 'Data_bank_encoded.csv'
OUTPUT_FILE = 'Data_bank_encoded.csv'

FEATURES = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

PHASE_NAME_MAP = {0: 'Giai đoạn Tăng trưởng', 1: 'Giai đoạn Suy thoái', 2: 'Giai đoạn Chạm đáy'}
PHASE_NUM_MAP  = {0: 1, 1: 2, 2: 3}
PHASE_ORDER    = ['Giai đoạn Tăng trưởng', 'Giai đoạn Suy thoái', 'Giai đoạn Chạm đáy']


def check_file_exists():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy file mã hóa '{INPUT_FILE}'. Vui lòng chạy file Data_preparation.py trước."
        )


def histogram():
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    X  = df[FEATURES]

    print("=" * 50)
    print("     THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)      ")
    print("=" * 50)
    stats          = X.describe().T
    stats['median'] = X.median()
    stats           = stats[['count', 'mean', 'median', 'std', 'min', 'max']]
    print(stats.to_string())

    print("\n" + "=" * 50)
    print("      ĐANG VẼ HISTOGRAM & MA TRẬN TƯƠNG QUAN      ")
    print("=" * 50)

    colors   = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.ravel()

    for i, col in enumerate(FEATURES):
        axes[i].hist(X[col], bins=30, color=colors[i], edgecolor='black', alpha=0.7, density=False)
        axes[i].set_title(f'Phân phối của biến {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Giá trị')
        axes[i].set_ylabel('Tần suất (Số lượng mẫu)')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('BIỂU ĐỒ HISTOGRAM KIỂM TRA PHÂN PHỐI ĐA ĐỈNH (MULTIMODAL)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('macro_histograms.png', dpi=300)
    print("-> Đã lưu biểu đồ phân phối vào file 'macro_histograms.png'")
    plt.show()


def correlation_matrix():
    check_file_exists()
    df          = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    X           = df[FEATURES]
    corr_matrix = X.corr().values

    print("=" * 50)
    print("BẢNG SỐ LIỆU MA TRẬN TƯƠNG QUAN CHÍNH XÁC:")
    print("=" * 50)
    print(X.corr().round(4))
    print("=" * 50)

    plt.figure(figsize=(9, 7))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Hệ số tương quan (r)')

    ticks = np.arange(len(FEATURES))
    plt.xticks(ticks, FEATURES, rotation=45, fontsize=11)
    plt.yticks(ticks, FEATURES, fontsize=11)

    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                     ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')

    plt.title('MA TRẬN TƯƠNG QUAN GIỮA CÁC BIẾN VĨ MÔ (MACRO)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('macro_correlation_matrix.png', dpi=300)
    print("-> Đã lưu biểu đồ ma trận tương quan vào file 'macro_correlation_matrix.png'")
    plt.show()


def scale_data():
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    X  = df[FEATURES]

    print("=" * 50)
    print("          ĐANG TIẾN HÀNH CHUẨN HÓA DỮ LIỆU        ")
    print("=" * 50)

    scaler        = StandardScaler()
    X_scaled      = scaler.fit_transform(X)
    X_scaled_df   = pd.DataFrame(X_scaled, columns=FEATURES)

    print("\n5 dòng dữ liệu đầu tiên SAU KHI CHUẨN HÓA:")
    print(X_scaled_df.head())

    print("\nKiểm tra Mean và Std sau khi chuẩn hóa (Mục tiêu: Mean ~ 0, Std ~ 1):")
    print(X_scaled_df.agg(['mean', 'std']).round(4))

    return X_scaled


def find_optimal_k_elbow(X_scaled):
    print("\n--- BƯỚC 1.2b: ĐANG TÍNH ĐỘ LỆCH NỘI CỤM (INERTIA) CHO K TỪ 1 ĐẾN 9 ---")

    k_range  = range(1, 10)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"Số cụm K = {k} | Inertia (WCSS) = {km.inertia_:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(k_range, inertias, marker='o', linewidth=2,
             color='#2c3e50', markerfacecolor='#e67e22', markersize=8)
    plt.axvline(x=3, color='r', linestyle='--', label='Điểm cùi chỏ tối ưu (K = 3)')
    plt.title('PHÂN TÍCH ĐƯỜNG CONG CÙI CHỎ (ELBOW METHOD) ĐỂ CHỌN K TỐI ƯU',
              fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Số lượng cụm (K)')
    plt.ylabel('Giá trị Inertia (Within-Cluster Sum of Squares)')
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('elbow_analysis.png', dpi=300)
    print("-> Đã lưu biểu đồ phân tích Elbow vào file 'elbow_analysis.png'")
    plt.show()


def find_optimal_k_silhouette(X_scaled):
    print("\n--- BƯỚC 1.3: ĐANG TÍNH SILHOUETTE SCORE CHO K TỪ 2 ĐẾN 8 ---")

    k_range           = range(2, 9)
    silhouette_scores = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        # Hạ sample_size xuống 3000 để tránh lỗi tràn bộ nhớ RAM (ArrayMemoryError)
        score  = silhouette_score(X_scaled, labels, sample_size=3000, random_state=42)
        silhouette_scores.append(score)
        print(f"Số cụm K = {k} | Silhouette Score = {score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"-> Kết luận định lượng: Số cụm tối ưu nhất là K = {optimal_k}")

    plt.figure(figsize=(9, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linewidth=2,
             color='#2c3e50', markerfacecolor='#e74c3c', markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'K tối ưu = {optimal_k} (Max Score)')
    plt.title('PHÂN TÍCH CHỈ SỐ SILHOUETTE SCORE ĐỂ CHỌN K TỐI ƯU',
              fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Số lượng cụm (K)')
    plt.ylabel('Giá trị Silhouette Score trung bình')
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=300)
    print("-> Đã lưu biểu đồ phân tích Silhouette vào file 'silhouette_analysis.png'")
    plt.show()

    return optimal_k


def run_kmeans_final(X_scaled):
    print("\n" + "=" * 50)
    print("     BƯỚC 1.4: CHẠY THUẬT TOÁN K-MEANS CHÍNH THỨC     ")
    print("=" * 50)

    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')

    kmeans_model   = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)

    print(f"-> K-Means hội tụ sau {kmeans_model.n_iter_} vòng lặp.")
    print(f"-> Inertia cuối cùng: {kmeans_model.inertia_:.4f}")

    df['KMeans_Cluster'] = cluster_labels + 1
    print("\nSố quan sát trong mỗi cụm kinh tế:")
    print(df['KMeans_Cluster'].value_counts().sort_index())
    print("\nBẢNG GIÁ TRỊ TRUNG BÌNH THEO 3 CỤM K-MEANS:")
    print(df.groupby('KMeans_Cluster')[FEATURES].mean().round(4))

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n====== HOÀN TẤT! Đã lưu kết quả phân cụm vào '{OUTPUT_FILE}' ======")

    return df, kmeans_model, cluster_labels


def AH_Clustering(X_scaled):
    """Phân cụm phân cấp (Hierarchical) để xác nhận kết quả K-Means."""
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')

    sample_size  = min(5500, len(df))
    sample_idx   = df.sample(n=sample_size, random_state=42).index
    X_sample     = X_scaled[sample_idx]

    print(f"\nĐang tính trên tập mẫu {sample_size} dòng để tránh tràn RAM...")

    plt.figure(figsize=(12, 7))
    Z = linkage(X_sample, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=40,
               leaf_font_size=10, show_contracted=True)
    plt.title(f'Sơ đồ cây Phân cấp (Dendrogram) — Mẫu {sample_size} dòng')
    plt.xlabel('Các nhánh/Cụm dữ liệu được gộp')
    plt.ylabel('Khoảng cách liên kết (Linkage Distance)')
    plt.axhline(y=85, color='r', linestyle='--', label='Đường cắt tạo ra 3 cụm tự nhiên')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dendrogram_plot.png', dpi=300)
    print("-> Đã lưu sơ đồ cây vào file 'dendrogram_plot.png'")
    plt.show()

    agg_labels    = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_sample)
    kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sample)

    ari = adjusted_rand_score(kmeans_labels, agg_labels)
    nmi = normalized_mutual_info_score(kmeans_labels, agg_labels)

    print("\n--- ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA K-MEANS VÀ HIERARCHICAL (TRÊN TẬP MẪU) ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    df_sample = df.loc[sample_idx].copy()
    df_sample['Hierarchical_Cluster'] = agg_labels
    print("\n--- GIÁ TRỊ TRUNG BÌNH CÁC CỤM THEO HIERARCHICAL CLUSTERING ---")
    print(df_sample.groupby('Hierarchical_Cluster')[FEATURES].mean())


def profile_and_name_phases(X_scaled, kmeans_model, cluster_labels):
    """Profile và đặt tên giai đoạn kinh tế dựa trên model đã huấn luyện từ run_kmeans_final."""
    print("\n" + "=" * 50)
    print("     BƯỚC 1.6: PROFILE VÀ ĐẶT TÊN GIAI ĐOẠN ĐỊNH LƯỢNG  ")
    print("=" * 50)

    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')

    df['Cluster_No']    = cluster_labels
    df['phase_name']    = df['Cluster_No'].map(PHASE_NAME_MAP)
    df['economic_phase'] = df['Cluster_No'].map(PHASE_NUM_MAP)

    # Bảng tâm cụm Z-score
    z_score_table = pd.DataFrame(kmeans_model.cluster_centers_, columns=FEATURES)
    z_score_table.index = [PHASE_NAME_MAP[i] for i in z_score_table.index]
    z_score_table = z_score_table.reindex(PHASE_ORDER)

    print("\n[BẢNG 1] MA TRẬN TÂM CỤM ĐÃ CHUẨN HÓA (Z-SCORE) — DÙNG CHO HEATMAP:")
    print("=" * 68)
    print(z_score_table.round(4))
    print("=" * 68)

    plt.figure(figsize=(9, 5.5))
    sns.heatmap(z_score_table, annot=True, fmt=".2f", cmap="RdBu_r", cbar=True,
                linewidths=0.8, annot_kws={"size": 11, "weight": "bold"})
    plt.title('BIỂU ĐỒ HEATMAP PROFILE TÂM CỤM VĨ MÔ (Z-SCORE)',
              fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Các giai đoạn kinh tế phát hiện', fontsize=10)
    plt.xlabel('Các biến số kinh tế vĩ mô', fontsize=10)
    plt.tight_layout()
    plt.savefig('cluster_profile_heatmap.png', dpi=300)
    print("-> Đã xuất biểu đồ Heatmap Z-score vào file 'cluster_profile_heatmap.png'")
    plt.show()

    print("\n[BẢNG 2] PROFILE CÁC BIẾN VĨ MÔ THEO ĐƠN VỊ GỐC:")
    print("=" * 68)
    profile_table = df.groupby('phase_name')[FEATURES].mean().reindex(PHASE_ORDER)
    print(profile_table.round(4))
    print("=" * 68)

    print("\nSố quan sát theo mã số giai đoạn kinh tế:")
    print(df['economic_phase'].value_counts().sort_index())

    df = df.drop(columns=['Cluster_No'])
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n====== HOÀN TẤT BƯỚC 1.6! Đã tích hợp 'phase_name' và 'economic_phase' vào '{OUTPUT_FILE}' ======")

    return df


def main():
    print("=" * 50)
    print("   CHẠY LUỒNG PHÂN TÍCH CHÍNH THỨC (QUY TRÌNH MỚI) ")
    print("=" * 50)

    histogram()
    correlation_matrix()

    X_scaled = scale_data()
    find_optimal_k_elbow(X_scaled)
    find_optimal_k_silhouette(X_scaled)

    df_final, kmeans_model, cluster_labels = run_kmeans_final(X_scaled)
    AH_Clustering(X_scaled)
    profile_and_name_phases(X_scaled, kmeans_model, cluster_labels)


if __name__ == "__main__":
    main()
