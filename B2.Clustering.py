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

# CẤU HÌNH ĐỒNG BỘ: Ghi đè trực tiếp kết quả vào file encoded để tạo Dataset trung tâm
INPUT_FILE = 'Data_bank_encoded.csv'
OUTPUT_FILE = 'Data_bank_encoded.csv'

def check_file_exists():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file mã hóa '{INPUT_FILE}'. Vui lòng chạy file Data Prep trước.")

def histogram():
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')

    # 4 biến vĩ mô cần phân tích
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']
    X = df[features]

    print("==================================================")
    print("     THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)      ")
    print("==================================================")
    stats = X.describe().T
    stats['median'] = X.median()  
    stats = stats[['count', 'mean', 'median', 'std', 'min', 'max']]
    print(stats.to_string())

    print("\n==================================================")
    print("      ĐANG VẼ HISTOGRAM & MA TRẬN TƯƠNG QUAN      ")
    print("==================================================")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel() 

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

    for i, col in enumerate(features):
        axes[i].hist(X[col], bins=30, color=colors[i], edgecolor='black', alpha=0.7, density=False)
        axes[i].set_title(f'Phân phối của biến {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Giá trị')
        axes[i].set_ylabel('Tần suất (Số lượng mẫu)')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('BIỂU ĐỒ HISTOGRAM KIỂM TRA PHÂN PHỐI ĐA ĐỈNH (MULTIMODAL)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('macro_histograms.png', dpi=300)
    print("-> Đã lưu biểu đồ phân phối vào file 'macro_histograms.png'")
    plt.show()


def correlation_matrix():
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')

    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']
    X = df[features]

    corr_matrix = X.corr().values

    print("==================================================")
    print("BẢNG SỐ LIỆU MA TRẬN TƯƠNG QUAN CHÍNH XÁC:")
    print("==================================================")
    print(X.corr().round(4))
    print("==================================================\n")

    plt.figure(figsize=(9, 7))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Hệ số tương quan (r)')

    ticks = np.arange(len(features))
    plt.xticks(ticks, features, rotation=45, fontsize=11)
    plt.yticks(ticks, features, fontsize=11)

    for i in range(len(features)):
        for j in range(len(features)):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')

    plt.title('MA TRẬ за ТƯƠNG QUAN GIỮA CÁC BIẾN VĨ MÔ (MACRO)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('macro_correlation_matrix.png', dpi=300)
    print("-> Đã lưu biểu đồ ma trận tương quan vào file ảnh: 'macro_correlation_matrix.png'")
    plt.show()


def scale_data():
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']
    X = df[features]

    print("==================================================")
    print("          ĐANG TIẾN HÀNH CHUẨN HÓA DỮ LIỆU        ")
    print("==================================================")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    print("\n5 dòng dữ liệu đầu tiên SAU KHI CHUẨN HÓA:")
    print(X_scaled_df.head())

    print("\nKiểm tra lại Mean và Std sau khi chuẩn hóa (Mục tiêu: Mean ~ 0, Std ~ 1):")
    print(X_scaled_df.agg(['mean', 'std']).round(4))
    
    return X_scaled


def find_optimal_k_silhouette(X_scaled):
    print("\n--- BƯỚC 1.3: ĐANG TÍNH SILHOUETTE SCORE CHO K TỪ 2 ĐẾN 8 ---")
    
    k_range = range(2, 9)
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Lấy mẫu 10000 dòng ngẫu nhiên để tối ưu tốc độ tính toán
        score = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
        silhouette_scores.append(score)
        print(f"Số cụm K = {k} | Silhouette Score = {score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"-> Kết luận định lượng: Số cụm tối ưu nhất là K = {optimal_k}")

    plt.figure(figsize=(9, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linewidth=2, color='#2c3e50', markerfacecolor='#e74c3c', markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'K tối ưu = {optimal_k} (Max Score)')
    
    plt.title('PHÂN TÍCH CHỈ SỐ SILHOUETTE SCORE ĐỂ CHỌN K TỐI ƯU', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Số lượng cụm (K)')
    plt.ylabel('Giá trị Silhouette Score trung bình')
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('silhouette_analysis.png', dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ phân tích Silhouette vào file 'silhouette_analysis.png'")
    
    return optimal_k


def run_kmeans_final(X_scaled):
    print("\n==================================================")
    print("      BƯỚC 1.4: CHẠY THUẬT TOÁN K-MEANS CHÍNH THỨC     ")
    print("==================================================")
    
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)
    
    final_inertia = kmeans_model.inertia_
    
    print(f"-> Thuật toán K-Means đã hội tụ thành công sau {kmeans_model.n_iter_} vòng lặp.")
    print(f"-> Giá trị Inertia (Tổng bình phương khoảng cách nội cụm) cuối cùng: {final_inertia:.4f}")
    
    df['KMeans_Cluster'] = cluster_labels + 1 
    
    print("\nSố lượng quan sát trong mỗi cụm kinh tế:")
    print(df['KMeans_Cluster'].value_counts().sort_index())
    
    print("\nBẢNG GIÁ TRỊ TRUNG BÌNH CỦA 4 BIẾN VĨ MÔ THEO 3 CỤM K-MEANS:")
    mean_table = df.groupby('KMeans_Cluster')[features].mean()
    print(mean_table.round(4))
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n====== HOÀN TẤT! ĐÃ LƯU KẾT QUẢ PHÂN CỤM VÀO FILE '{OUTPUT_FILE}' ======")
    
    return df


def AH_Clustering():
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

    sample_size = min(5500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    X_sample = df_sample[features]

    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)

    print(f"Đang tính toán trên tập mẫu {sample_size} dòng để tránh tràn RAM và vẽ Dendrogram...")
    plt.figure(figsize=(12, 7))

    Z = linkage(X_sample_scaled, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=40, leaf_font_size=10, show_contracted=True)

    plt.title(f'Sơ đồ cây Phân cấp (Dendrogram) - Lấy mẫu {sample_size} dòng')
    plt.xlabel('Các nhánh/Cụm dữ liệu được gộp')
    plt.ylabel('Khoảng cách liên kết (Linkage Distance)')
    plt.axhline(y=85, color='r', linestyle='--', label='Đường cắt tạo ra 3 cụm tự nhiên')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dendrogram_plot.png', dpi=300)
    print("-> Đã lưu sơ đồ cây thành công vào file 'dendrogram_plot.png'")
    plt.show()

    agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    agg_labels = agg_clustering.fit_predict(X_sample_scaled)

    kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sample_scaled)

    ari_agg = adjusted_rand_score(kmeans_labels, agg_labels)
    nmi_agg = normalized_mutual_info_score(kmeans_labels, agg_labels)

    print("\n--- ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA K-MEANS VÀ HIERARCHICAL (TRÊN TẬP MẪU) ---")
    print(f"Adjusted Rand Index (ARI): {ari_agg:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_agg:.4f}\n")

    df_sample['Hierarchical_Cluster'] = agg_labels
    print("--- GIÁ TRỊ TRUNG BÌNH CỦA CÁC CỤM THEO HIERARCHICAL CLUSTERING ---")
    print(df_sample.groupby('Hierarchical_Cluster')[features].mean())


def profile_and_name_phases(X_scaled):
    print("\n==================================================")
    print("     BƯỚC 1.6: PROFILE VÀ ĐẶT TÊN GIAI ĐOẠN ĐỊNH LƯỢNG  ")
    print("==================================================")
    
    check_file_exists()
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)
    df['Cluster_No'] = cluster_labels

    # 1. TẠO CỘT CHỮ: Ánh xạ nhãn văn bản thân thiện phục vụ viết báo cáo lý thuyết
    phase_name_mapping = {
        0: 'Giai đoạn Tăng trưởng',
        1: 'Giai đoạn Suy thoái',
        2: 'Giai đoạn chạm đáy'
    }
    df['phase_name'] = df['Cluster_No'].map(phase_name_mapping)

    # 2. TẠO CỘT SỐ: Ánh xạ mã định danh số (1, 2, 3) phục vụ chia nhánh Mô hình cây định lượng
    phase_num_mapping = {
        0: 1,
        1: 2,
        2: 3
    }
    df['economic_phase'] = df['Cluster_No'].map(phase_num_mapping)

    # 3. TRÍCH XUẤT TÂM CỤM DẠNG Z-SCORE CHO BIỂU ĐỒ HEATMAP
    cluster_centers_z = kmeans_model.cluster_centers_
    z_score_table = pd.DataFrame(cluster_centers_z, columns=features)
    
    # Đồng bộ hóa Index theo tên chu kỳ để Reindex không bị sinh ra giá trị rỗng (NaN)
    z_score_table.index = [phase_name_mapping[i] for i in z_score_table.index]
    order_phases = ['Giai đoạn Tăng trưởng', 'Giai đoạn Suy thoái', 'Giai đoạn chạm đáy']
    z_score_table = z_score_table.reindex(order_phases)

    print("\n[BẢNG 1] MA TRẬN TÂM CỤM ĐÃ CHUẨN HÓA (Z-SCORE) - DÙNG CHO HEATMAP:")
    print("====================================================================")
    print(z_score_table.round(4))
    print("====================================================================")

    plt.figure(figsize=(9, 5.5))
    sns.heatmap(z_score_table, annot=True, fmt=".2f", cmap="RdBu_r", cbar=True,
                linewidths=0.8, annot_kws={"size": 11, "weight": "bold"})
    
    plt.title('BIỂU ĐỒ HEATMAP PROFILE TÂM CỤM VĨ MÔ (Z-SCORE)', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Các giai đoạn kinh tế phát hiện', fontsize=10)
    plt.xlabel('Các biến số kinh tế vĩ mô', fontsize=10)
    plt.tight_layout()
    
    plt.savefig('cluster_profile_heatmap.png', dpi=300)
    print("-> [XONG] Đã xuất biểu đồ Heatmap Z-score vào file: 'cluster_profile_heatmap.png'")

    print("\n[BẢNG 2] PROFILE CÁC BIẾN VĨ MÔ THEO ĐƠN VỊ GỐC (DỄ DIỄN GIẢI):")
    print("====================================================================")
    profile_table = df.groupby('phase_name')[features].mean()
    profile_table = profile_table.reindex(order_phases)
    print(profile_table.round(4))
    print("====================================================================")
    
    print("\nSố lượng dòng quan sát rơi vào từng giai đoạn kinh tế (Theo mã số định lượng):")
    print(df['economic_phase'].value_counts().sort_index())

    # Đồng bộ hóa cấu trúc: Xóa bỏ cột mã hóa cụm tạm thời để giữ sạch tệp kết quả phẳng
    df = df.drop(columns=['Cluster_No'])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n====== [HOÀN TẤT BƯỚC 1.6] ĐÃ TÍCH HỢP ĐỒNG THỜI 'phase_name' VÀ 'economic_phase' VÀO FILE TRUNG TÂM '{OUTPUT_FILE}' ======")
    
    return df


def main():
    print("==================================================")
    print("   CHẠY LUỒNG PHÂN TÍCH CHÍNH THỨC (QUY TRÌNH MỚI) ")
    print("==================================================")

    # Khóa các bước EDA cũ (Bỏ comment nếu nhóm muốn xuất lại ảnh biểu đồ)
    # histogram()
    # correlation_matrix()
    
    # 1. BẮT BUỘC GIỮ: Chuẩn hóa dữ liệu gốc để tạo mảng dữ liệu X_scaled
    X_scaled = scale_data()
    
    # 2. Khóa bước tìm K của Silhouette (Bỏ comment nếu muốn chạy lại phân tích K)
    # find_optimal_k_silhouette(X_scaled)
    
    # 3. Khóa bước chạy KMeans cơ bản (Bỏ comment nếu muốn xuất file nhãn KMeans_Cluster số)
    # run_kmeans_final(X_scaled)
    
    # 4. Khóa bước chạy phân tầng đối chiếu (Bỏ comment nếu muốn vẽ lại Dendrogram mẫu)
    # AH_Clustering()

    # 5. CHẠY CHÍNH THỨC: Trích xuất hồ sơ tâm cụm vĩ mô và xuất Dataset đồng bộ cuối cùng
    df_final = profile_and_name_phases(X_scaled)

if __name__ == "__main__":
    main()