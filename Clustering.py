import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score  # Thêm dòng này ở nhóm import đầu file

def histogram():
    # 1. Đọc dữ liệu từ file của bạn
    df = pd.read_csv('Data_bank_o.csv', encoding='latin1')

    # 4 biến vĩ mô cần phân tích
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']
    X = df[features]

    print("==================================================")
    print("     THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS)      ")
    print("==================================================")
    # Tính toán mean, median (50%), std, min, max
    stats = X.describe().T
    stats['median'] = X.median()  # bổ sung cột median rõ ràng
    # Sắp xếp lại thứ tự cột cho đẹp mắt
    stats = stats[['count', 'mean', 'median', 'std', 'min', 'max']]
    print(stats.to_string())


    print("\n==================================================")
    print("      ĐANG VẼ HISTOGRAM & MA TRẬN TƯƠNG QUAN      ")
    print("==================================================")

    # --- PHẦN 1: VẼ HISTOGRAM ĐỂ KIỂM TRA PHÂN PHỐI ĐA ĐỈNH ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel() # Phẳng hóa ma trận đồ thị 2x2 thành mảng 1 chiều để duyệt vòng lặp

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

    for i, col in enumerate(features):
        # Vẽ histogram với đường ước lượng mật độ (Density line) mô phỏng ngầm
        counts, bins, patches = axes[i].hist(X[col], bins=30, color=colors[i], edgecolor='black', alpha=0.7, density=False)
        
        axes[i].set_title(f'Phân phối của biến {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Giá trị')
        axes[i].set_ylabel('Tần suất (Số lượng mẫu)')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('BIỂU ĐỒ HISTOGRAM KIỂM TRA PHÂN PHỐI ĐA ĐỈNH (MULTIMODAL)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('macro_histograms.png')
    print("-> Đã lưu biểu đồ phân phối vào file 'macro_histograms.png'")
    plt.show()


def correlation_matrix():
    # 1. Đọc dữ liệu từ file của bạn
    df = pd.read_csv('Data_bank_o.csv', encoding='latin1')

    # Lọc ra 4 biến kinh tế vĩ mô
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']
    X = df[features]

    # 2. Tính toán ma trận hệ số tương quan Pearson
    corr_matrix = X.corr().values

    print("==================================================")
    # In ra bảng số liệu trực tiếp trên màn hình Terminal/PowerShell để bạn xem trước
    print("BẢNG SỐ LIỆU MA TRẬN TƯƠNG QUAN CHÍNH XÁC:")
    print("==================================================")
    print(X.corr().round(4))
    print("==================================================\n")

    # 3. Vẽ biểu đồ Heatmap bằng Matplotlib thuần
    plt.figure(figsize=(9, 7))

    # Sử dụng bảng màu 'coolwarm' (Màu đỏ: tương quan dương mạnh, Màu xanh: tương quan âm mạnh)
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Thêm thanh thước màu đo cấp độ ở bên phải
    plt.colorbar(label='Hệ số tương quan (r)')

    # Cấu hình tên các trục
    ticks = np.arange(len(features))
    plt.xticks(ticks, features, rotation=45, fontsize=11)
    plt.yticks(ticks, features, fontsize=11)

    # Điền trực tiếp giá trị số lên từng ô vuông
    for i in range(len(features)):
        for j in range(len(features)):
            # Nếu hệ số lớn hơn 0.5 thì chữ màu trắng cho dễ nhìn trên nền đỏ, ngược lại chữ màu đen
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')

    plt.title('MA TRẬN TƯƠNG QUAN GIỮA CÁC BIẾN VĨ MÔ (MACRO)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # Lưu biểu đồ thành file ảnh riêng biệt
    plt.savefig('macro_correlation_matrix.png', dpi=300)
    print("-> Đã lưu biểu đồ ma trận tương quan vào file ảnh: 'macro_correlation_matrix.png'")

    # Hiển thị biểu đồ lên màn hình
    plt.show()


def scale_data():
    # 1. Đọc dữ liệu (giả định bạn đã có bước này ở trên)
    df = pd.read_csv('Data_bank_o.csv', encoding='latin1')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']
    X = df[features]

    # --- BƯỚC 1.2: CHUẨN HÓA DỮ LIỆU ---
    print("==================================================")
    print("          ĐANG TIẾN HÀNH CHUẨN HÓA DỮ LIỆU        ")
    print("==================================================")

    # Khởi tạo bộ scaler
    scaler = StandardScaler()

    # Tiến hành tính toán Mean, Std và ép dữ liệu về dạng chuẩn hóa (Z-score)
    X_scaled = scaler.fit_transform(X)

    # Chuyển mảng Numpy sau khi chuẩn hóa ngược lại thành DataFrame để dễ quan sát và kiểm tra
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    print("\n5 dòng dữ liệu đầu tiên SAU KHI CHUẨN HÓA:")
    print(X_scaled_df.head())

    print("\nKiểm tra lại Mean và Std sau khi chuẩn hóa (Mục tiêu: Mean ~ 0, Std ~ 1):")
    print(X_scaled_df.agg(['mean', 'std']).round(4))
    
    # SỬA ĐỔI: Thêm dòng return để chuyển tiếp dữ liệu X_scaled cho hàm Silhouette nhận
    return X_scaled


def find_optimal_k_silhouette(X_scaled):
    print("\n--- BƯỚC 1.3: ĐANG TÍNH SILHOUETTE SCORE CHO K TỪ 2 ĐẾN 8 ---")
    
    k_range = range(2, 9)
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Vì Silhouette tính toán trên dữ liệu lớn rất tốn thời gian,
        # ta sử dụng sample_size=10000 ngẫu nhiên để tính score nhanh và chuẩn xác.
        score = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
        silhouette_scores.append(score)
        print(f"Số cụm K = {k} | Silhouette Score = {score:.4f}")

    # Tìm K có điểm số cao nhất
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"-> Kết luận định lượng: Số cụm tối ưu nhất là K = {optimal_k}")

    # Vẽ biểu đồ đường Silhouette
    plt.figure(figsize=(9, 5))
    plt.plot(k_range, silhouette_scores, marker='o', linewidth=2, color='#2c3e50', markerfacecolor='#e74c3c', markersize=8)
    
    # Đánh dấu đỉnh cao nhất bằng một vòng tròn lớn hoặc đường kẻ
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
    
    # Đọc lại file dữ liệu gốc để chuẩn bị dán nhãn kết quả
    df = pd.read_csv('Data_bank_o.csv', encoding='latin1')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

    # Khởi tạo và fit mô hình K-Means với K=3, n_init=10 để tránh local minimum
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)
    
    # Lấy chỉ số Inertia cuối cùng
    final_inertia = kmeans_model.inertia_
    
    print(f"-> Thuật toán K-Means đã hội tụ thành công sau {kmeans_model.n_iter_} vòng lặp.")
    print(f"-> Giá trị Inertia (Tổng bình phương khoảng cách nội cụm) cuối cùng: {final_inertia:.4f}")
    
    # Dán nhãn cụm vào DataFrame gốc (để nhãn hiển thị từ Cụm 1, Cụm 2, Cụm 3 cho dễ đọc trong báo cáo)
    df['KMeans_Cluster'] = cluster_labels + 1 
    
    # Thống kê số lượng quan sát rơi vào mỗi cụm để kiểm tra độ cân bằng
    print("\nSố lượng quan sát trong mỗi cụm kinh tế:")
    print(df['KMeans_Cluster'].value_counts().sort_index())
    
    # Xuất bảng giá trị trung bình của từng cụm để phục vụ phân tích kinh tế
    print("\nBẢNG GIÁ TRỊ TRUNG BÌNH CỦA 4 BIẾN VĨ MÔ THEO 3 CỤM K-MEANS:")
    mean_table = df.groupby('KMeans_Cluster')[features].mean()
    print(mean_table.round(4))
    
    # Lưu toàn bộ dữ liệu đã dán nhãn ra file mới
    output_file = 'Data_bank_final_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n====== HOÀN TẤT! ĐÃ LƯU KẾT QUẢ PHÂN CỤM VÀO FILE '{output_file}' ======")
    
    return df

def AH_Clustering():
    # 1. Đọc dữ liệu từ file của bạn
    df = pd.read_csv('Data_bank_o.csv', encoding='latin1')

    # 2. Lọc ra 4 biến kinh tế vĩ mô
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

    # --- GIẢI PHÁP SỬA LỖI TRÀN RAM: LẤY MẪU NGẪU NHIÊN 1500 DÒNG ĐỂ VẼ DENDROGRAM ---
    sample_size = min(5500, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    X_sample = df_sample[features]

    # 3. Chuẩn hóa dữ liệu mẫu
    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)

    # --- BƯỚC 1: VẼ DENDROGRAM TRÊN DỮ LIỆU MẪU ---
    print(f"Đang tính toán trên tập mẫu {sample_size} dòng để tránh tràn RAM và vẽ Dendrogram...")
    plt.figure(figsize=(12, 7))

    # Tính toán ma trận liên kết trên tập mẫu đã chuẩn hóa
    Z = linkage(X_sample_scaled, method='ward')

    # Vẽ sơ đồ cây (chỉ hiện thị 20 nhánh cuối cùng cho đẹp mắt)
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=40, leaf_font_size=10, show_contracted=True)

    plt.title(f'Sơ đồ cây Phân cấp (Dendrogram) - Lấy mẫu {sample_size} dòng')
    plt.xlabel('Các nhánh/Cụm dữ liệu được gộp')
    plt.ylabel('Khoảng cách liên kết (Linkage Distance)')

    # Vẽ một đường cắt ngang để minh họa việc chia 3 cụm tự nhiên
    plt.axhline(y=85, color='r', linestyle='--', label='Đường cắt tạo ra 3 cụm tự nhiên')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dendrogram_plot.png')
    print("-> Đã lưu sơ đồ cây thành công vào file 'dendrogram_plot.png'")
    plt.show()

    # --- BƯỚC 2: CHẠY THUẬT TOÁN HIERARCHICAL TRÊN TẬP MẪU ĐỂ SO SÁNH VỚI K-MEANS ---
    agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    agg_labels = agg_clustering.fit_predict(X_sample_scaled)

    # Chạy K-Means trên cùng tập mẫu để so sánh tính nhất quán
    kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sample_scaled)

    # Đánh giá độ tương đồng trên tập mẫu
    ari_agg = adjusted_rand_score(kmeans_labels, agg_labels)
    nmi_agg = normalized_mutual_info_score(kmeans_labels, agg_labels)

    print("\n--- ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA K-MEANS VÀ HIERARCHICAL (TRÊN TẬP MẪU) ---")
    print(f"Adjusted Rand Index (ARI): {ari_agg:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi_agg:.4f}\n")

    # --- BƯỚC 3: THỐNG KÊ GIÁ TRỊ TRUNG BÌNH CỦA TẬP MẪU THEO HIERARCHICAL ---
    df_sample['Hierarchical_Cluster'] = agg_labels
    print("--- GIÁ TRỊ TRUNG BÌNH CỦA CÁC CỤM THEO HIERARCHICAL CLUSTERING ---")
    print(df_sample.groupby('Hierarchical_Cluster')[features].mean())

def profile_and_name_phases(X_scaled):
    print("\n==================================================")
    print("     BƯỚC 1.6: PROFILE VÀ ĐẶT TÊN GIAI ĐOẠN ĐỊNH LƯỢNG  ")
    print("==================================================")
    
    # 1. Đọc lại tệp dữ liệu gốc ban đầu để lấy giá trị đơn vị gốc
    df = pd.read_csv('Data_bank_o.csv', encoding='latin1')
    features = ['euribor3m', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate']

    # 2. Thực thi thuật toán K-Means với cấu hình K=3
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(X_scaled)
    df['Cluster_No'] = cluster_labels

    # Đặt tên giai đoạn đơn giản theo mức lãi suất thực tế của bạn
    # Cụm 0 (Lãi suất ~4.8) -> Lại suất cao
    # Cụm 1 (Lãi suất ~1.2) -> Lại suất trung bình
    # Cụm 2 (Lãi suất ~0.7) -> Lại suất thấp
    phase_mapping = {
        0: 'Tăng trưởng',
        1: 'Suy thoái',
        2: 'Phục hồi'
    }
    df['economic_phase'] = df['Cluster_No'].map(phase_mapping)

    # 3. MA TRẬN TÂM CỤM DẠNG Z-SCORE (DÙNG ĐỂ VẼ HEATMAP)
    cluster_centers_z = kmeans_model.cluster_centers_
    
    # Chuyển tâm cụm Z-score thành DataFrame để in ra màn hình cho bạn xem
    z_score_table = pd.DataFrame(cluster_centers_z, columns=features)
    # Ánh xạ tên giai đoạn tương ứng vào chỉ mục (Index)
    z_score_table.index = [phase_mapping[i] for i in z_score_table.index]
    z_score_table = z_score_table.reindex(['1. Lãi suất cao', '2. Lãi suất trung bình', '3. Lãi suất thấp'])

    print("\n[BẢNG 1] MA TRẬN TÂM CỤM ĐÃ CHUẨN HÓA (Z-SCORE) - DÙNG CHO HEATMAP:")
    print("====================================================================")
    print(z_score_table.round(4))
    print("====================================================================")

    # 4. TRỰC QUAN HÓA: Vẽ đồ thị Heatmap dựa trên tâm cụm Z-Score vừa in
    plt.figure(figsize=(9, 5.5))
    
    # annot=True sẽ điền chính xác con số Z-score vào từng ô màu
    sns.heatmap(z_score_table, annot=True, fmt=".2f", cmap="RdBu_r", cbar=True,
                linewidths=0.8, annot_kws={"size": 11, "weight": "bold"})
    
    plt.title('BIỂU ĐỒ HEATMAP PROFILE TÂM CỤM VĨ MÔ (Z-SCORE)', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Các giai đoạn kinh tế phát hiện', fontsize=10)
    plt.xlabel('Các biến số kinh tế vĩ mô', fontsize=10)
    plt.tight_layout()
    
    # Lưu đồ thị thành file ảnh để chèn vào Word
    plt.savefig('cluster_profile_heatmap.png', dpi=300)
    print("-> [XONG] Đã xuất biểu đồ Heatmap Z-score vào file: 'cluster_profile_heatmap.png'")

    # 5. IN BẢNG PROFILE THEO ĐƠN VỊ GỐC (ORIGINAL UNITS) ĐỂ BIỆN LUẬN
    print("\n[BẢNG 2] PROFILE CÁC BIẾN VĨ MÔ THEO ĐƠN VỊ GỐC (DỄ DIỄN GIẢI):")
    print("====================================================================")
    profile_table = df.groupby('economic_phase')[features].mean()
    profile_table = profile_table.reindex(['1. Lãi suất cao', '2. Lãi suất trung bình', '3. Lãi suất thấp'])
    print(profile_table.round(4))
    print("====================================================================")
    
    # Thống kê phân bổ mẫu
    print("\nSố lượng dòng quan sát rơi vào từng giai đoạn kinh tế:")
    print(df['economic_phase'].value_counts().sort_index())

    # Làm sạch cấu trúc và lưu file dữ liệu phẳng cuối cùng
    df = df.drop(columns=['Cluster_No'])
    output_filename = 'Data_bank_final_results.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n====== [HOÀN TẤT BƯỚC 1.6] ĐÃ GHI CỘT 'economic_phase' VÀO FILE '{output_filename}' ======")
    
    return df
# =====================================================================
# HÀM ĐIỀU KHIỂN CHÍNH (MAIN FUNCTION)
# =====================================================================
def main():
    print("==================================================")
    print("   CHẠY LUỒNG PHÂN TÍCH CHÍNH THỨC (QUY TRÌNH MỚI) ")
    print("==================================================")

    # 1. Khóa các bước EDA cũ (Nếu đã vẽ xong ảnh rồi thì không cần chạy lại)
    # histogram()
    # correlation_matrix()
    
    # 2. BẮT BUỘC GIỮ: Chuẩn hóa dữ liệu gốc để tạo nhiên liệu X_scaled
    X_scaled = scale_data()
    
    # 3. Khóa bước tìm K của Silhouette (Đã có ảnh silhouette_analysis.png rồi)
    # find_optimal_k_silhouette(X_scaled)
    
    # 4. BƯỚC 1.4: CHẠY K-MEANS CHÍNH THỨC VỚI K=3 TRƯỚC
    # Hàm này sẽ tính toán, in ra Inertia và xuất file 'Data_bank_final_results.csv'
    # df_final = run_kmeans_final(X_scaled)
    
    # 5. BƯỚC 1.5: CHẠY AH CLUSTERING PHÍA SAU ĐỂ ĐỐI CHIẾU
    # Hàm này sẽ lấy mẫu, vẽ Dendrogram và chạy Hierarchical Clustering
    # AH_Clustering()

    # 6. CHẠY CHÍNH THỨC BƯỚC 1.6 PROFILE & XUẤT DATASET TRUNG TÂM:
    df_final = profile_and_name_phases(X_scaled)

if __name__ == "__main__":
    main()