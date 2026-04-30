import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ============================================================
# BƯỚC 1: ĐỌC DỮ LIỆU VÀ CHIA TRAIN/TEST
# ============================================================
def load_data(dataset_path, n_train=7):
    """
    Đọc toàn bộ ảnh từ thư mục AT&T, chia thành tập Train và Test.
    
    Tham số:
        dataset_path : đường dẫn tới thư mục AT&T (chứa s1, s2, ..., s40)
        n_train      : số ảnh mỗi người dùng để train (mặc định 7, còn lại 3 để test)
    
    Trả về:
        train_X : ma trận ảnh train, mỗi hàng là 1 vector ảnh  (280 x 10304)
        train_y : nhãn tương ứng của tập train                  (280,)
        test_X  : ma trận ảnh test                               (120 x 10304)
        test_y  : nhãn tương ứng của tập test                    (120,)
    """
    train_X, train_y = [], []
    test_X, test_y = [], []

    # Duyệt qua 40 người (thư mục s1 → s40)
    for person_id in range(1, 41):
        folder = os.path.join(dataset_path, f"s{person_id}")

        # Duyệt qua 10 ảnh của mỗi người (1.pgm → 10.pgm)
        for img_idx in range(1, 11):
            img_path = os.path.join(folder, f"{img_idx}.pgm")

            # Đọc ảnh grayscale bằng cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Vector hóa: duỗi ảnh 112x92 thành vector 10304 chiều
            img_vector = img.flatten().astype(np.float64)

            # Chia train/test: 7 ảnh đầu → train, 3 ảnh sau → test
            if img_idx <= n_train:
                train_X.append(img_vector)
                train_y.append(person_id)
            else:
                test_X.append(img_vector)
                test_y.append(person_id)

    # Chuyển list thành numpy array
    train_X = np.array(train_X)    # (280, 10304)
    train_y = np.array(train_y)    # (280,)
    test_X = np.array(test_X)      # (120, 10304)
    test_y = np.array(test_y)      # (120,)

    print(f"[Bước 1] Load xong!")
    print(f"  Train: {train_X.shape[0]} ảnh | Test: {test_X.shape[0]} ảnh")
    print(f"  Kích thước mỗi vector ảnh: {train_X.shape[1]} chiều")
    return train_X, train_y, test_X, test_y


# ============================================================
# BƯỚC 2: XÂY DỰNG KHÔNG GIAN PCA (Eigenfaces)
# ============================================================
def compute_pca(X):
    """
    Tính PCA trên tập dữ liệu train.
    
    Quy trình:
        1. Tính mean face (khuôn mặt trung bình)
        2. Chuẩn hóa: trừ mỗi ảnh cho mean face
        3. Dùng trick ma trận nhỏ (A*A^T thay vì A^T*A) để tính eigenvectors
        4. Sắp xếp eigenvectors theo eigenvalue giảm dần
    
    Tham số:
        X : ma trận ảnh train (n_samples x n_pixels), ví dụ (280 x 10304)
    
    Trả về:
        mean_face          : vector trung bình (10304,)
        X_normalized       : ma trận đã chuẩn hóa (280 x 10304)
        sorted_eigenvectors: eigenvectors đã sắp xếp (10304 x n_samples)
    """
    n_samples = X.shape[0]     # Số ảnh train = 280

    # --- 2a. Tính Mean Face ---
    # Cộng tất cả vector ảnh lại, chia cho số ảnh
    mean_face = np.mean(X, axis=0)    # (10304,)

    # --- 2b. Chuẩn hóa: trừ mean face ---
    # Mỗi ảnh trừ đi khuôn mặt trung bình → đưa dữ liệu về tâm 0
    X_normalized = X - mean_face      # (280, 10304)

    # --- 2c. Tính eigenvectors bằng TRICK MA TRẬN NHỎ ---
    #
    # Lý thuyết: Ma trận hiệp phương sai C = A^T * A có kích thước (10304 x 10304) → QUÁ LỚN
    # Trick:     Tính S = A * A^T có kích thước chỉ (280 x 280) → rất nhỏ, tính nhanh
    #            Sau đó chuyển eigenvectors của S về eigenvectors thật bằng: u = A^T * v
    #
    S = np.dot(X_normalized, X_normalized.T)    # (280 x 280) - ma trận nhỏ

    # Tính eigenvalues và eigenvectors của ma trận nhỏ S
    eigenvalues, eigenvectors_small = np.linalg.eigh(S)

    # Chuyển eigenvectors từ không gian nhỏ (280 chiều) về không gian lớn (10304 chiều)
    # Công thức: eigenvector_thật = A^T × eigenvector_nhỏ
    eigenvectors_large = np.dot(X_normalized.T, eigenvectors_small)    # (10304 x 280)

    # Chuẩn hóa (normalize) mỗi eigenvector về độ dài = 1
    for i in range(eigenvectors_large.shape[1]):
        norm = np.linalg.norm(eigenvectors_large[:, i])
        if norm > 0:
            eigenvectors_large[:, i] = eigenvectors_large[:, i] / norm

    # --- 2d. Sắp xếp theo eigenvalue GIẢM DẦN ---
    # eigh() trả eigenvalues tăng dần → đảo ngược
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors_large[:, sorted_indices]    # (10304 x 280)

    print(f"[Bước 2] PCA hoàn tất!")
    print(f"  Mean face shape: {mean_face.shape}")
    print(f"  Số eigenvectors: {sorted_eigenvectors.shape[1]}")
    return mean_face, X_normalized, sorted_eigenvectors


# ============================================================
# BƯỚC 3: GIẢM CHIỀU DỮ LIỆU (GIỮ LẠI k THÀNH PHẦN)
# ============================================================
def extract_features(X_normalized, sorted_eigenvectors, k):
    """
    Giảm chiều bằng cách chỉ giữ lại k eigenfaces lớn nhất.
    
    Tham số:
        X_normalized       : ma trận đã chuẩn hóa (280 x 10304)
        sorted_eigenvectors: tất cả eigenvectors (10304 x 280)
        k                  : số thành phần chính cần giữ
    
    Trả về:
        eigenfaces_k   : k eigenfaces đầu tiên (10304 x k)
        train_weights  : trọng số tập train trong không gian k chiều (280 x k)
    """
    # Lấy k eigenvectors đầu tiên (chứa nhiều thông tin nhất)
    eigenfaces_k = sorted_eigenvectors[:, :k]    # (10304, k)

    # Chiếu tập train vào không gian k chiều
    # Mỗi ảnh 10304 chiều → giảm còn k chiều
    train_weights = np.dot(X_normalized, eigenfaces_k)    # (280, k)

    return eigenfaces_k, train_weights


# ============================================================
# BƯỚC 4: NHẬN DẠNG KHUÔN MẶT
# ============================================================
def recognize_face(test_image, mean_face, eigenfaces_k, train_weights, train_labels):
    """
    Nhận dạng 1 ảnh test bằng khoảng cách Euclid.
    
    Quy trình:
        1. Chuẩn hóa ảnh test (trừ mean_face)
        2. Chiếu vào không gian PCA
        3. Tính khoảng cách Euclid đến mọi ảnh train
        4. Chọn ảnh gần nhất
    
    Tham số:
        test_image    : vector ảnh test (10304,)
        mean_face     : khuôn mặt trung bình (10304,)
        eigenfaces_k  : k eigenfaces (10304 x k)
        train_weights : trọng số tập train (280 x k)
        train_labels  : nhãn tập train (280,)
    
    Trả về:
        predicted_label : nhãn người được dự đoán
    """
    # Bước 4.1: Chuẩn hóa ảnh test
    test_normalized = test_image - mean_face

    # Bước 4.2: Chiếu vào không gian PCA → lấy vector trọng số
    test_weight = np.dot(test_normalized, eigenfaces_k)    # (k,)

    # Bước 4.3: Tính khoảng cách Euclid đến TẤT CẢ ảnh train
    # Công thức: distance = sqrt( sum( (a - b)^2 ) )
    distances = np.linalg.norm(train_weights - test_weight, axis=1)    # (280,)

    # Bước 4.4: Tìm ảnh train có khoảng cách NHỎ NHẤT
    min_index = np.argmin(distances)
    predicted_label = train_labels[min_index]

    return predicted_label


# ============================================================
# BƯỚC 5: ĐÁNH GIÁ HIỆU SUẤT VÀ VẼ BIỂU ĐỒ
# ============================================================
def evaluate_accuracy(train_X, train_y, test_X, test_y, k_values):
    """
    Đánh giá accuracy với nhiều giá trị k khác nhau, rồi vẽ biểu đồ.
    
    Tham số:
        train_X  : ma trận ảnh train (280 x 10304)
        train_y  : nhãn train (280,)
        test_X   : ma trận ảnh test (120 x 10304)
        test_y   : nhãn test (120,)
        k_values : danh sách các giá trị k cần thử, ví dụ [5, 10, 20, ...]
    """
    # Chạy PCA một lần duy nhất (vì không phụ thuộc k)
    mean_face, X_normalized, sorted_eigenvectors = compute_pca(train_X)

    accuracies = []

    for k in k_values:
        # Giảm chiều với k thành phần
        eigenfaces_k, train_weights = extract_features(X_normalized, sorted_eigenvectors, k)

        # Đếm số ảnh test nhận đúng
        correct = 0
        for i in range(len(test_X)):
            predicted = recognize_face(
                test_X[i], mean_face, eigenfaces_k, train_weights, train_y
            )
            if predicted == test_y[i]:
                correct += 1

        # Tính accuracy
        accuracy = correct / len(test_y) * 100
        accuracies.append(accuracy)
        print(f"  k = {k:3d}  →  Accuracy = {accuracy:.2f}%  ({correct}/{len(test_y)} đúng)")

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Principal Components (k)', fontsize=13)
    plt.ylabel('Accurracy (%)', fontsize=13)
    plt.title('PCA Face Recognition - Accuracy vs k', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)

    # Ghi giá trị accuracy lên mỗi điểm
    for i, (k, acc) in enumerate(zip(k_values, accuracies)):
        plt.annotate(f'{acc:.1f}%', (k, acc),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('accuracy_vs_k.png', dpi=150)
    plt.show()
    print(f"\n[Bước 5] Biểu đồ đã lưu: accuracy_vs_k.png")

    return accuracies


# ============================================================
# BONUS: HIỂN THỊ MEAN FACE VÀ EIGENFACES
# ============================================================
def show_eigenfaces(mean_face, sorted_eigenvectors, img_shape=(112, 92)):
    """
    Hiển thị Mean Face và 10 Eigenfaces đầu tiên.
    Giúp trực quan hóa kết quả PCA.
    """
    plt.figure(figsize=(14, 4))

    # Hiển thị Mean Face
    plt.subplot(2, 6, 1)
    plt.imshow(mean_face.reshape(img_shape), cmap='gray')
    plt.title('Mean Face', fontsize=9)
    plt.axis('off')

    # Hiển thị 10 Eigenfaces đầu tiên
    for i in range(10):
        plt.subplot(2, 6, i + 2)
        eigenface = sorted_eigenvectors[:, i].reshape(img_shape)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'EF {i+1}', fontsize=9)
        plt.axis('off')

    plt.suptitle('Mean Face and 10 Eigenfaces ', fontsize=13)
    plt.tight_layout()
    plt.savefig('eigenfaces.png', dpi=150)
    plt.show()
    print("[Bonus] Eigenfaces đã lưu: eigenfaces.png")


# ============================================================
# HÀM MAIN: CHẠY TOÀN BỘ CHƯƠNG TRÌNH
# ============================================================
if __name__ == "__main__":
    # === CẤU HÌNH ===
    dataset_path = "ATnT"     # Đường dẫn tới thư mục dataset
    n_train = 7               # Số ảnh train mỗi người (còn lại = test)

    # === BƯỚC 1: LOAD DỮ LIỆU ===
    print("=" * 50)
    print("BƯỚC 1: ĐỌC DỮ LIỆU")
    print("=" * 50)
    train_X, train_y, test_X, test_y = load_data(dataset_path, n_train)

    # === BƯỚC 2: TÍNH PCA ===
    print("\n" + "=" * 50)
    print("BƯỚC 2: TÍNH PCA VÀ EIGENFACES")
    print("=" * 50)
    mean_face, X_normalized, sorted_eigenvectors = compute_pca(train_X)

    # === BONUS: HIỂN THỊ EIGENFACES ===
    print("\n" + "=" * 50)
    print("BONUS: HIỂN THỊ EIGENFACES")
    print("=" * 50)
    show_eigenfaces(mean_face, sorted_eigenvectors)

    # === BƯỚC 3-4-5: ĐÁNH GIÁ VỚI NHIỀU GIÁ TRỊ k ===
    print("\n" + "=" * 50)
    print("BƯỚC 3-4-5: ĐÁNH GIÁ ACCURACY VỚI NHIỀU k")
    print("=" * 50)
    k_values = [5, 10, 20, 30, 50, 80, 100, 150, 200]
    accuracies = evaluate_accuracy(train_X, train_y, test_X, test_y, k_values)

    # === TÓM TẮT KẾT QUẢ ===
    print("\n" + "=" * 50)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 50)
    best_idx = np.argmax(accuracies)
    print(f"  Accuracy cao nhất: {accuracies[best_idx]:.2f}% tại k = {k_values[best_idx]}")