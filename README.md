# 🧠 Face Recognition with PCA (Eigenfaces)

Nhận dạng khuôn mặt sử dụng thuật toán **PCA (Principal Component Analysis)** — hay còn gọi là **Eigenfaces** — trên tập dữ liệu AT&T (ORL).

## 📌 Giới thiệu

Dự án này xây dựng hệ thống nhận dạng khuôn mặt từ đầu (from scratch) bằng Python, không dùng thư viện ML có sẵn. Toàn bộ pipeline PCA được implement thủ công bằng NumPy.

## 🗂 Dataset

- **AT&T Face Database** (ORL): 40 người × 10 ảnh = 400 ảnh grayscale (112×92 px)
- Chia: 7 ảnh/người → train (280 ảnh), 3 ảnh/người → test (120 ảnh)

## ⚙️ Pipeline

1. **Load data** — đọc ảnh `.pgm`, vector hóa thành mảng 10304 chiều
2. **PCA** — tính mean face, chuẩn hóa, dùng trick ma trận nhỏ (A·Aᵀ) để tính eigenvectors hiệu quả
3. **Feature extraction** — chiếu ảnh vào không gian k chiều (Eigenfaces)
4. **Recognition** — so sánh khoảng cách Euclid với tập train, chọn nearest neighbor
5. **Evaluation** — đánh giá accuracy với nhiều giá trị k khác nhau

## 📊 Kết quả

| k (số thành phần) | Accuracy |
|---|---|
| 5 | ~80% |
| 50 | ~95%+ |
| 150 | ~96%+ |

Xem biểu đồ: `accuracy_vs_k.png`

## 🖼 Visualizations

- `eigenfaces.png` — Mean Face + 10 Eigenfaces đầu tiên
- `accuracy_vs_k.png` — Biểu đồ accuracy theo số thành phần k

## 🚀 Chạy thử

```bash
pip install numpy opencv-python matplotlib
python pca.py
```

> Đặt thư mục dataset AT&T vào cùng cấp với `pca.py`, đổi tên thành `ATnT/`

## 📦 Requirements

```
numpy
opencv-python
matplotlib
```
