Softmax Regression for Wine Quality Classification
Dự án này triển khai thuật toán Softmax Regression từ đầu (from scratch) bằng ngôn ngữ Python để phân loại chất lượng rượu vang dựa trên tập dữ liệu UCI Red Wine Quality. Dự án tập trung vào việc hiểu rõ cơ chế toán học của quá trình lan truyền xuôi (forward propagation), tính toán hàm mất mát (loss function) và tối ưu hóa bằng Gradient Descent.

📌 Tính năng chính
Triển khai nguyên bản (From Scratch): Không sử dụng các thư viện học máy cấp cao như Scikit-learn hay TensorFlow cho phần mô hình cốt lõi.

Tiền xử lý dữ liệu: Bao gồm chuẩn hóa dữ liệu (Standardization) và mã hóa nhãn (One-hot Encoding).

Thuật toán tối ưu: Sử dụng Gradient Descent để cập nhật trọng số và độ lệch (Bias).

Đánh giá mô hình: Tính toán độ chính xác (Accuracy) và theo dõi giá trị hàm Loss qua từng Epoch.

📂 Cấu trúc dự án
data/winequality-red.csv: Tập dữ liệu chứa 1599 mẫu rượu với 11 đặc trưng hóa học và 1 nhãn chất lượng.

utils.py: Chứa các hàm hỗ trợ đọc dữ liệu, chuẩn hóa đặc trưng và xử lý nhãn.

model.py: Định nghĩa hàm Softmax, quá trình huấn luyện mô hình và hàm dự đoán.

train.py: Tệp thực thi chính để kết nối dữ liệu và huấn luyện mô hình.

🚀 Kết quả huấn luyện
Mô hình đạt được kết quả khả quan sau 2000 vòng lặp (epochs) với tốc độ học (learning rate) 0.05:

Initial Loss: ~1.7918

Final Loss: ~0.9319

Training Accuracy: ~60.28%

🛠 Cách chạy dự án
Đảm bảo bạn đã cài đặt thư viện numpy.

Chạy lệnh sau trong terminal:

Bash
python train.py
