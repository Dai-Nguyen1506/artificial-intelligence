# Lộ trình Học tập Trí tuệ Nhân tạo Thực chiến - Giai đoạn 1: Toán học Ứng dụng

## Objective

Xây dựng nền tảng toán học cốt lõi (Đại số tuyến tính, Giải tích, Xác suất Thống kê) phục vụ trực tiếp cho việc biểu diễn dữ liệu, tối ưu hóa mô hình và xử lý chuỗi thời gian trong AI (đặc biệt cho dữ liệu tài chính/streaming).

## Current State

Đã hoàn thành hệ thống hóa toàn bộ tri thức Giai đoạn 1. Các khái niệm toán học hàn lâm đã được ánh xạ thành các công cụ kỹ thuật thực chiến trong tiền xử lý dữ liệu, cơ chế học của mạng Nơ-ron và đánh giá thống kê. Sẵn sàng làm nền tảng đầu vào để triển khai Giai đoạn 2 (Cốt lõi Deep Learning & Kiến trúc Mạng).

## Confirmed Knowledge

**1. Đại số tuyến tính (Biểu diễn & Nén dữ liệu)**

* **Cấu trúc dữ liệu Tensor:** Dữ liệu thực tế được đóng gói thành Tensor để tối ưu tính toán song song. Ví dụ hệ thống Market Data: Tensor 3 chiều `(Assets, Timesteps, Features)`. Việc cắt lát (slicing) rút ra ma trận 2D của từng tài sản.
* **Phép nhân ma trận:** $Y = XW + b$ là phép chiếu dữ liệu sang không gian đặc trưng mới để làm nổi bật tín hiệu.
* **Phân tích giá trị kỳ dị (SVD):** Thuật toán nén và khử nhiễu. Phân rã ma trận $A = U \Sigma V^T$.
* **Quy trình SVD thực chiến:**
1. *Chuẩn hóa (Standardization):* Bắt buộc dùng Z-score để đưa dữ liệu (ví dụ: các mã tài sản có thị giá chênh lệch lớn) về cùng thang đo trung bình 0, phương sai 1, tránh SVD bị thiên lệch.
2. *Lựa chọn điểm cắt $k$:* Dựa trên "Tỷ lệ phương sai được giải thích" (Explained Variance Ratio). Điểm cắt tối ưu (Elbow point) thường giữ khoảng 80-85% thông tin cốt lõi để loại bỏ nhiễu và tránh Overfitting.
3. *Ứng dụng:* * **Giảm chiều ($U_k \Sigma_k$):** Dùng cho mô hình Tree-based (XGBoost) để tránh Curse of Dimensionality.
* **Khử nhiễu ($U_k \Sigma_k V_k^T$):** Dùng cho phân tích chuỗi thời gian đơn lẻ hoặc vẽ biểu đồ, giữ nguyên số chiều nhưng làm mượt tín hiệu.

**2. Giải tích (Động cơ Tối ưu hóa)**

* **Hàm mất mát (Loss Function):** Lượng hóa sai số dự đoán. Ví dụ MSE: $L = \frac{1}{N} \sum_{i=1}^{N} (Y_{predict} - Y_{true})^2$.
* **Gradient Descent:** Cập nhật trọng số để tìm đáy Hàm Loss theo công thức: $W_{new} = W_{old} - \alpha \times \nabla L$.
* **Tốc độ học ($\alpha$ - Learning Rate):** Tham số sinh tử. Thực tế dùng giá trị nhỏ (0.001 - 0.01) để tránh hiện tượng văng quỹ đạo (divergence) do bề mặt Loss của không gian đa chiều rất gồ ghề.
* **Backpropagation & Ma trận Jacobian:** Ứng dụng quy tắc chuỗi (Chain Rule) để tính đạo hàm riêng (Ma trận Jacobian) ngược từ Output về Input, phân bổ mức độ "trách nhiệm" sai số cho từng trọng số để thực hiện cập nhật.

**3. Xác suất Thống kê (Đo lường Sự bất định)**

* **Phân phối dữ liệu:** Dữ liệu tài chính tuân theo "Phân phối đuôi dày" (Fat-Tailed), các sự kiện cực đoan xảy ra nhiều hơn lý thuyết Phân phối chuẩn. Cần lưu ý khi thiết lập ngưỡng Anomaly Detection.
* **Ước lượng MLE (Maximum Likelihood Estimation):** Nền tảng toán học của Hàm Loss. Tối đa hóa xác suất tạo ra dữ liệu quan sát được (MLE) tương đương với việc tối thiểu hóa hàm sai số (MSE) trong điều kiện nhiễu phân phối chuẩn.
* **Tính Dừng (Stationarity) trong Chuỗi thời gian:** Mô hình AI (như ARIMA) không thể học trực tiếp trên "Giá" tuyệt đối vì tính biến thiên của trung bình/phương sai (Non-stationary). Bắt buộc phải chuyển đổi thành chuỗi dừng bằng phương pháp Sai phân (Differencing), tức là dự báo "Tỷ suất sinh lời" (Returns).