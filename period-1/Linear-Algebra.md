# Đại số Tuyến tính Ứng dụng trong Trí tuệ Nhân tạo

## Objective

Xây dựng nền tảng đại số tuyến tính cốt lõi phục vụ trực tiếp cho việc biểu diễn dữ liệu, biến đổi đặc trưng và giảm chiều/khử nhiễu dữ liệu trong 파ipeline của các hệ thống AI và học máy.

## Current State

Đã hoàn thành hệ thống hóa các tri thức nền tảng về cấu trúc dữ liệu đa chiều, phép biến đổi tuyến tính và thuật toán phân rã ma trận. Các khái niệm toán học đã được chuyển hóa thành phương pháp xử lý dữ liệu thực chiến, sẵn sàng tích hợp làm đầu vào cho các mô hình dự báo.

## Confirmed Knowledge

* **Cấu trúc Dữ liệu Đa chiều (Tensor):**
* Quá trình số hóa dữ liệu thực tế yêu cầu đóng gói thông tin vào các không gian: Vô hướng (0D), Vector (1D), Ma trận (2D), và Tensor (3D trở lên).
* *Ứng dụng cấu trúc:* Đóng gói dữ liệu chuỗi thời gian đồng thời của nhiều đối tượng (ví dụ: thị trường tài chính) thành Tensor 3 chiều với định dạng `(Batch/Assets, Sequence/Timesteps, Features)`. Cấu trúc này tối ưu hóa việc truy xuất dữ liệu (slicing) thành các ma trận 2D độc lập và cho phép GPU thực hiện tính toán biến đổi song song cho toàn bộ batch.


* **Phép nhân Ma trận (Matrix Multiplication):**
* Công thức cốt lõi: $Y = XW + b$.
* Bản chất là phép biến đổi đặc trưng, chiếu dữ liệu từ không gian ban đầu ($d$ chiều) sang một không gian mới ($k$ chiều) do mô hình tự động tinh chỉnh qua các trọng số, nhằm làm nổi bật các tín hiệu quan trọng nhất.


* **Đo lường Khoảng cách Không gian:**
* *Khoảng cách Euclidean (L2 Norm):* Đo khoảng cách tuyệt đối, nhạy cảm với biên độ giá trị.
* *Độ tương đồng Cosine:* $\text{Cosine}(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$. Đánh giá mức độ đồng pha về "hướng" của các chuỗi dữ liệu (ví dụ: xu hướng cùng tăng/giảm) mà không bị nhiễu bởi biên độ hoặc quy mô tuyệt đối.


* **Phân rã Giá trị Kỳ dị (SVD - Singular Value Decomposition):**
* Công thức phân rã mọi ma trận: $A = U \Sigma V^T$. Bóc tách ma trận dữ liệu thành các xu hướng ẩn (Singular Vectors) và sức mạnh của các xu hướng đó (Singular Values).
* **Quy trình SVD Thực chiến:**
1. *Chuẩn hóa (Standardization):* Áp dụng Z-score để đưa toàn bộ đặc trưng về cùng thang đo (trung bình 0, phương sai 1), ngăn chặn các đặc trưng có giá trị tuyệt đối lớn chi phối thuật toán.
2. *Lựa chọn điểm cắt $k$ (Truncation):* Phân tích "Tỷ lệ phương sai được giải thích" (Explained Variance Ratio) để tìm điểm cùi chỏ (Elbow point). Thường giữ lại khoảng 80-85% phương sai (3-5 giá trị kỳ dị lớn nhất) để bắt được xu hướng cốt lõi và vứt bỏ phần đuôi chứa nhiễu vi mô, giúp tránh hiện tượng Overfitting.
3. *Chiến lược đầu ra:*
* **Giảm chiều dữ liệu ($U_k \Sigma_k$):** Nén ma trận xuống số chiều $k$ nhỏ hơn rất nhiều so với ban đầu. Sử dụng làm đầu vào cho các mô hình Tree-based hoặc Neural Networks để giải quyết Lời nguyền số chiều (Curse of Dimensionality) và tăng tốc độ huấn luyện.
* **Khử nhiễu ($U_k \Sigma_k V_k^T$):** Tái tạo lại ma trận về đúng cấu trúc số chiều ban đầu nhưng với tín hiệu đã được làm mượt. Phù hợp cho trực quan hóa dữ liệu hoặc làm đầu vào cho các mô hình dự báo chuỗi thời gian đơn lẻ.