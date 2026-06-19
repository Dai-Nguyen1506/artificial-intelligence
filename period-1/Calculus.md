# Giải Tích Ứng Dụng Trong Trí Tuệ Nhân Tạo

## Objective

Xây dựng nền tảng giải tích cốt lõi (Đạo hàm, Gradient Descent, Backpropagation) đóng vai trò làm động cơ tối ưu hóa, giúp mô hình học máy đo lường sai số và tự động tinh chỉnh ma trận trọng số trong môi trường dữ liệu đa chiều.

## Current State

Đã hoàn thành chưng cất cơ chế hoạt động của Giải tích trong AI. Các khái niệm đạo hàm đa biến và quy tắc chuỗi đã được ánh xạ thành quy trình lan truyền ngược thực chiến, xác định rõ bản chất của ma trận Jacobian và chiến lược thiết lập tham số Tốc độ học (Learning Rate) an toàn cho các bề mặt hàm mất mát thực tế.

## Confirmed Knowledge

* **Hàm Mất Mát (Loss Function):** * Công cụ lượng hóa độ lệch giữa dự đoán của hệ thống và thực tế.
* Hàm cơ bản (ví dụ MSE): $L = \frac{1}{N} \sum_{i=1}^{N} (Y_{predict} - Y_{true})^2$. Mục tiêu tối thượng của mô hình là tìm ra ma trận trọng số $W$ để $L$ đạt giá trị cực tiểu toàn cục hoặc cục bộ.


* **Đạo Hàm và Gradient Descent (Giảm dốc):**
* Đạo hàm ($\nabla L$) xác định hướng và độ dốc của bề mặt hàm mất mát tại vị trí hiện tại.
* Thuật toán Gradient Descent điều hướng mô hình di chuyển ngược chiều đạo hàm để tìm đáy.
* Công thức cập nhật trọng số: $W_{new} = W_{old} - \alpha \times \nabla L$.


* **Tốc độ học ($\alpha$ - Learning Rate):**
* Tham số kiểm soát kích thước bước nhảy trong quá trình Gradient Descent.
* *Thực tiễn cấu hình:* Trong không gian thực tế đa chiều phức tạp và gồ ghề của mạng Nơ-ron, $\alpha$ tuyệt đối không được đặt ở mức lý tưởng toán học học thuật (như $0.5$) vì sẽ gây ra hiện tượng nhảy vọt qua đáy, văng quỹ đạo (divergence) và làm Loss trở thành `NaN`. Giá trị an toàn và phổ biến để bắt đầu thường rất nhỏ, ở mức $0.001$ hoặc $0.01$.


* **Quy tắc chuỗi (Chain Rule) & Ma trận Jacobian:**
* Quy tắc chuỗi đa biến là cốt lõi để tính toán đạo hàm liên hoàn qua nhiều lớp ẩn (Hidden Layers).
* Ma trận Jacobian tập hợp toàn bộ các đạo hàm riêng bậc một của mạng, lưu trữ thông tin về mức độ ảnh hưởng trực tiếp (tỷ lệ phần trăm lỗi) của từng nơ-ron lên kết quả đầu ra cuối cùng.


* **Lan truyền ngược (Backpropagation):**
* Là quy trình thực thi thực tế của hệ thống: Tính toán ma trận Jacobian từ lớp đầu ra (Output/Loss) và nhân lùi dần về lớp đầu vào (Input).
* Bản chất là quá trình "truy cứu trách nhiệm", phân bổ chính xác độ sai lệch (Gradient) cho từng trọng số $W$ trên toàn mạng lưới, làm cơ sở kích hoạt lệnh cập nhật Gradient Descent cho chu kỳ học tiếp theo.