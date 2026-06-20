# Lộ trình Học tập Trí tuệ Nhân tạo Thực chiến - Giai đoạn 2: Nền tảng PyTorch & Tối ưu hóa

## Objective

Làm chủ nền tảng PyTorch, cơ chế hoạt động nội tại của quá trình huấn luyện mạng thần kinh nhân tạo và tối ưu hóa hệ thống. Chuyển tiếp từ tư duy mô hình "hộp đen" sang tư duy thiết kế phần mềm (OOP) để tự kiểm soát vòng đời huấn luyện mô hình Deep Learning.

## Current State

Đã hoàn thành phân tích lý thuyết cốt lõi và chạy thử nghiệm thành công pipeline huấn luyện PyTorch đầu tiên trên máy tính cá nhân. Đã hiểu bản chất của đồ thị tính toán, vòng lặp 5 bước, và tự tay thực hiện tinh chỉnh siêu tham số (Hyperparameter Tuning) để đưa mô hình từ trạng thái Underfitting tới mức hội tụ hoàn hảo.

## Confirmed Knowledge

* **Vòng lặp Huấn luyện Chuẩn (5 Bước):**
1. Forward Pass: `y_pred = model(x)`
2. Tính Loss: `loss = criterion(y_pred, y_true)`
3. Xóa Gradient: `optimizer.zero_grad()`
4. Lan truyền ngược (Autograd): `loss.backward()`
5. Cập nhật Trọng số: `optimizer.step()`


* **Tinh chỉnh Siêu tham số (Khắc phục Underfitting):** Khi Loss đang giảm nhưng mô hình chưa đạt kết quả tối ưu, cần cung cấp thêm thời gian hoặc tăng tốc độ hội tụ bằng cách tăng số chu kỳ (`Epochs`) hoặc tăng Tốc độ học (`Learning Rate - lr`). Đã kiểm chứng thành công với `Epochs=100` và `lr=0.1`.
* **Bản chất Kiến trúc nn.Linear (Dense Layer):** Tạo kết nối đầy đủ (Fully Connected). Đại diện cho phương trình $y = xW^T + b$. Mô hình tự tìm ra quy luật bằng cách vặn chỉnh các tham số $W$ và $b$ lọc qua nhiễu (noise). Tổng tham số = `(in_features * out_features) + out_features`.
* **Quản lý Bộ nhớ & Tối ưu hóa:**
* Dùng `loss.item()` khi tích lũy loss để cắt đứt đồ thị tính toán, ngăn chặn rò rỉ RAM/VRAM.
* Khi đánh giá/suy luận (Inference), luôn bọc trong context manager `with torch.no_grad():` để vô hiệu hóa Autograd.


* **Chiến lược Phần cứng (Intel Core Ultra 7):** Trong Giai đoạn 2, sử dụng **CPU** (`.to('cpu')`) để huấn luyện nhằm đảm bảo tính ổn định và tránh lỗi môi trường thư viện. NPU chỉ được thiết kế cho quá trình Inference sinh thái thấp, sẽ áp dụng ở Giai đoạn 4 (MLOps).
* **Thiết kế Luồng Dữ liệu (OOP):**
* **Dataset:** Kế thừa `torch.utils.data.Dataset`, ghi đè `__init__`, `__len__`, và `__getitem__`.
* **DataLoader:** Bọc Dataset để xử lý Batching, Shuffle và Multiprocessing.



### Open Questions

* Cách triển khai thuật toán Gradient Descent và ma trận Jacobian bằng thuần NumPy để bóc tách lớp toán học ngầm của PyTorch.
* Thiết kế kiến trúc `Custom Dataset` thực chiến kết hợp kỹ thuật trượt cửa sổ thời gian (sliding window) cho dữ liệu chuỗi (Sequential Data/Time-Series).
* Cách áp dụng hàm `pack_padded_sequence` để xử lý các batch dữ liệu chuỗi có độ dài không đồng đều.