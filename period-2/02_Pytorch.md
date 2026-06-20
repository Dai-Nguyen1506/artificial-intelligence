# Lộ trình Học tập Trí tuệ Nhân tạo Thực chiến - Giai đoạn 2: Nền tảng PyTorch

## Objective

Làm chủ nền tảng PyTorch và cơ chế hoạt động nội tại của quá trình huấn luyện mạng thần kinh nhân tạo. Chuyển tiếp từ tư duy mô hình "hộp đen" (black box) sang tư duy thiết kế phần mềm (OOP) để kiểm soát, tối ưu hóa và tự xây dựng toàn bộ vòng đời huấn luyện mô hình Deep Learning.

## Current State

Đã phân tích và nắm vững bản chất 4 mảnh ghép cốt lõi của PyTorch (Dataset/DataLoader, Hardware Management, Model Architecture, Training Loop). Đã làm rõ sự khác biệt giữa cấu trúc code học thuật cơ bản và thiết lập thực chiến. Đã chốt phương án sử dụng phần cứng tối ưu cho việc thực hành trên máy tính cá nhân (Intel Core Ultra 7).

## Confirmed Knowledge

**1. Cơ chế Autograd & Computation Graph**

* PyTorch theo dõi các phép toán trên Tensor (`requires_grad=True`) để xây dựng Đồ thị tính toán có hướng (DAG).
* Quá trình lan truyền ngược (Backward pass) tính toán vector gradient thông qua quy tắc chuỗi (kế thừa từ tính toán ma trận Jacobian), cho phép tự động điều chỉnh tham số mô hình để giảm sai số.

**2. Vòng lặp Huấn luyện Chuẩn (5 Bước)**

1. **Forward Pass:** `y_pred = model(x)` (Đưa dữ liệu qua mô hình lấy dự đoán).
2. **Tính Loss:** `loss = criterion(y_pred, y_true)` (Đo lường sai số).
3. **Xóa Gradient:** `optimizer.zero_grad()` (Bắt buộc để tránh lỗi cộng dồn gradient từ batch trước).
4. **Lan truyền ngược:** `loss.backward()` (Tính toán đạo hàm).
5. **Cập nhật Trọng số:** `optimizer.step()` (Điều chỉnh mô hình theo gradient).

**3. Quản lý Bộ nhớ & Tối ưu hóa (Hardware Context)**

* **Đồng bộ:** Data và Model bắt buộc phải nằm trên cùng một thiết bị thông qua câu lệnh `.to(device)`.
* **Chống rò rỉ bộ nhớ:** Khi log giá trị loss theo epoch, luôn dùng `loss.item()` để trích xuất giá trị thuần túy, cắt đứt sự lưu trữ đồ thị tính toán gây tràn RAM/VRAM.
* **Inference/Đánh giá:** Luôn sử dụng context manager `with torch.no_grad():` và chuyển trạng thái `model.eval()` để vô hiệu hóa autograd, tiết kiệm bộ nhớ và tăng tốc.
* **Chiến lược Phần cứng cá nhân (Intel Core Ultra 7):** Trong Giai đoạn 2 (tập trung xây dựng kiến trúc từ đầu), ưu tiên huấn luyện bằng **CPU** (`.to('cpu')`) để đảm bảo tính ổn định và tránh lỗi môi trường. NPU chỉ thiết kế cho Inference (sẽ dùng với OpenVINO ở Giai đoạn 4 MLOps). iGPU (IPEX) thiết lập phức tạp, tạm thời bỏ qua.

**4. Thiết kế Luồng Dữ liệu (Dataset & DataLoader)**

* **Dataset (OOP):** Kế thừa `torch.utils.data.Dataset`. Bắt buộc ghi đè 3 hàm: `__init__` (khởi tạo dữ liệu), `__len__` (tổng số lượng mẫu), và `__getitem__` (logic lấy 1 mẫu dữ liệu đơn lẻ).
* **DataLoader:** Bọc lớp Dataset để quản lý batching (gộp batch), shuffle (xáo trộn), và multiprocessing (nạp song song), giúp đẩy dữ liệu liên tục không làm nghẽn mô hình.

**5. Bản chất Kiến trúc nn.Linear (Dense Layer)**

* Tạo kết nối đầy đủ (Fully Connected) giữa mọi node đầu vào với mọi node đầu ra.
* Phương trình toán học: $y = xW^T + b$ (Tìm kiếm quy luật thông qua việc vặn chỉnh $W$ và $b$).
* **Tổng tham số cần học:** Bằng `(in_features * out_features) + out_features`. Bao gồm Ma trận trọng số ($W$) kích thước `in x out` và Vector độ lệch ($b$) kích thước `out`.

**6. Cấu trúc PyTorch Thực chiến**

* Mô hình không phải là lệnh đơn lẻ mà phải là một Class kế thừa `nn.Module`, thiết lập các thành phần ở `__init__` và định nghĩa dòng chảy tensor ở hàm `forward()`.
* Vòng lặp huấn luyện thực chiến cần chia các khối độc lập: Train loop, Validation/Evaluation loop, và cơ chế lưu trữ mô hình tốt nhất (Checkpoint: `torch.save(model.state_dict())`).