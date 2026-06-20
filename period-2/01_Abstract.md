# Lộ trình Học tập Trí tuệ Nhân tạo Thực chiến - Trọng tâm Giai đoạn 2

## Objective

Xây dựng một lộ trình học tập AI chuyên sâu, thực chiến và toàn diện (tổng 6-10 tháng), định hướng từ nền tảng máy học cổ điển, cốt lõi deep learning, kỹ năng chuyên sâu (NLP/Time-Series) đến MLOps. Trọng tâm hiện tại là làm chủ Giai đoạn 2: Cốt lõi Deep Learning & Kiến trúc Mạng.

## Current State

Lộ trình tổng thể đã được xác định qua 4 giai đoạn. Người học đã hoàn thành Giai đoạn 1 và đang tập trung giải quyết Giai đoạn 2. Lý thuyết Giai đoạn 2 đã được chưng cất thành một danh sách các kỹ năng thực hành (checklist) bắt buộc cần đạt được, xoay quanh PyTorch, Computer Vision (CNN) và Sequential Data (RNN/LSTM).

## Confirmed Knowledge

**Lộ trình Tổng quát (4 Giai đoạn):**

1. **Machine Learning Nâng cao (1-2 tháng):** Toán tối ưu, Tree-based models (XGBoost, LightGBM), Time-series cổ điển (ARIMA), Giảm chiều dữ liệu (SVD, PCA).
2. **Cốt lõi Deep Learning (2-3 tháng):** Đồ thị tính toán, PyTorch cơ bản, kiến trúc CNN và RNN.
3. **Chuyên ngành (2-3 tháng):** NLP (Transformer, LLM fine-tuning, RAG) hoặc Dữ liệu Streaming (Reinforcement Learning, Time-Series Transformers).
4. **MLOps (1-2 tháng):** MLflow/W&B, FastAPI, Docker, CI/CD, Feature Stores.

**Checklist Thực hành Bắt buộc - Giai đoạn 2:**

* **Nền tảng PyTorch & Tối ưu hóa:**
* Tự xây dựng vòng lặp huấn luyện chuẩn: Cập nhật Forward, Loss, `loss.backward()`, `optimizer.step()`, và `optimizer.zero_grad()`.
* Hiểu cơ chế Autograd, Computation Graph và ma trận Jacobian (có thể tự viết lại Gradient Descent và Linear Layer bằng NumPy).
* Quản lý bộ nhớ GPU tối ưu (`.to('cuda')`, `with torch.no_grad():`).
* Xây dựng Custom Dataset (kế thừa `torch.utils.data.Dataset`) và DataLoader để quản lý batching.


* **Kiến trúc Không gian (CNN):**
* Nắm vững cơ chế hoạt động và tham số của `nn.Conv2d`, `nn.MaxPool2d`, `nn.BatchNorm2d`.
* Tự lập trình kiến trúc ResNet (đặc biệt là Residual Block - kết nối tắt) từ đầu bằng PyTorch, không dùng thư viện có sẵn.
* Sử dụng mô hình pre-trained YOLO (ví dụ YOLOv8) để thực hiện suy luận (inference) phát hiện vật thể (bounding box) theo thời gian thực.


* **Kiến trúc Chuỗi (RNN/LSTM/GRU):**
* Hiểu và giải thích được dòng chảy thông tin qua các cổng (Forget, Input, Output Gate của LSTM và Update, Reset Gate của GRU).
* Thành thạo thao tác với Tensor có chiều `(batch_size, sequence_length, input_size)`.
* Triển khai mô hình Multi-step Forecasting ứng dụng vào phân tích luồng dữ liệu (ví dụ: dự báo giá/khối lượng tài chính).
* Sử dụng hàm `pack_padded_sequence` để tối ưu huấn luyện các batch dữ liệu chuỗi có độ dài không đồng đều.