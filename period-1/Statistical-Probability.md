# Xác Suất Thống Kê Ứng Dụng Trong Trí Tuệ Nhân Tạo

## Objective

Xây dựng nền tảng xác suất thống kê cốt lõi giúp hệ thống AI định lượng sự bất định, xây dựng cơ sở toán học cho hàm mất mát, và chuẩn hóa dữ liệu chuỗi thời gian trước khi đưa vào mô hình học máy.

## Current State

Đã chưng cất thành công các khái niệm thống kê ứng dụng định hình kiến trúc AI thực chiến. Các quy chuẩn về xử lý phân phối dữ liệu đuôi dày, cơ sở MLE cho hàm tối ưu hóa, và yêu cầu bắt buộc về tính dừng đối với chuỗi thời gian đã được hợp nhất thành quy tắc tiền xử lý dữ liệu chuẩn.

## Confirmed Knowledge

* **Phân phối Dữ liệu & Hiện tượng Đuôi dày (Fat-Tailed Distribution):**
* Dữ liệu thế giới thực (đặc biệt là tài chính) không tuân theo Phân phối chuẩn (Normal Distribution) lý tưởng. Các sự kiện cực đoan, đột biến (Black Swan) có xác suất xảy ra cao hơn đáng kể so với dự báo của mô hình hình chuông.
* *Cảnh báo hệ thống:* Việc áp dụng trực tiếp các phương pháp thống kê cơ bản như Z-score dựa trên giả định phân phối chuẩn cho hệ thống Phát hiện bất thường (Anomaly Detection) trên luồng dữ liệu thời gian thực sẽ sinh ra tỷ lệ báo động giả (false positives) rất lớn.


* **Ước lượng Hợp lý Cực đại (MLE - Maximum Likelihood Estimation):**
* Nền tảng thống kê giải thích cơ chế hoạt động của Hàm mất mát (Loss Function). Nguyên lý cốt lõi là tìm kiếm bộ trọng số mô hình có xác suất (likelihood) cao nhất để sinh ra bộ dữ liệu quan sát được.
* *Định lý ứng dụng:* Việc tối đa hóa hàm hợp lý MLE trong điều kiện nhiễu ngẫu nhiên tuân theo phân phối chuẩn hoàn toàn tương đương với việc tối thiểu hóa hàm Sai số bình phương trung bình (MSE).


* **Tính Dừng (Stationarity) Trong Phân Tích Chuỗi Thời Gian:**
* *Khái niệm:* Một chuỗi thời gian có "Tính dừng" khi các đặc trưng thống kê của nó (giá trị trung bình, phương sai) không thay đổi theo thời gian.
* *Quy tắc bất di bất dịch:* Tuyệt đối không đưa dữ liệu chuỗi Không dừng (Non-stationary) - điển hình là chuỗi "Giá" tuyệt đối của tài sản (với trung bình trôi dạt liên tục) - trực tiếp vào các mô hình học máy (như ARIMA).
* *Phương pháp biến đổi:* Bắt buộc áp dụng phương pháp Sai phân (Differencing) để chuyển đổi dữ liệu thô thành chuỗi Dừng (ví dụ: chuyển biến động giá thành "Tỷ suất sinh lời" - Returns). Phép biến đổi này ép trung bình chuỗi về cố định quanh mức 0, tạo cấu trúc ổn định để thuật toán có thể học được các quy luật Tự tương quan (Autocorrelation).