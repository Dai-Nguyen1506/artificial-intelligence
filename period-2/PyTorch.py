import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# MẢNH GHÉP 1: DATASET & DATALOADER
# ==========================================
class SyntheticDataset(Dataset):
    def __init__(self):
        # Tạo dữ liệu giả lập gồm 100 mẫu (X) và nhãn (y) tương ứng theo công thức y = 2x + 1 + nhiễu
        self.X = torch.randn(100, 1)
        self.y = 2 * self.X + 1 + torch.randn(100, 1) * 0.1

    def __len__(self):
        # Trả về tổng số lượng mẫu dữ liệu
        return len(self.X)

    def __getitem__(self, idx):
        # Trả về một cặp dữ liệu (X, y) tại vị trí chỉ mục idx
        return self.X[idx], self.y[idx]

# Khởi tạo tập dữ liệu và bộ tải dữ liệu theo từng batch (khối)
dataset = SyntheticDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# ==========================================
# MẢNH GHÉP 2: QUẢN LÝ THIẾT BỊ (GPU/CPU)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# KHỞI TẠO MÔ HÌNH VÀ THUẬT TOÁN TỐI ƯU
# ==========================================
# Tạo một tầng tuyến tính (Linear Layer) đầu vào 1, đầu ra 1 và đẩy thẳng lên thiết bị đã chọn
model = nn.Linear(1, 1).to(device)

# Xác định hàm đo sai số (MSE Loss) và thuật toán cập nhật trọng số (SGD)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


# ==========================================
# MẢNH GHÉP 3 & 4: TRAINING LOOP & AUTOGRAD
# ==========================================
print("--- Bắt đầu huấn luyện ---")
for epoch in range(100):  # Huấn luyện qua 100 chu kỳ (Epoch)
    total_loss = 0.0
    
    for batch_X, batch_y in dataloader:
        # Đẩy dữ liệu của batch hiện tại lên cùng thiết bị với mô hình (GPU/CPU)
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 5 BƯỚC CỐT LÕI CỦA MỘT VÒNG LẶP HUẤN LUYỆN
        y_pred = model(batch_X)             # Bước 1: Forward pass (Mô hình dự đoán)
        loss = criterion(y_pred, batch_y)   # Bước 2: Tính toán sai số (Loss)
        
        optimizer.zero_grad()               # Bước 3: Xóa sạch gradient (sai số tích lũy) của batch trước
        loss.backward()                     # Bước 4: Backward pass (Autograd tính toán đạo hàm ngược)
        optimizer.step()                    # Bước 5: Cập nhật trọng số mới dựa trên gradient
        
        # Tích lũy loss để theo dõi, dùng .item() để giải phóng bộ nhớ đồ thị tính toán
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/100 - Loss trung bình: {avg_loss:.4f}")


# ==========================================
# KIỂM TRA MÔ HÌNH SAU KHI HUẤN LUYỆN (INFERENCE)
# ==========================================
print("\n--- Thử nghiệm mô hình với dữ liệu mới ---")
sample_input = torch.tensor([[3.0]]).to(device)
print("W đang học được:", model.weight.item())
print("b đang học được:", model.bias.item())

# Tắt tính năng tính đạo hàm và xây dựng đồ thị tính toán để tối ưu bộ nhớ GPU
with torch.no_grad():
    prediction = model(sample_input)
    print(f"Đầu vào: X = 3.0 -> Mô hình dự đoán Y = {prediction.item():.4f} (Giá trị thực tế lý thuyết: 7.0)")