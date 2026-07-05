# Hệ Thống Trích Xuất Thông Tin Giấy Tờ

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellow)

## 1. Giới thiệu 
Dự án này là một hệ thống trích xuất thông tin từ CCCD. Hệ thống được triển khai dưới dạng API (FastAPI) với luồng xử lý (Pipeline) như sau:

1. **Card Detection:** Sử dụng **YOLO** để phát hiện và khoanh vùng vị trí thẻ trong ảnh.
2. **Skew Correction:** Kết hợp **OpenCV (Perspective Transform)** để nắn thẳng ảnh nếu thẻ bị nghiêng/lệch góc.
3. **Field Detection:** Tiếp tục dùng YOLO để cắt các trường thông tin (ID, Họ Tên, Ngày Sinh, Quê Quán...).
4. **Text Recognition (OCR):** Đưa các trường thông tin đã cắt vào **VietOCR (VGG_Transformer)** 

Dự án được xây dựng nhằm đáp ứng các tiêu chuẩn về mã nguồn sạch, tối ưu hóa tốc độ (Latency) và khả năng triển khai thực tế.


##  2. Cấu trúc thư mục (Directory Structure)

```text
├── config/
│   └── config.py               # Chứa các tham số cấu hình (ngưỡng conf, đường dẫn...)
├── models/                     # Thư mục chứa trọng số mô hình (weights)
│   ├── detect_card.onnx          # Model YOLO phát hiện thẻ (PyTorch)
│   ├── detect_field.onnx        # Model YOLO phát hiện trường thông tin
├── main.py                     # File chạy server FastAPI (API Endpoints)
├── ocr.py                      # Core logic xử lý ảnh, pipeline nhận diện
├── requirements.txt            # Danh sách thư viện Python
├── Dockerfile                  # Cấu hình đóng gói Docker
└── README.md                   # Tài liệu hướng dẫn
```

## 3. Mô hình sử dụng

Hệ thống không yêu cầu huấn luyện lại từ đầu mà tận dụng sức mạnh của các mô hình đã được Pre-trained, giúp dễ dàng triển khai:

* **Card & Field Detection:** Sử dụng mô hình **YOLOv8n** fine-tuned trên tập dữ liệu giấy tờ tùy thân.
* **Text Recognition (OCR):** Sử dụng kiến trúc **VGG_Transformer** với trọng số pre-trained được cung cấp bởi thư viện `vietocr`, tối ưu hóa rất tốt cho việc nhận dạng ngôn ngữ tiếng Việt có dấu.
  
## 4. Cài đặt

```bash
# 1. Clone repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# 2. Tạo và kích hoạt môi trường ảo (tuỳ chọn nhưng khuyến nghị)
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

## 5. Inference

Để khởi động Server API, chạy lệnh sau:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 5. Benchmark

Test trên RTX3050:

Thời gian xử lý trung bình (End-to-End): ~1s / ảnh

Thời gian OCR: ~800ms / ảnh
