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
│   ├── detect_card.pt          # Model YOLO phát hiện thẻ (PyTorch)
│   ├── detect_field.pt         # Model YOLO phát hiện trường thông tin
├── main.py                     # File chạy server FastAPI (API Endpoints)
├── ocr.py                      # Core logic xử lý ảnh, pipeline nhận diện
├── requirements.txt            # Danh sách thư viện Python
├── Dockerfile                  # Cấu hình đóng gói Docker
└── README.md                   # Tài liệu hướng dẫn
