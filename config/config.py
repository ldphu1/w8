import os
import torch

# ==========================================
# 1. PATH & MODEL CONFIG
# ==========================================
DETECT_CARD = "./models/detect_card.onnx"
DETECT_FIELD = "./models/detect_field.onnx"
OCR_MODEL_NAME = "vgg_transformer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. IMAGE PROCESSING CONFIG
# ==========================================
CROP_PADDING = 30
WARP_WIDTH = 960
WARP_HEIGHT = 600

# ==========================================
# 3. AI THRESHOLD CONFIG
# ==========================================
FIELD_CONFIDENCE_THRES = 0.5

# ==========================================
# 4. API & SERVER CONFIG
# ==========================================
MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
HOST = "0.0.0.0"
PORT = 8000