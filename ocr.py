import torch

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from config import config

class ocr:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.detect_card = YOLO(config.DETECT_CARD, task="detect")
        self.detect_field = YOLO(config.DETECT_FIELD, task="detect")

        vietocr_config = Cfg.load_config_from_name(config.OCR_MODEL_NAME)

        vietocr_config['device'] = self.device

        self.ocr = Predictor(vietocr_config)

    def warmup(self):
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

        try:
            _ = self.detect_field(dummy_img, verbose=False)

            _ = self.detect_card(dummy_img, verbose=False)

            dummy_pil = Image.fromarray(cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB))

            _ = self.ocr.predict(dummy_pil)
        except Exception as e:
            print(f"{e}")

    def get_info(self):
        infor = {
            "device": self.device,
             "yolo_model": {
                 "framework": "Ultralyircs",
                 "detect_card_model_path": getattr(self.detect_card, 'ckpt_path', 'model/detect_card.pt'),
                 "detect_field_model_path": getattr(self.detect_field, 'ckpt_path', 'model/detect_field.pt'),
                 "task": getattr(self.detect_card, 'task', 'detect'),
             },
            "viet_ocr": {
                "framework": "VietOCR",
                "model_type": "vgg_transformer",
                "device": self.device,
            }
        }
        return infor

    def crop_card(self, image):
        cards = self.detect_card(image)
        if len(cards[0].boxes) == 0:
            return

        best_idx = cards[0].boxes.conf.argmax()

        box = cards[0].boxes.xyxy[best_idx].cpu().numpy()

        x1, y1, x2, y2 = map(int, box)

        cropped_img = image[y1:y2, x1:x2]

        return cropped_img

    def inference(self, image):
        cropped_img = self.crop_card(image)
        if cropped_img is None:
            raise ValueError("YOLO did not detect any cards")

        fields = self.detect_field(cropped_img)
        if len(fields[0].boxes) == 0:
            raise ValueError("YOLO did not detect any fields")

        boxes = fields[0].boxes.xyxy.cpu().numpy()
        classes = fields[0].boxes.cls.cpu().numpy()
        scores = fields[0].boxes.conf.cpu().numpy()

        results = []

        for box, score in zip(boxes, scores):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            field = cropped_img[y1:y2, x1:x2]
            field_rgb = cv2.cvtColor(field, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(field_rgb)

            results.append(self.ocr.predict(pil_image, return_prob=True))

        if results is not None:
            return results, classes
        raise ValueError("Cannot OCR")

if __name__ == "__main__":
    ocr = ocr()
    image = cv2.imread("./data/159.jpg")

    if image is not None:
        try:
            results, cls = ocr.inference(image)
            print("--- KẾT QUẢ OCR ---")

            for result in results:
                text, prop = result
                print(text, prop)

        except Exception as e:
            print(f"Lỗi khi nhận diện: {e}")
    else:
        print("Không thể đọc được ảnh, vui lòng kiểm tra lại đường dẫn!")

