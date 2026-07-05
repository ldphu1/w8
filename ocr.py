import torch

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from config import config

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: Trái-Trên, Phải-Trên, Phải-Dưới, Trái-Dưới"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Kéo phẳng ảnh dựa vào 4 điểm góc"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    cv2.imshow("warped", warped)
    cv2.imshow("image.jpg", image)
    cv2.waitKey(0)
    return warped

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
        results = self.detect_card(image, verbose=False)

        if len(results[0].boxes) == 0:
            return None

        # Lấy box có confidence cao nhất
        best_idx = results[0].boxes.conf.argmax()
        x1, y1, x2, y2 = map(
            int,
            results[0].boxes.xyxy[best_idx].cpu().numpy()
        )

        # Padding
        pad = 30
        h, w = image.shape[:2]

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            # 1. Morphological Gradient Làm nổi bật mọi viền
            kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)

            # Nhị phân hóa
            _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Đóng các khe hở của viền
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            cv2.imshow("Gradient Edged", edged)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            card_contour = None

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < crop.shape[0] * crop.shape[1] * 0.2:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    if min(w, h) == 0:
                        continue
                    ratio = max(w, h) / float(min(w, h))

                    if 1.3 < ratio < 1.8:
                        card_contour = approx.reshape(4, 2)
                        break

            #Fallback: Nếu approxPolyDP không tìm được 4 góc, quay về dùng minAreaRect cho contour to nhất
            if card_contour is None and len(contours) > 0:
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                if area >= crop.shape[0] * crop.shape[1] * 0.2:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    card_contour = box

            if card_contour is not None:
                # Ép kiểu về float32 cho hàm four_point_transform
                warped = four_point_transform(crop, card_contour.astype("float32"))

                h_warp, w_warp = warped.shape[:2]

                if h_warp > w_warp:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                warped = cv2.resize(warped, (960, 600))
                return warped

        except Exception as e:
            print(f"[WARN] Perspective transform failed: {e}")

        # fallback
        fallback = cv2.resize(crop, (960, 600))

        cv2.imshow("Crop", crop)
        cv2.waitKey(0)

        return fallback

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

        pil_images = []
        filtered_classes = []

        for box, score, cls in zip(boxes, scores, classes):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            field = cropped_img[y1:y2, x1:x2]
            field_rgb = cv2.cvtColor(field, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(field_rgb)

            pil_images.append(pil_image)
            filtered_classes.append(cls)

        if not pil_images:
            raise ValueError("YOLO did not detect any valid fields (score >= 0.5)")

        sents, probs = self.ocr.predict_batch(pil_images, return_prob=True)

        if sents is not None and probs is not None:
            return sents, probs, filtered_classes
        raise ValueError("Cannot OCR")

if __name__ == "__main__":
    ocr = ocr()
    image = cv2.imread("./data/z8008223059742_9483c35263599e9ed8aa059e34c7d6ab.jpg")

    if image is not None:
        try:
            text, prob, cls = ocr.inference(image)
            print("--- KẾT QUẢ OCR ---")
            print(text, prob)

        except Exception as e:
            print(f"Lỗi khi nhận diện: {e}")
    else:
        print("Không thể đọc được ảnh, vui lòng kiểm tra lại đường dẫn!")

