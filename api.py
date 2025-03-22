from ultralytics import YOLO
import os

class GarbagePredictor:
    def __init__(self, model_position_path="models/GarbageSortingModel_position.pt", model_type_path="models/GarbageSortingModel_type.pt"):
        self.model_position = YOLO(model_position_path)  # 加载训练好的位置识别YOLO模型
        self.model_type = YOLO(model_type_path)  # 加载训练好的类型识别YOLO模型

    def predict(self, image_paths):
        predictions = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            # 先进行位置识别
            position_results = self.model_position(image_path)  # 进行位置识别推理
            position_pred = position_results[0].boxes  # 获取边界框信息
            # 解析位置识别结果
            boxes = position_pred.xyxy.tolist()  # 边界框坐标
            scores = position_pred.conf.tolist()  # 置信度
            classes = position_pred.cls.tolist()  # 类别

            # 对每个检测到的对象进行类型识别
            type_predictions = []
            for box in boxes:
                x1, y1, x2, y2 = box
                cropped_image = self.crop_image(image_path, x1, y1, x2, y2)
                type_results = self.model_type(cropped_image)  # 进行类型识别推理
                type_pred = type_results[0].boxes  # 获取边界框信息
                if len(type_pred) > 0:
                    type_classes = type_pred.cls.tolist()  # 类别
                    type_predictions.append(type_classes[0])
                else:
                    type_predictions.append(None)

            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "type_predictions": type_predictions
            })
        return predictions

    def crop_image(self, image_path, x1, y1, x2, y2):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image