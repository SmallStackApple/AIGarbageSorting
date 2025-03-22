from ultralytics import YOLO
import os

class GarbagePredictor:
    def __init__(self, model_path="models/GarbageSortingModel.pt"):
        self.model = YOLO(model_path)  # 加载训练好的YOLO模型

    def predict(self, image_paths):
        predictions = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            results = self.model(image_path)  # 进行推理
            # 解析预测结果
            pred = results[0].boxes  # 获取边界框信息
            predictions.append({
                "boxes": pred.xyxy.tolist(),  # 边界框坐标
                "scores": pred.conf.tolist(),  # 置信度
                "classes": pred.cls.tolist()   # 类别
            })
        return predictions