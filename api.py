from ultralytics import YOLO
import os


class GarbagePredictor:
    def __init__(self, model_position_path="models/GarbageSortingModel_position.pt",
                 model_type_path="models/GarbageSortingModel_type.pt"):
        """Initialize the predictor
        Args:
            model_position_path (str): Path to the position detection model
            model_type_path (str): Path to the type detection model
        """
        self.model_position = YOLO(model_position_path)  # Load trained YOLO model for position detection
        self.model_type = YOLO(model_type_path)  # Load trained YOLO model for type detection

    def predict(self, images):
        """执行垃圾位置和类型的联合预测 | Perform joint prediction of garbage position and type
    
        Args:
            images (list): 输入图像的列表，支持以下类型 | List of input images, supports the following types:
                - str: 图像路径或Base64编码字符串（以'data:image/'开头） | image path or base64 encoded string (starting with 'data:image/')
                - PIL.Image.Image: PIL图像对象 | PIL image object
                
        Returns:
            list: 预测结果列表，每个元素包含以下内容 | Prediction results list, each element contains:
                - boxes (list): 边界框坐标列表 [[x1,y1,x2,y2], ...] | Bounding box coordinates list
                - scores (list): 位置检测置信度分数 [score1, score2, ...] | Confidence scores for position detection
                - classes (list): 位置分类ID列表 [class_id1, class_id2, ...] | Position class IDs list
                - type_predictions (list): 类型预测分类ID列表 [type_id1, type_id2, ...] | Type prediction class IDs list
                
        Raises:
            FileNotFoundError: 图像路径不存在时抛出 | Raised when image path does not exist
            TypeError: 输入类型不支持时抛出 | Raised when input type is unsupported
        """
        predictions = []
        for image in images:
            # 新增输入类型处理逻辑
            if isinstance(image, str):
                if image.startswith('data:image/'):
                    img = self._base64_to_image(image)
                else:
                    if not os.path.exists(image):
                        raise FileNotFoundError(f"Image not found: {image}")
                    img = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                raise TypeError("Unsupported image type, must be path (str), base64, or PIL.Image")

            # First perform position detection
            position_results = self.model_position(img)  # Modify parameter to PIL image
            position_pred = position_results[0].boxes  # Get bounding box information
            # Parse position detection results
            boxes = position_pred.xyxy.tolist()  # Bounding box coordinates
            scores = position_pred.conf.tolist()  # Confidence
            classes = position_pred.cls.tolist()  # Class

            # Perform type detection for each detected object
            type_predictions = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # Modify cropping method to directly use PIL image
                cropped_image = self.crop_image(img, x1, y1, x2, y2)  # Parameter changed to pass PIL image
                type_results = self.model_type(cropped_image)  # Perform type detection inference
                type_pred = type_results[0].boxes  # Get bounding box information
                if len(type_pred) > 0:
                    type_classes = type_pred.cls.tolist()  # Class
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

    def crop_image(self, image, x1, y1, x2, y2):
        """Crop a PIL image
        
        Args:
            image (PIL.Image.Image): Input image
            x1 (int): Top-left X coordinate
            y1 (int): Top-left Y coordinate
            x2 (int): Bottom-right X coordinate
            y2 (int): Bottom-right Y coordinate
            
        Returns:
            PIL.Image.Image: Cropped image
        """
        return image.crop((x1, y1, x2, y2))

    def _base64_to_image(self, base64_str):
        """Convert a base64 encoded string to a PIL image
        
        Args:
            base64_str (str): Format like "data:image/png;base64,xxxx"
            
        Returns:
            PIL.Image.Image: Decoded RGB image
        """
        from base64 import b64decode
        from io import BytesIO
        _, img_str = base64_str.split(',', 1)
        img_bytes = b64decode(img_str)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
