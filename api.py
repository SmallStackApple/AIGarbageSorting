from ultralytics import YOLO
from PIL import Image


class GarbagePredictor:
    def __init__(self, model_position_path="models/GarbageSortingModel_position.pt",
                 model_type_path="models/GarbageSortingModel_type.pt"):
        """初始化预测器 | Initialize the predictor
        Args:
            model_position_path (str): 模型位置路径 | Path to the position detection model
            model_type_path (str): 类型模型路径 | Path to the type detection model
        """
        # 加载位置检测模型 | Load position detection model
        self.model_position = YOLO(model_position_path)
        # 加载类型分类模型 | Load type classification model
        self.model_type = YOLO(model_type_path)

    def predict_position(self, images):
        """预测垃圾位置信息 | Predict garbage position
        Args:
            images (list): 支持路径、base64或PIL.Image的图像列表 | List of images (support path, base64 or PIL.Image)
        Returns:
            list: 包含boxes/scores/classes的位置预测结果列表 | List containing position prediction results (boxes, scores, classes)
        """
        processed_images = []
        for img in images:
            if isinstance(img, str):
                if img.startswith('data:image'):
                    img_pil = self._base64_to_image(img)
                else:
                    img_pil = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img_pil = img
            processed_images.append(img_pil)

        results = self.model_position.predict(processed_images, conf=0.3)  # 置信度阈值0.3
        outputs = []
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().tolist() if res.boxes else []
            scores = res.boxes.conf.cpu().numpy().tolist() if res.boxes else []
            classes = res.boxes.cls.int().cpu().numpy().tolist() if res.boxes else []
            outputs.append({
                "boxes": boxes,
                "scores": scores,
                "classes": classes
            })
        return outputs

    def predict_type(self, images):
        """直接预测垃圾类型 | Directly predict garbage type without segmentation
        Args:
            images (list): 原始图像列表 | List of original images
        Returns:
            list: 类型预测结果列表 | List of type prediction results
        """
        if not isinstance(images, list):
            raise ValueError("images 必须是列表 | images must be a list")
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img_pil = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img_pil = img
            else:
                raise TypeError("图像必须是路径或PIL.Image对象 | Image must be a path or PIL.Image object")
            processed_images.append(img_pil)

        type_results = self.model_type.predict(processed_images)
        # 提取分类概率 | Extract classification probabilities
        outputs = []
        for res in type_results:
            probs = res.probs.cpu().numpy().tolist() if res.probs is not None else []
            outputs.append({
                "probs": probs,
                "class": int(res.probs.argmax()) if res.probs is not None else -1
            })
        return outputs

    def predict(self, images):
        """联合预测位置和类型 | Joint prediction of position and type
        Args:
            images (list): 输入图像列表 | List of input images
        Returns:
            list: 包含位置和类型预测的完整结果 | Combined prediction results including position and type
        """
        pos_results = self.predict_position(images)
        for pos_res in pos_results:
            for box in pos_res["boxes"]:
                x1, y1, x2, y2 = box
                cropped_image = self.crop_image(images[pos_res["boxes"].index(box)], x1, y1, x2, y2)
                pos_res["cropped_image"] = cropped_image
        type_results = self.predict_type(pos_results)
        
        combined = []
        for pos_res, type_res in zip(pos_results, type_results):
            combined_item = {
                "position": pos_res,
                "type": type_res
            }
            combined.append(combined_item)
        return combined

    def crop_image(self, image, x1, y1, x2, y2):
        """裁剪PIL图像 | Crop PIL image
        Args:
            image (PIL.Image.Image): 输入图像 | Input image
            x1 (int): 左上角X坐标 | Top-left X coordinate
            y1 (int): 左上角Y坐标 | Top-left Y coordinate
            x2 (int): 右下角X坐标 | Bottom-right X coordinate
            y2 (int): 右下角Y坐标 | Bottom-right Y coordinate
        Returns:
            PIL.Image.Image: 裁剪后的图像 | Cropped image
        """
        return image.crop((x1, y1, x2, y2))

    # 新增Base64转码方法 | Added Base64 conversion method
    def _base64_to_image(self, img_str):
        """将Base64编码字符串转换为PIL图像 | Convert Base64 string to PIL image"""
        import base64
        import io
        format, imgstr = img_str.split(';base64,')
        ext = format.split('/')[-1]
        return Image.open(io.BytesIO(base64.b64decode(imgstr)))
