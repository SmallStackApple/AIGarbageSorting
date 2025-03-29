import os

import torch
from ultralytics import YOLO


# 训练函数 | Training function
def train_model(data_yaml, epochs=10, batch_size=32, img_size=640, train_type=False):
    # 修改模型加载逻辑，避免重复加载.pt后缀 | Modify model loading logic to avoid duplicate .pt suffix loading
    if train_type:
        model_name = "yolov8n-cls"
        output_model_path = "models/GarbageSortingModel_type.pt"
    else:
        model_name = "yolov8n"
        output_model_path = "models/GarbageSortingModel_position.pt"

    # 使用官方推荐的预训练权重加载方式 | Use official recommended pretrained weights loading method
    model = YOLO(model_name) if not os.path.exists(output_model_path) else YOLO(output_model_path)

    # 开始训练 | Start training
    results = model.train(
        data=data_yaml,  # 数据集配置文件路径 | Dataset configuration file path
        epochs=epochs,  # 训练轮数 | Number of training epochs
        batch=batch_size,  # 批量大小 | Batch size
        imgsz=img_size,  # 输入图像尺寸 | Input image size
        device="cuda" if torch.cuda.is_available() else "cpu",  # 设备选择 | Device selection
        workers=4  # 数据加载线程数 | Number of data loading threads
    )
    print("Training completed.")
    print(results)
    # 保存训练好的模型 | Save the trained model
    model.save(output_model_path)


if __name__ == "__main__":
    train_type = True
    train_model('./datasets/type' if train_type else './datasets/position', train_type=train_type)  # 训练类型识别模型
