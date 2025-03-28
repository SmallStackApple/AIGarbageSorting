import os

import torch
from ultralytics import YOLO


# 训练函数
def train_model(data_yaml, epochs=10, batch_size=32, img_size=640, train_type=False):
    # 根据train_type选择不同的数据路径和模型名称
    if train_type:
        model_name = "yolov8n-cls"  # 使用yolov8n-cls模型进行类型识别
        output_model_path = "models/GarbageSortingModel_type.pt"
    else:
        model_name = "yolov8n"  # 使用原版yolov8n模型进行位置识别
        output_model_path = "models/GarbageSortingModel_position.pt"

    # 加载预训练的YOLO模型
    model = YOLO(f"{output_model_path}.pt" if os.path.exists(f"{output_model_path}.pt") else f"{model_name}.pt")

    # 开始训练
    results = model.train(
        data=data_yaml,  # 数据集配置文件路径
        epochs=epochs,  # 训练轮数
        batch=batch_size,  # 批量大小
        imgsz=img_size,  # 输入图像尺寸
        device="cuda" if torch.cuda.is_available() else "cpu",
        workers=4  # 数据加载线程数
    )
    print("Training completed.")
    print(results)
    # 保存训练好的模型
    model.save(output_model_path)


if __name__ == "__main__":
    train_type = True
    train_model('./datasets/type' if train_type else './datasets/position', train_type=train_type)  # 训练类型识别模型
