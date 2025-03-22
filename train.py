import os

import torch
from ultralytics import YOLO


# 训练函数
def train_model(data_yaml, model_name="yolov8n", epochs=10, batch_size=16, img_size=640, train_type=False):
    # 根据train_type选择不同的数据路径和模型名称
    if train_type:
        data_yaml = {
            "train": "E:/PyProject/AIGarbageSorting/images/train/type",
            "val": "E:/PyProject/AIGarbageSorting/images/val/type",
            "nc": 4,
            "names": ['Harmful', 'Kitchen', 'Other', 'Recyclable']
        }
        model_name = "yolov8n-cls"  # 使用yolov8n-cls模型进行类型识别
    else:
        data_yaml = {
            "train": "E:/PyProject/AIGarbageSorting/images/train/position",
            "val": "E:/PyProject/AIGarbageSorting/images/val/position",
            "nc": 4,
            "names": ['Harmful', 'Kitchen', 'Other', 'Recyclable']
        }
        model_name = "yolov8n"  # 使用原版yolov8n模型进行位置识别

    # 加载预训练的YOLO模型
    model = YOLO(f"{model_name}.pt")

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
    print(f"Results saved to {results.save_dir}")


if __name__ == "__main__":
    data_yaml = "E:/PyProject/AIGarbageSorting/data.yaml"  # 数据集配置文件路径
    train_model(data_yaml, train_type=False)  # 默认训练位置识别模型
    # 若要训练类型识别模型，可以调用 train_model(data_yaml, train_type=True)