import argparse
import os
from PIL import Image
from api import GarbagePredictor  # 新增导入语句

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['both', 'position', 'type'], default='both',
                        help='选择识别模式: both(默认), position仅位置检测, type仅类型检测 | Select detection mode: both(default), position only, type only')
    parser.add_argument('--input_dir', type=str, default='./test_images', 
                        help='测试图片目录路径 | Test images directory path')
    
    args = parser.parse_args()
    test_images_dir = args.input_dir

    predictor = GarbagePredictor()  # 初始化预测器实例 | Initialize predictor instance

    # 遍历测试图片目录 | Iterate through test images directory
    for image_name in os.listdir(test_images_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_images_dir, image_name)
            image = Image.open(img_path)  # 读取图像 | Read image

            # 处理不同模式的预测逻辑 | Handle different prediction modes
            if args.mode == 'both':
                # 联合预测 | Joint prediction
                results = predictor.predict([image])
                print(f"联合预测结果：{results}")
                
            elif args.mode == 'position':
                # 仅位置预测 | Position prediction only
                pos_results = predictor.predict_position([image])
                print(f"位置预测结果：{pos_results}")
                
            elif args.mode == 'type':
                # 先预测位置再类型 | Predict position then type
                pos_results = predictor.predict_position([image])
                boxes_list = [res.boxes for res in pos_results]  # 假设结果包含boxes属性
                type_results = predictor.predict_type([image])
                print(f"类型预测结果：{type_results}")