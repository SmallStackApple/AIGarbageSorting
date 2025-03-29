# AIGarbageSorting :: AIGarbageSorting

## 项目简介 :: Project Introduction  
基于YOLO的垃圾智能分类系统，实现垃圾位置定位与类型识别的端到端解决方案。  
The intelligent garbage classification system based on YOLO, providing an end-to-end solution for garbage position localization and type recognition.

## 功能特性 :: Features  
- 支持多输入格式（图片路径/ Base64编码/PIL图像）  
- 双模型联合预测：先定位垃圾位置，再进行类型分类  
- 可视化预测结果展示  
- 支持模型训练与参数调优  

## 依赖环境 :: Dependencies
### 使用虚拟环境 (推荐)
1. 确保已安装Python 3.x(最好使用3.12.X)。
2. 安装virtualenv(如果没有安装):
   ```commandline
   pip install virtualenv
   ```
3. 切换到项目目录下:
   ```commandline
   cd AIGarbageSorting
   ```
4. 创建虚拟环境:
   ```commandline
   virtualenv .venv
   ```
5. 激活虚拟环境:
   bash:
   ```commandline
   source .venv/bin/activate
   ```
   cmd:
   ```commandline
   .\.venv\Scripts\activate.bat
   ```
   powershell:
   ```commandline
      .\.venv\Scripts\Activate.ps1
   ```
6. 使用pip安装项目所需依赖:
   ```commandline
   pip install -r requirements.txt
   ```
### 不使用虚拟环境 (不推荐)
1. 确保已安装Python 3.x。(最好使用3.12.X)
2. 使用pip安装项目所需依赖
## 使用
### 1.标注
1. 运行`labeling_tool.py`文件，启动标注程序。
2. 第一次运行时会创建[unlabeled_images](unlabeled_images)和[labeled_images](labeled_images)
3. 将标注图片放进[unlabeled_images](unlabeled_images)中
4. 选择对应模式进行标注
### ~~2.不想写了~~

## 贡献
欢迎任何形式的贡献，包括但不限于:
- 提交~~无关~~问题或建议
- ~~不~~贡献模型文件
- ~~不~~共献代码
- ~~发送垃圾邮件给作者~~
- ~~辱骂作者~~
- ~~删除项目~~
- ~~开户作者~~
- ~~skid后商业化或私有化~~
