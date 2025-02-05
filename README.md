# AIGarbageSorting
# 中文

## 项目概述
AIGarbageSorting是一个基于深度学习算法的垃圾分类系统，用于对垃圾进行分类和识别。
## 功能
- **检测模式**：标注垃圾的位置。
- **材质模式**：标注垃圾的材质类别。
- **自动训练**：根据标注的数据自动训练模型，并清理标注后的图片。
## 安装依赖
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
