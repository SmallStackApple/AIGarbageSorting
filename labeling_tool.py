import json
import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, messagebox, simpledialog
import shutil
import tkinter as tk
from tkinter import ttk


class LabelingTool:
    def __init__(self):
        self.image_path = None
        self.image = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.boxes = []
        self.labels = []
        self.unlabeled_dir = 'unlabeled_images'
        self.labeled_dir = 'labeled_images'
        self.mode = self.select_mode()  # 修改: 启动时询问模式
        self.class_labels = ['plastic', 'paper', 'metal', 'glass', 'organic', 'battery', 'trash']  # 修改: 更新垃圾类别标签
        self.current_label = 0  # 新增: 当前选中的垃圾类别索引

        # 确保目录存在
        if not os.path.exists(self.unlabeled_dir):
            os.makedirs(self.unlabeled_dir)
        if not os.path.exists(self.labeled_dir):
            os.makedirs(self.labeled_dir)

    def select_mode(self):

        # 新增: 创建一个简单的GUI来选择模式
        root = tk.Tk()
        root.title("Select Mode")

        mode_var = tk.StringVar(value='detection')  # 默认选择'detection'

        ttk.Label(root, text="Select Mode:").pack(pady=10)

        ttk.Radiobutton(root, text="Detection", variable=mode_var, value='detection').pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(root, text="Material", variable=mode_var, value='material').pack(anchor=tk.W, padx=20)

        def on_select():
            self.mode = mode_var.get()
            root.destroy()

        ttk.Button(root, text="Select", command=on_select).pack(pady=20)

        root.mainloop()

        return self.mode

    def select_image(self):
        root = Tk()
        root.withdraw()
        self.image_path = filedialog.askopenfilename(initialdir=self.unlabeled_dir)
        self.image = cv2.imread(self.image_path)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_rectangle)

    def draw_rectangle(self, event, x, y, flag, param):
        if self.mode == 'detection':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    # 清除之前的矩形框
                    self.image = cv2.imread(self.image_path)
                    for box in self.boxes:
                        cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                self.boxes.append([self.ix, self.iy, x, y])
                self.labels.append(self.current_label)
        elif self.mode == 'material':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    # 清除之前的矩形框
                    self.image = cv2.imread(self.image_path)
                    cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                self.boxes.append([self.ix, self.iy, x, y])
                self.labels.append(self.current_label)

    def save_annotations(self):
        if self.image_path:
            base_name = os.path.basename(self.image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            annotations_path = os.path.join(self.labeled_dir, f'{name_without_ext}_annotations.json')
            
            # 如果标注文件已经存在，则读取现有标注信息
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
                # 更新标注信息
                if self.mode == 'detection':
                    if 'boxes' not in annotations:
                        annotations['boxes'] = []
                    if 'labels' not in annotations:
                        annotations['labels'] = []
                    annotations['boxes'].extend(self.boxes)
                    annotations['labels'].extend(self.labels)
                elif self.mode == 'material':
                    annotations['material_labels'] = self.current_label
            else:
                # 创建新的标注信息
                if self.mode == 'detection':
                    annotations = {
                        'image_name': base_name,
                        'boxes': self.boxes,
                        'labels': self.labels
                    }
                elif self.mode == 'material':
                    annotations = {
                        'image_name': base_name,
                        'material_labels': self.current_label
                    }
            
            # 保存更新后的标注信息
            with open(annotations_path, 'w') as f:
                json.dump(annotations, f)

            # 清空当前标注信息以便进行下一次标注
            self.boxes = []
            self.labels = []

            # 询问用户是否继续标注当前图片
            continue_labeling = messagebox.askyesno("Continue Labeling", "Do you want to continue labeling this image?")
            if not continue_labeling:
                # 将图片移动到已标记目录
                shutil.move(self.image_path, os.path.join(self.labeled_dir, base_name))
                # 自动选择下一张图片
                self.select_next_image()

    def select_next_image(self):
        # 获取未标记目录中的所有图片文件
        images = [f for f in os.listdir(self.unlabeled_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            # 选择第一张图片进行标注
            self.image_path = os.path.join(self.unlabeled_dir, images[0])
            self.image = cv2.imread(self.image_path)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.draw_rectangle)
        else:
            print("No more images to label.")
            cv2.destroyAllWindows()

    def run(self):
        images = [f for f in os.listdir(self.unlabeled_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for filename in images:
            self.image_path = os.path.join(self.unlabeled_dir, filename)
            self.image = cv2.imread(self.image_path)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.draw_rectangle)
            while True:
                cv2.imshow('image', self.image)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC key to exit
                    break
                elif k == ord('s'):  # 's' key to save
                    self.save_annotations()
                    break
                elif k == ord('l'):  # 新增: 'l' key to select label
                    if self.mode == 'detection' or self.mode == 'material':  # 修改: 支持两种模式
                        self.current_label = (self.current_label + 1) % len(self.class_labels)
                        print(f"Selected label: {self.class_labels[self.current_label]}")
                elif k == ord('n'):  # 新增: 'n' key to next image
                    self.save_annotations()
                    break

if __name__ == "__main__":
    os.mkdir('unlabeled_images')
    os.mkdir('labeled_images')
    tool = LabelingTool()
    tool.run()