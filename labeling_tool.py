import json
import cv2
import numpy as np
import os
from tkinter import Tk, filedialog
import shutil


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

        # 确保目录存在
        if not os.path.exists(self.unlabeled_dir):
            os.makedirs(self.unlabeled_dir)
        if not os.path.exists(self.labeled_dir):
            os.makedirs(self.labeled_dir)

        # 将所有图片从已标记目录移动回未标记目录
        for filename in os.listdir(self.labeled_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(os.path.join(self.labeled_dir, filename), self.unlabeled_dir)

    def select_image(self):
        root = Tk()
        root.withdraw()
        self.image_path = filedialog.askopenfilename(initialdir=self.unlabeled_dir)
        self.image = cv2.imread(self.image_path)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_rectangle)

    def draw_rectangle(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.boxes.append([self.ix, self.iy, x, y])
            label = input("Enter label for the box: ")
            self.labels.append(label)

    def save_annotations(self):
        if self.image_path:
            base_name = os.path.basename(self.image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            annotations = {
                'image_name': base_name,
                'boxes': self.boxes,
                'labels': self.labels
            }
            with open(os.path.join(self.labeled_dir, f'{name_without_ext}_annotations.json'), 'w') as f:
                json.dump(annotations, f)

            # 将图片移动到已标记目录
            shutil.move(self.image_path, os.path.join(self.labeled_dir, base_name))
            # 自动选择下一张图片
            self.select_image()

    def run(self):
        for filename in os.listdir(self.unlabeled_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tool = LabelingTool()
    tool.run()
