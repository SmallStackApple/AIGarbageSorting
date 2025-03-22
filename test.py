from PIL import Image
from api import GarbagePredictor
import os

def visualize_predictions(image_path, predictions):
    from PIL import ImageDraw
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, score, cls in zip(predictions["boxes"], predictions["scores"], predictions["classes"]):
        x1, y1, x2, y2 = box
        label = f"Class {int(cls)} ({score:.2f})"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
    image.show()

if __name__ == "__main__":
    model_path = "models/GarbageSortingModel.pt"
    predictor = GarbagePredictor(model_path)

    test_images_dir = "E:/PyProject/AIGarbageSorting/test_images"
    for image_name in os.listdir(test_images_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_images_dir, image_name)
            predictions = predictor.predict([image_path])[0]
            print(f"Image: {image_name}, Predictions: {predictions}")
            visualize_predictions(image_path, predictions)