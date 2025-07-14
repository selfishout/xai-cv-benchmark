import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import urllib.request

# ---------------------------
# Configuration
# ---------------------------
IMAGE_PATH = "sample.jpg"  # Replace with your image file
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Load class labels
# ---------------------------
def load_imagenet_classes():
    class_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    local_path = "imagenet_classes.txt"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(class_url, local_path)
    with open(local_path) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# ---------------------------
# Load image and transform
# ---------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), image

# ---------------------------
# Inference
# ---------------------------
def run_inference(model, input_tensor, class_names):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top5 = torch.topk(probs, 5)
        for i, idx in enumerate(top5.indices):
            print(f"{i+1}: {class_names[idx]} ({probs[idx]*100:.2f}%)")
        return class_names[top5.indices[0]]

# ---------------------------
# Main
# ---------------------------
def main():
    print("🔍 Loading model...")
    model = models.resnet50(pretrained=True)
    model.eval()

    print("📷 Preprocessing image...")
    input_tensor, original_image = preprocess_image(IMAGE_PATH)

    print("📚 Loading class labels...")
    class_names = load_imagenet_classes()

    print("🧠 Running inference...")
    predicted_label = run_inference(model, input_tensor, class_names)

    print(f"✅ Predicted: {predicted_label}")

    # Save image with label
    save_path = os.path.join(SAVE_DIR, f"{predicted_label.replace(' ', '_')}.jpg")
    original_image.save(save_path)
    print(f"📁 Saved labeled image to: {save_path}")

if __name__ == "__main__":
    main()
