# XAI-CV-Benchmark

This project demonstrates image classification using a pre-trained ResNet-50 model from PyTorch on a sample image. The script loads an image, preprocesses it, runs inference, and saves the result.

## Features
- Loads a sample image (`sample.jpg`)
- Uses a pre-trained ResNet-50 model
- Predicts the top-5 ImageNet classes
- Saves the input image with the predicted label in the `outputs/` directory

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python main.py
```

- The script expects a file named `sample.jpg` in the project root. You can replace it with your own image.
- The output will be saved in the `outputs/` directory with the predicted class as the filename.

## Example Output
After running the script, you will see output similar to:
```
🔍 Loading model...
📷 Preprocessing image...
📚 Loading class labels...
🧠 Running inference...
1: Ibizan hound (57.27%)
2: wire-haired fox terrier (15.44%)
3: Brittany spaniel (10.38%)
4: Irish terrier (3.15%)
5: basenji (2.95%)
✅ Predicted: Ibizan hound
📁 Saved labeled image to: outputs/Ibizan_hound.jpg
```

## Visualization
Below is the input image and the output image (saved in `outputs/`).

| Input Image | Output Image |
|-------------|--------------|
| ![](sample.jpg) | ![](outputs/Ibizan_hound.jpg) |

The output image is the same as the input, but saved with the predicted class as the filename for easy reference.

---
Feel free to modify `main.py` to add more advanced visualizations or explanations! 