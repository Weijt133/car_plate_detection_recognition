import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path, WindowsPath
import os
import json
from settings import Config
from train import LitCRNN
import re
from difflib import SequenceMatcher  # Used to compute string similarity

# Fix WindowsPath serialization issue for checkpoint loading
torch.serialization.add_safe_globals([WindowsPath])


def calculate_accuracy(predicted: str, actual: str) -> float:
    """
    Compute similarity between the predicted and actual strings using Levenshtein distance.
    :param predicted: The predicted license plate string
    :param actual: The actual license plate string (extracted from the image filename)
    :return: Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, predicted, actual).ratio()


def predict_single(image_path: str, model, transform, device):
    try:
        image = Image.open(image_path).convert("L")  # Load and convert image to grayscale
        image_np = np.array(image)

        augmented = transform(image=image_np)
        img_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)  # [SeqLen, B, C]
            logits = logits.permute(1, 0, 2)  # [B, SeqLen, C]

        decoded_text = model.decode_predictions(logits)[0]
        return decoded_text
    except Exception as e:
        return f"ERROR: Unable to process image: {e}"


def predict(test_dir: str, model_path: str,
            alphabets: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_',
            img_h: int = 32, img_w: int = 100):
    """
    Run prediction on all images in the test directory using a pre-trained CRNN model.
    :param test_dir: Path to the directory containing test images
    :param model_path: Path to the model checkpoint file
    :param alphabets: Supported character set for decoding
    :param img_h: Image height for resizing
    :param img_w: Image width for resizing
    :return: Dictionary of results for each image
    """
    print("\n========== Loading model and configuration ==========")

    # Build label mapping for decoding
    label2char = {i + 1: c for i, c in enumerate(alphabets)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LitCRNN.load_from_checkpoint(
        model_path,
        img_h=img_h,
        n_channels=1,
        n_classes=len(alphabets) + 1,
        n_hidden=64,
        lstm_input=32,
        lr=0.001,
        lr_reduce_factor=0.1,
        lr_patience=5,
        min_lr=1e-6,
        label2char=label2char,
        strict=False,
        map_location=device
    )
    model.eval().to(device)
    print(f"Loaded model checkpoint from: {model_path}")

    # Define preprocessing pipeline
    transform = A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=Config.mean, std=Config.std, max_pixel_value=255.0),
        A.ToGray(always_apply=True),
        ToTensorV2()
    ])

    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"❌ Test directory does not exist: {test_dir}")

    # Filter for supported image formats
    image_paths = [p for p in test_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_paths:
        raise ValueError(f"❌ No valid images found in: {test_dir}")

    results = {}
    accuracy_scores = []

    for img_path in sorted(image_paths):
        prediction = predict_single(str(img_path), model, transform, device)

        # Extract the actual plate string from the filename (text after the first underscore)
        actual_plate = img_path.stem.split('_', 1)[-1]
        # Optionally clean the extracted string with regex:
        # actual_plate = re.sub(r'[^0-9A-Za-z\[\]_]', '', actual_plate)

        accuracy = calculate_accuracy(prediction, actual_plate)
        accuracy_scores.append(accuracy)

        print(f"{img_path.name} -> Predicted: {prediction} | Actual: {actual_plate} | Accuracy: {accuracy * 100:.2f}%")
        results[img_path.name] = {
            "predicted": prediction,
            "actual": actual_plate,
            "accuracy": accuracy * 100
        }

    # Compute overall average accuracy
    overall_accuracy = np.mean(accuracy_scores)
    print(f"\n========== Average accuracy: {overall_accuracy * 100:.2f}% ==========")

    return results


# Example usage parameters
test_dir = r"D:\444prj\crnn-pytorch-master\data-dir\test_img"
model_path = r"D:\444prj\crnn-pytorch-master\output\exp_4\best.ckpt"

if __name__ == "__main__":
    predict(test_dir=test_dir, model_path=model_path)
