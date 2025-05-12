import os
from argparse import ArgumentParser
from os.path import join, split

import albumentations
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from deep_utils import split_extension, log_print
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class CRNNDataset(Dataset):

    def __init__(self, root, characters, transform=None, logger=None):
        self.transform = transform
        # index zero is reserved for CTC's blank token
        self.char2label = {char: i + 1 for i, char in enumerate(characters)}
        self.label2char = {label: char for char, label in self.char2label.items()}
        self.image_paths, self.labels, self.labels_length = self.get_image_paths(
            root, characters,
            chars2label=self.char2label,
            logger=logger
        )
        # +1 accounts for CTC's blank token
        self.n_classes = len(self.label2char) + 1

    @staticmethod
    def text2label(char2label: dict, text: str):
        return [char2label[t] for t in text]

    @staticmethod
    def get_image_paths(root, chars, chars2label, logger=None):
        paths, labels, labels_length = [], [], []
        discards = 0
        for img_name in os.listdir(root):
            img_path = join(root, img_name)
            try:
                # Only process JPG files with numeric names
                if img_name.lower().endswith(('.jpg', '.JPG')):
                    print(f"Processing image: {img_name}")  # Print the file being processed
                    text = CRNNDataset.get_label(img_path)
                    is_valid, invalid_char = CRNNDataset.check_validity(text, chars)
                    if is_valid:
                        label = CRNNDataset.text2label(chars2label, text)
                        paths.append(img_path)
                        labels.append(label)
                        labels_length.append(len(label))
                    else:
                        log_print(
                            logger,
                            f"[Warning] Text for sample {img_path} is invalid due to character: {invalid_char}"
                        )
                        discards += 1
                else:
                    log_print(
                        logger,
                        f"[Warning] Sample {img_path} has an invalid extension or name. Skipping..."
                    )
                    discards += 1
            except Exception as e:
                log_print(
                    logger,
                    f"[Warning] Sample {img_path} is invalid. Skipping... Error: {e}"
                )
                discards += 1
        assert len(labels) == len(paths)
        log_print(
            logger,
            f"Successfully gathered {len(labels)} samples and discarded {discards} samples!"
        )
        return paths, labels, labels_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        assert index < len(self), 'Index out of range'
        img_path = self.image_paths[index]

        # Convert the image directly to grayscale and load it
        img = Image.open(img_path).convert("L")  # 'L' mode yields a single-channel grayscale image

        if isinstance(self.transform, albumentations.core.composition.Compose):
            # Prepare as 1-channel array for albumentations
            img_array = np.array(img)[..., None]
            aug = self.transform(image=img_array)
            img = aug['image']  # shape: (1, H, W)
            assert img.ndim == 3 and img.shape[0] == 1, f"Unexpected channel shape: {img.shape}"
        else:
            img = self.transform(img).unsqueeze(0)  # Torch transforms expect a PIL image

        label = torch.LongTensor(self.labels[index]).unsqueeze(0)
        label_length = torch.LongTensor([self.labels_length[index]]).unsqueeze(0)

        return img, label, label_length

    @staticmethod
    def get_label(img_path):
        # Extract the base filename without extension, then split by '_' and take the last part as label
        filename = split(img_path)[-1]
        name_no_ext = split_extension(filename)[0]
        return name_no_ext.split('_')[-1]

    @staticmethod
    def check_validity(text, chars):
        for c in text:
            if c not in chars:
                print(f"[Debug] Invalid character found: {c}")
                return False, c
        return True, None

    @staticmethod
    def collate_fn(batch):
        images, labels, labels_lengths = zip(*batch)
        # Stack images into a batch tensor
        images = torch.stack(images, dim=0)
        # Remove extra batch dim from each label
        labels = [label.squeeze(0) for label in labels]
        # Pad labels with -100 (ignored by CTC loss) according to the longest label in batch
        labels = nn.utils.rnn.pad_sequence(labels, padding_value=-100).T
        labels_lengths = torch.cat(labels_lengths, dim=0)
        return images, labels, labels_lengths


def get_mean_std(dataset_dir, alphabets, batch_size, img_h, img_w):
    """
    Compute channel-wise mean and standard deviation over the dataset.
    """
    # Define preprocessing transforms: grayscale conversion, resizing, tensor conversion
    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()
    ])

    dataset = CRNNDataset(
        root=dataset_dir,
        transform=transformations,
        characters=alphabets
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn
    )

    mean_sum, std_sum = 0, 0
    n_samples = len(dataset)
    for images, _, _ in tqdm(data_loader, desc="Computing mean and std"):
        # Sum mean and std over spatial dimensions and then batch
        mean_sum += torch.sum(torch.mean(images, dim=(2, 3)), dim=0)
        std_sum += torch.sum(torch.std(images, dim=(2, 3)), dim=0)

    mean = mean_sum / n_samples
    std = std_sum / n_samples
    return [round(m, 4) for m in mean.tolist()], [round(s, 4) for s in std.tolist()]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", help="path to dataset")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--alphabets", default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_')
    parser.add_argument("--img_h", default=32, type=int)
    parser.add_argument("--img_w", default=100, type=int)
    args = parser.parse_args()

    mean, std = get_mean_std(
        args.dataset_dir,
        alphabets=args.alphabets,
        batch_size=args.batch_size,
        img_h=args.img_h,
        img_w=args.img_w
    )
    log_print(None, f"MEAN: {mean}, STD: {std}")
