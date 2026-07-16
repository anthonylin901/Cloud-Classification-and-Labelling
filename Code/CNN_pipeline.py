import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import ResNetForImageClassification, TrainingArguments, Trainer, AutoImageProcessor

from settings.config import IMAGE_SOURCE, LABEL_SOURCE, N_CLASSES, MODEL_NAME
from utils.image_utils import list_images, image_to_patches

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
processor.crop_pct = 1.0

class CloudDataset(Dataset):
    def __init__(self, images):
        self.patches = []
        self.labels = []
        for url in images:
            filename = os.path.basename(url)
            stem = os.path.splitext(filename)[0]
            label_path = os.path.join(LABEL_SOURCE, stem + ".npy")
            grid = np.load(label_path).reshape(-1)
            patches = image_to_patches(url)
            for i in range(len(patches)):
                self.patches.append(patches[i])
                self.labels.append(int(grid[i]))
        print("training cells:", len(self.labels))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        out = processor(self.patches[i], return_tensors="pt")
        return {
            "pixel_values": out["pixel_values"][0],
            "labels": torch.tensor(self.labels[i]),
        }

def train():
    images = list_images(IMAGE_SOURCE)[:5]
    dataset = CloudDataset(images)
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, ignore_mismatched_sizes=True)

    args = TrainingArguments(
        output_dir="hf_out",
        num_train_epochs=5,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("cloud_resnet18_hf")
    return model

if __name__ == "__main__":
    train()