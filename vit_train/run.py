import functools
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification, BatchFeature
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, RandomResizedCrop, RandomHorizontalFlip, CenterCrop

import torch
from torchvision.datasets import CIFAR10, ImageNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from img_model import ImageClassifier


class SimpleCustomBatch:
    def __init__(self, tokenizer, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

        self.tokenizer = tokenizer

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {**self.tokenizer(images=self.inp, return_tensors="pt"), 'labels': self.tgt}


def my_collate(batch, tokenizer):
    return SimpleCustomBatch(tokenizer, batch)


class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path, split="train"):
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        transform = []

        if feature_extractor.do_resize:
            transform.append(
                RandomResizedCrop(feature_extractor.size) if split == "train" else Resize(feature_extractor.size)
            )

        transform.append(RandomHorizontalFlip() if split == "train" else CenterCrop(feature_extractor.size))
        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x.convert("RGB"))

if __name__ == '__main__':

    home = os.path.expanduser('~')
    dataset_path = os.path.join(home, "ImageNet")

    model_name_or_path = os.path.join(home,'HuggingFace/google/vit-huge-patch14-224-in21k')
    num_labels = 1000
    batch_size = 16
    num_workers = 20
    max_epochs = 10

    feature_extractor=ViTFeatureExtractor.from_pretrained(model_name_or_path)

    train_loader = DataLoader( 
        ImageNet(dataset_path, download=False, split="train", transform=ViTFeatureExtractorTransforms(model_name_or_path, split="train")),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=functools.partial(my_collate, tokenizer=feature_extractor)
    ) 
    val_loader = DataLoader( 
        ImageNet(dataset_path, download=False, split="val", transform=ViTFeatureExtractorTransforms(model_name_or_path, split="val")),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=functools.partial(my_collate, tokenizer=feature_extractor)
    )


    model = ImageClassifier(model_name_or_path, learning_rate=3e-4, weight_decay=0.1)
    # HACK - put this somewhere else
    model.total_steps = (
        (len(train_loader.dataset) // (batch_size))
        // 1
        * float(max_epochs)
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        every_n_epochs=1,
        save_top_k=5
    )

    # pixel_values, labels = next(iter(train_loader))
    trainer = pl.Trainer(
        gpus=-1, 
        auto_select_gpus=False,
        max_epochs=max_epochs, 
        precision=32, 
        limit_train_batches=5, 
        enable_checkpointing=True, 
        enable_progress_bar=True, 
        strategy='ddp',
        accumulate_grad_batches=4,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint="lightning_logs/version_2/checkpoints/epoch=3-step=7.ckpt",
    )
    trainer.fit(model, train_loader, val_loader)
