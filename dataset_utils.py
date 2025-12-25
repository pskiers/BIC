from torch.utils.data import IterableDataset
from datasets import load_dataset
from PIL import Image
import random


class LogoAugmentedDataset(IterableDataset):
    def __init__(
        self,
        num_logos=20,
        split="train",
        pre_transform=None,
        post_transform=None,
        fixed_logo_size=False,
        streaming=True,
    ):
        """
        Args:
            split (str): "train" or "validation".
            streaming (bool): If True, streams from HuggingFace. If False, downloads dataset.
        """
        self.split = split
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.fixed_logo_size = fixed_logo_size
        self.streaming = streaming

        print(f"Loading top {num_logos} logos into memory...")
        logo_ds = load_dataset("AngelUrq/logos", split=f"train[:{num_logos}]")

        self.logos = []
        self.logo_classes = []

        for item in logo_ds:
            img = item["image"].convert("RGBA")
            raw_label = item.get("text", "unknown")
            clean_label = raw_label.lower().replace("logo of ", "").replace("logo", "").strip()
            if not clean_label:
                clean_label = f"logo_{len(self.logo_classes)}"
            self.logos.append(img)
            self.logo_classes.append(clean_label)

        self.class_to_idx = {name: i for i, name in enumerate(self.logo_classes)}

        print(f"Setting up ImageNet-100 (Split: {split}, Streaming: {streaming})...")
        self.bg_dataset = load_dataset("clane9/imagenet-100", split=split, streaming=streaming)

        if split == "train":
            if streaming:
                self.bg_dataset = self.bg_dataset.shuffle(seed=42, buffer_size=1000)
            else:
                self.bg_dataset = self.bg_dataset.shuffle(seed=42)

    def _overlay_logo(self, bg_image, logo_img):
        if bg_image.mode != "RGB":
            bg_image = bg_image.convert("RGB")

        bg_w, bg_h = bg_image.size
        base_dim = min(bg_w, bg_h)

        if self.fixed_logo_size:
            scale = 0.1
        else:
            scale = random.uniform(0.05, 0.15)

        logo_aspect = logo_img.width / logo_img.height
        target_w = int(base_dim * scale)
        target_h = int(target_w / logo_aspect)

        if target_w <= 0 or target_h <= 0:
            return bg_image

        resized_logo = logo_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        max_x = bg_w - target_w
        max_y = bg_h - target_h

        pos_x = random.randint(0, max(0, max_x))
        pos_y = random.randint(0, max(0, max_y))

        result = bg_image.copy()
        result.paste(resized_logo, (pos_x, pos_y), resized_logo)
        return result

    def __iter__(self):
        for bg_item in self.bg_dataset:
            try:
                bg_image = bg_item["image"]

                if self.pre_transform:
                    bg_image = self.pre_transform(bg_image)

                logo_idx = random.randint(0, len(self.logos) - 1)
                logo_img = self.logos[logo_idx]
                label_int = self.class_to_idx[self.logo_classes[logo_idx]]

                final_image = self._overlay_logo(bg_image, logo_img)

                if self.post_transform:
                    final_image = self.post_transform(final_image)

                yield final_image, label_int
            except Exception:
                continue
