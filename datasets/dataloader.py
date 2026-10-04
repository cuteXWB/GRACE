import torch
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

import random
import cv2
from pathlib import Path


import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2


CLIP_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

def to_long_tensor(pic):
    if isinstance(pic, Image.Image):
        pic = np.array(pic, dtype=np.uint8)
    return torch.from_numpy(pic).long()

def random_rotate(image, label):
    angle = random.randint(-20, 20)
    image = image.rotate(angle)
    label = label.rotate(angle)
    return image, label

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        text = sample['text_prompt']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))


        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)


        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)


        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['text_prompt'] = text

        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        text = sample['text_prompt']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if random.random() > 0.5:
            image, mask = random_rotate(image, mask)


        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)


        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)


        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['text_prompt'] = text

        return sample

def to_long_tensor(pic):

    img = torch.from_numpy(np.array(pic, np.uint8))

    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class DatasetSegmentation(Dataset):

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: list,
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224
    ) -> None:

        self.dataset_path = dataset_path
        self.image_size = image_size




        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'label')
        self.legacy_layout = os.path.isdir(self.input_path) and os.path.isdir(self.output_path)
        self._recursive_path_index = None
        dataset_path_norm = str(dataset_path).replace("\\", "/").upper()
        task_name_norm = str(task_name).upper()
        self._enable_eus_resolve = (
            task_name_norm.startswith("EUS") or "/EUS/" in dataset_path_norm
        )
        if (not self.legacy_layout) and self._enable_eus_resolve:
            self._recursive_path_index = self._build_recursive_path_index(dataset_path)
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name

        self.data_pairs = [
            (row['Image'], row['Ground Truth'], row['Description'])
            for row in row_text
        ]

        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x: x

    @staticmethod
    def _sanitize_eus_token(token: str):
        t = str(token).strip().replace("\\", "/").lstrip("./")
        prefixes = (
            "train_folder/img/",
            "train_folder/label/",
            "train_folder/",
            "val_folder/img/",
            "val_folder/label/",
            "val_folder/",
            "test_folder/img/",
            "test_folder/label/",
            "test_folder/",
            "img/",
            "label/",
            "image/",
            "images/",
            "mask/",
            "masks/",
        )
        low = t.lower()
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if low.startswith(p):
                    t = t[len(p):].lstrip("./")
                    low = t.lower()
                    changed = True
                    break
        return t

    def _resolve_nonlegacy_in_root(self, root_dir: str, token: str, legacy_subdir: str):
        token = str(token).strip().replace("\\", "/")




        if "/" not in token and "_" in token:
            folder, fname = token.split("_", 1)
            if os.path.isdir(os.path.join(root_dir, folder)):
                token = f"{folder}/{fname}"


        direct = os.path.join(root_dir, token)
        resolved = self._resolve_with_ext_fallback(direct)
        if resolved is not None:
            return resolved


        if "_" in token:
            folder, fname = token.split("_", 1)
            cand = os.path.join(root_dir, folder, fname)
            resolved = self._resolve_with_ext_fallback(cand)
            if resolved is not None:
                return resolved


        if self._recursive_path_index is not None:
            token_base = os.path.basename(token).strip().replace("\\", "/")
            token_name = token_base.lower()
            token_stem = os.path.splitext(token_name)[0]

            candidates = self._recursive_path_index["by_name"].get(token_name, [])
            picked = self._pick_candidate(candidates, legacy_subdir)
            if picked is not None:
                return os.path.join(root_dir, picked)

            candidates = self._recursive_path_index["by_stem"].get(token_stem, [])
            picked = self._pick_candidate(candidates, legacy_subdir)
            if picked is not None:
                return os.path.join(root_dir, picked)



            if "_" in token_stem:
                tail_stem = token_stem.split("_", 1)[1]
                candidates = self._recursive_path_index["by_tail"].get(tail_stem, [])
                picked = self._pick_candidate(candidates, legacy_subdir)
                if picked is not None:
                    return os.path.join(root_dir, picked)


        return os.path.join(root_dir, token)

    def _resolve_candidate_roots(self, root_dir: str):
        roots = [root_dir]
        if not self._enable_eus_resolve:
            return roots

        norm = str(root_dir).replace("\\", "/").rstrip("/")
        low = norm.lower()
        split_markers = ("/train_folder", "/val_folder", "/test_folder")
        for m in split_markers:
            if low.endswith(m):
                parent = os.path.dirname(norm)
                if parent:
                    roots.append(parent)
                break


        base = os.path.basename(low)
        if base in {"img", "label", "images", "masks", "mask"}:
            parent = os.path.dirname(norm)
            if parent:
                roots.append(parent)
                parent_base = os.path.basename(parent.lower())
                if parent_base in {"train_folder", "val_folder", "test_folder"}:
                    roots.append(os.path.dirname(parent))


        uniq = []
        seen = set()
        for r in roots:
            k = str(r).replace("\\", "/").rstrip("/")
            if k not in seen and k:
                seen.add(k)
                uniq.append(r)
        return uniq

    def _resolve_path(self, root_dir: str, token: str, legacy_subdir: str):








        token = str(token).strip().replace("\\", "/")
        if self._enable_eus_resolve:
            token = self._sanitize_eus_token(token)

        if self.legacy_layout:
            return self._resolve_with_ext_fallback(os.path.join(root_dir, legacy_subdir, token))
        candidate_roots = self._resolve_candidate_roots(root_dir)
        final_fallback = None
        for r in candidate_roots:
            resolved = self._resolve_nonlegacy_in_root(r, token, legacy_subdir)
            if resolved is not None and os.path.exists(resolved):
                return resolved
            if final_fallback is None:
                final_fallback = resolved

        return final_fallback if final_fallback is not None else os.path.join(root_dir, token)

    @staticmethod
    def _build_recursive_path_index(root_dir: str):
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        by_name = {}
        by_stem = {}
        by_tail = {}

        root = Path(root_dir)
        if not root.exists():
            return {"by_name": by_name, "by_stem": by_stem, "by_tail": by_tail}

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in valid_exts:
                continue

            rel = p.relative_to(root).as_posix()
            low_name = p.name.lower()
            low_stem = p.stem.lower()

            by_name.setdefault(low_name, []).append(rel)
            by_stem.setdefault(low_stem, []).append(rel)

            if "_" in low_stem:
                tail = low_stem.split("_", 1)[1]
                by_tail.setdefault(tail, []).append(rel)

        for d in (by_name, by_stem, by_tail):
            for k in d:
                d[k] = sorted(set(d[k]))

        return {"by_name": by_name, "by_stem": by_stem, "by_tail": by_tail}

    @staticmethod
    def _pick_candidate(candidates, legacy_subdir: str):
        if not candidates:
            return None
        legacy_subdir = str(legacy_subdir).lower()
        if legacy_subdir == "img":
            preferred_tags = ("/img/", "/image/", "/images/")
        else:
            preferred_tags = ("/label/", "/labels/", "/mask/", "/masks/", "/gt/")

        for rel in candidates:
            rel_low = "/" + rel.lower()
            if any(tag in rel_low for tag in preferred_tags):
                return rel
        return candidates[0]

    @staticmethod
    def _resolve_with_ext_fallback(path: str):
        if os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        ext_l = ext.lower()
        alt_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
        if ext_l in alt_exts:
            alt_exts = [e for e in alt_exts if e != ext_l]
        for ae in alt_exts:
            cand = stem + ae
            if os.path.exists(cand):
                return cand
        return None

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):

        image_filename, mask_filename, text = self.data_pairs[idx]

        image_path = self._resolve_path(self.dataset_path, image_filename, "img")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask_path = self._resolve_path(self.dataset_path, mask_filename, "label")
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found or unreadable: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask[mask < 127] = 0
        mask[mask >= 127] = 1

        image, mask = correct_dims(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        inputs = {
            "image": image,
            "ground_truth_mask": mask,
            "image_name": image_filename,
            "mask_name": mask_filename,
            "text_prompt": text,
            "dataset_name": self.task_name
        }

        if self.joint_transform:
            inputs = self.joint_transform(inputs)

        return inputs
