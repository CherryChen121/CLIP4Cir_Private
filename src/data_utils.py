import os
import json
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

base_path = Path(__file__).absolute().parents[1].absolute()

DEFAULT_EXTERNAL_DATA_ROOT = Path("/data0/qrchen/datasets")
FASHIONIQ_REQUIRED_DIRS = ("captions", "image_splits", "images")
MEDICAL_FIQ_ROOT_BY_CATEGORY = {
    "CH": "UWF_CIR_Dataset_cold",
    "CO": "UWF_CIR_Dataset_cold",
    "NM": "UWF_CIR_Dataset_cold",
    "RB": "UWF_CIR_Dataset_cold",
    "RCH": "UWF_CIR_Dataset_cold",
    "UM": "UWF_CIR_Dataset_cold",
    "IDRiD": "IDRiD_CIR_Dataset_cold",
}


def _has_fashioniq_layout(path: Path) -> bool:
    return path.exists() and all((path / name).is_dir() for name in FASHIONIQ_REQUIRED_DIRS)


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _data_root() -> Path:
    return _env_path("CLIP4CIR_DATA_ROOT") or DEFAULT_EXTERNAL_DATA_ROOT


def _candidate_fashioniq_roots(category: str = None) -> List[Path]:
    roots = []

    explicit_fiq_root = _env_path("CLIP4CIR_FASHIONIQ_ROOT")
    if explicit_fiq_root is not None:
        roots.append(explicit_fiq_root)

    if category == "IDRiD":
        explicit_idrid_root = _env_path("CLIP4CIR_IDRID_ROOT")
        if explicit_idrid_root is not None:
            roots.append(explicit_idrid_root)
    elif category in {"CH", "CO", "NM", "RB", "RCH", "UM"}:
        explicit_uwf_root = _env_path("CLIP4CIR_UWF_ROOT")
        if explicit_uwf_root is not None:
            roots.append(explicit_uwf_root)

    data_root = _data_root()
    if category in MEDICAL_FIQ_ROOT_BY_CATEGORY:
        roots.append(data_root / MEDICAL_FIQ_ROOT_BY_CATEGORY[category])

    roots.extend([
        data_root / "fashionIQ_dataset",
        base_path / "fashionIQ_dataset",
    ])

    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def resolve_fashioniq_root(category: str = None, required_split: str = None) -> Path:
    for root in _candidate_fashioniq_roots(category):
        if not _has_fashioniq_layout(root):
            continue
        if category and required_split:
            caption = root / "captions" / f"cap.{category}.{required_split}.json"
            image_split = root / "image_splits" / f"split.{category}.{required_split}.json"
            if not caption.exists() or not image_split.exists():
                continue
        return root

    searched = ", ".join(str(path) for path in _candidate_fashioniq_roots(category))
    hint = (
        "Set CLIP4CIR_FASHIONIQ_ROOT, CLIP4CIR_UWF_ROOT, CLIP4CIR_IDRID_ROOT, "
        "or CLIP4CIR_DATA_ROOT to point at your dataset directory."
    )
    raise FileNotFoundError(f"Could not resolve FashionIQ-format dataset root for {category}. Searched: {searched}. {hint}")


def list_fashioniq_categories(split: str = "train") -> List[str]:
    categories = set()
    roots = [
        _env_path("CLIP4CIR_FASHIONIQ_ROOT"),
        _env_path("CLIP4CIR_UWF_ROOT"),
        _env_path("CLIP4CIR_IDRID_ROOT"),
        _data_root() / "UWF_CIR_Dataset_cold",
        _data_root() / "IDRiD_CIR_Dataset_cold",
        _data_root() / "fashionIQ_dataset",
        base_path / "fashionIQ_dataset",
    ]
    for root in roots:
        if root is None or not _has_fashioniq_layout(root):
            continue
        for path in (root / "captions").glob(f"cap.*.{split}.json"):
            categories.add(path.name.split(".")[1])
    return sorted(categories)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class ToClipTensor:
    """
    Convert PIL image to tensor compatible with CLIP input conventions.

    - force_rgb=True: convert to RGB first (legacy/default behavior)
    - force_rgb=False: keep original mode, then coerce tensor to 3 channels
    """

    def __init__(self, force_rgb: bool = True):
        self.force_rgb = force_rgb

    def __call__(self, image):
        if self.force_rgb:
            image = image.convert("RGB")

        tensor = F.to_tensor(image)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] > 3:
            tensor = tensor[:3, :, :]
        return tensor


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int, force_rgb: bool = True):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        ToClipTensor(force_rgb=force_rgb),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int, force_rgb: bool = True, apply_targetpad: bool = True):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    ops = []
    if apply_targetpad:
        ops.append(TargetPad(target_ratio, dim))

    ops.extend([
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        ToClipTensor(force_rgb=force_rgb),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return Compose(ops)


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        self.preprocess = preprocess
        self.dataset_roots = {
            dress_type: resolve_fashioniq_root(dress_type, split)
            for dress_type in dress_types
        }
        print(
            "FashionIQ-format dataset roots: "
            + ", ".join(f"{dress_type}={root}" for dress_type, root in self.dataset_roots.items())
        )

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.dataset_roots[dress_type] / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(self.dataset_roots[dress_type] / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        # --- 修改后的过滤逻辑 ---   
        def get_path(name):
            # 优先检查 .png，其次是 .jpg
            for root in self.dataset_roots.values():
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    p = root / 'images' / f"{name}{ext}"
                    if os.path.exists(p):
                        return p
            return None

        print("正在校验图片文件是否存在 (自动匹配 .png/.jpg)...")
        
        # 1. 过滤 image_names
        original_names_count = len(self.image_names)
        self.image_names = [n for n in self.image_names if get_path(n) is not None]
        
        # 2. 过滤 triplets
        original_triplets_count = len(self.triplets)
        self.triplets = [
            t for t in self.triplets 
            if get_path(t['candidate']) is not None 
            and (split == 'test' or get_path(t['target']) is not None)
        ]

        print(f"数据校验完成: 索引图片剩余 {len(self.image_names)}/{original_names_count}, 三元组剩余 {len(self.triplets)}/{original_triplets_count}")
        # --- 过滤逻辑结束 ---

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        # 内部辅助函数，确保能找到正确后缀的文件
        def _get_existing_path(name):
            for root in self.dataset_roots.values():
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    p = root / 'images' / f"{name}{ext}"
                    if os.path.exists(p):
                        return p
            raise FileNotFoundError(f"Image {name} not found under {[str(root / 'images') for root in self.dataset_roots.values()]}")

        try:
            if self.mode == 'relative':
                # 原本这里是一个列表，例如 ["描述1", "描述2"]
                raw_captions = self.triplets[index]['captions']
                
                # 保持原始列表格式返回，让训练循环自行处理：
                # - FashionIQ (2条caption): 通过 generate_randomized_fiq_caption() 随机组合
                # - IDRiD (1条caption): 直接使用
                image_captions = raw_captions if isinstance(raw_captions, list) else [raw_captions]

                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image = self.preprocess(PIL.Image.open(_get_existing_path(reference_name)))
                    target_name = self.triplets[index]['target']
                    target_image = self.preprocess(PIL.Image.open(_get_existing_path(target_name)))
                    # 此时返回的是 (Image, Image, String)
                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions
                
                # ... 剩下的 test 和 classic 部分同理，确保返回的是单条 caption ...
                elif self.split == 'test':
                    # 修改这里
                    reference_image = self.preprocess(PIL.Image.open(_get_existing_path(reference_name)))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                # 修改这里
                image = self.preprocess(PIL.Image.open(_get_existing_path(image_name)))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
