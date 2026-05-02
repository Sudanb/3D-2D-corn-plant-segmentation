# ---------------------------------------------------------------
# Maize instance segmentation dataset loader for EoMT.
# Classes: 0=stem, 1=leaf  (2 total — all leaf1-16 merged)
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset

# COCO category IDs (1-indexed): 1=stem, 2-17=leaf1-16
CLASS_MAPPING_2  = {1: 0, **{i: 1 for i in range(2, 18)}}   # 2-class: all leaves → 1
CLASS_MAPPING_17 = {i: i - 1 for i in range(1, 18)}          # 17-class: preserve per-leaf identity


class MaizeInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 17,
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        **kwargs
    ):
        class_mapping = kwargs.get("_class_mapping", CLASS_MAPPING_2)
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():
            if cls_id not in class_mapping:
                continue

            segmentation = polygons_by_id[label_id]
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles) if isinstance(rles, list) else rles

            masks.append(tv_tensors.Mask(coco_mask.decode(rle), dtype=torch.bool))
            labels.append(class_mapping[cls_id])
            is_crowd.append(is_crowd_by_id[label_id])

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        class_mapping = CLASS_MAPPING_17 if self.num_classes == 17 else CLASS_MAPPING_2

        def target_parser_with_mapping(*args, **kwargs):
            kwargs["_class_mapping"] = class_mapping
            return MaizeInstance.target_parser(*args, **kwargs)

        dataset_kwargs = {
            "img_suffix": ".png",
            "target_parser": target_parser_with_mapping,
            "only_annotations_json": True,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./images/train"),
            annotations_json_path_in_zip=Path("./annotations/instances_train.json"),
            target_zip_path=Path(self.path, "maize_annotations.zip"),
            zip_path=Path(self.path, "maize_train.zip"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./images/val"),
            annotations_json_path_in_zip=Path("./annotations/instances_val.json"),
            target_zip_path=Path(self.path, "maize_annotations.zip"),
            zip_path=Path(self.path, "maize_val.zip"),
            **dataset_kwargs,
        )
        self.test_dataset = Dataset(
            img_folder_path_in_zip=Path("./images/test"),
            annotations_json_path_in_zip=Path("./annotations/instances_test.json"),
            target_zip_path=Path(self.path, "maize_annotations.zip"),
            zip_path=Path(self.path, "maize_test.zip"),
            **dataset_kwargs,
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )