import json
import pathlib
from abc import ABC, abstractmethod
from typing import Any
from typing import Iterable

import warnings

import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval


class COCO(_COCO):
    """This class is almost the same as official pycocotools package.

    It implements some snake case function aliases. So that the COCO class has
    the same interface as LVIS class.
    """

    def __init__(self, annotation_file=None):
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


# just for the ease of import
COCOeval = _COCOeval


class Categories:

    def __init__(self, bases: Iterable[str], novels: Iterable[str]) -> None:
        self._bases = tuple(bases)
        self._novels = tuple(novels)

    @property
    def bases(self) -> tuple[str, ...]:
        return self._bases

    @property
    def novels(self) -> tuple[str, ...]:
        return self._novels

    @property
    def all_(self) -> tuple[str, ...]:
        return self._bases + self._novels

    @property
    def num_bases(self) -> int:
        return len(self._bases)

    @property
    def num_novels(self) -> int:
        return len(self._novels)

    @property
    def num_all(self) -> int:
        return len(self.all_)


coco = Categories(
    bases=(
        'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
        'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
        'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
        'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
        'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase',
        'toothbrush'
    ),
    novels=(
        'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
        'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
        'sink', 'scissors'
    ),
)


Data = dict[str, Any]


class Builder(ABC):

    def __init__(self, categories: Categories, root: str) -> None:
        self._categories = categories
        self._root = pathlib.Path(root)

    @abstractmethod
    def _load(self, file: pathlib.Path) -> Data:
        pass

    def _map_cat_ids(self, data: Data, cat_oid2nid: dict[int, int]) -> None:
        for cat in data['categories']:
            cat['id'] = cat_oid2nid[cat['id']]
        for ann in data['annotations']:
            ann['category_id'] = cat_oid2nid[ann['category_id']]

    def _dump(self, data: Data, file: pathlib.Path, suffix: str) -> None:
        file = file.with_stem(f'{file.stem}.{suffix}')
        with file.open('w') as f:
            json.dump(data, f, separators=(',', ':'))

    def _filter_anns(self, data: Data) -> Data:
        anns = [
            ann for ann in data['annotations']
            if ann['category_id'] < self._categories.num_bases
        ]
        return data | dict(annotations=anns)

    def _filter_imgs(self, data: Data) -> Data:
        img_ids = {ann['image_id'] for ann in data['annotations']}
        imgs = [img for img in data['images'] if img['id'] in img_ids]
        return data | dict(images=imgs)

    def build(self, filename: str, min: bool) -> None:
        file = self._root / filename
        data = self._load(file)

        cat_oid2nid = {  # nid = new id, oid = old id
            cat['id']: self._categories.all_.index(cat['name'])
            for cat in data['categories']
        }
        self._map_cat_ids(data, cat_oid2nid)
        data['categories'] = sorted(
            data['categories'], key=lambda cat: cat['id']
        )

        self._dump(data, file, str(self._categories.num_all))
        filtered_data = self._filter_anns(data)
        self._dump(filtered_data, file, str(self._categories.num_bases))
        if min:
            filtered_data = self._filter_imgs(data)
            self._dump(filtered_data, file, f'{self._categories.num_all}.min')


class COCOBuilder(Builder):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(coco, *args, **kwargs)

    def _load(self, file: pathlib.Path) -> Data:
        data = COCO(file)
        cat_ids = data.get_cat_ids(cat_names=self._categories.all_)
        ann_ids = data.get_ann_ids(cat_ids=cat_ids)
        img_ids = data.get_img_ids()
        cats = data.load_cats(cat_ids)
        anns = data.load_anns(ann_ids)
        imgs = data.load_imgs(img_ids)
        return dict(categories=cats, annotations=anns, images=imgs)


def main() -> None:
    coco_builder = COCOBuilder('/home/shenghao/dataset/coco/annotations')
    coco_builder.build('instances_val2017.json', True)
    coco_builder.build('instances_train2017.json', False)


if __name__ == '__main__':
    main()
