'''Finetune Stage'''

import os
import mxnet as mx
import numpy as np
from gluoncv.data.mscoco.utils import try_import_pycocotools
from gluoncv.data.base import VisionDataset
from gluoncv.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from utils import convert_gt_to_coco


class COCODetection(VisionDataset):
    CLASSES = ['person']

    def __init__(self,
                 data_path=os.path.join('/', 'opt', 'ml', 'input', 'data', 'train'),
                 channel='train',
                 image_path=os.path.join('/', 'opt', 'ml', 'input', 'data', 'images'),
                 field_name='labels',
                 transform=None):
        super(COCODetection, self).__init__(data_path)
        self._data_path = data_path
        self._channel = channel
        self._image_path = image_path
        self._field_name = field_name
        self._transform = transform
        # to avoid trouble, we always use contiguous IDs except dealing with cocoapi
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._coco = []
        self._items, self._labels, self._im_aspect_ratios = self._load_data()

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files. \
                Please use single JSON dataset and evaluate one by one".format(len(self._coco)))
        return self._coco[0]

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    @property
    def channel(self):
        return self._channel

    @property
    def data_path(self):
        return self._data_path

    @property
    def image_path(self):
        return self._image_path

    @property
    def field_name(self):
        return self._field_name

    @property
    def annotation_path(self):
        return 'coco.json'

    def get_im_aspect_ratio(self):
        """Return the aspect ratio of each image in the order of the raw data."""
        if self._im_aspect_ratios is not None:
            return self._im_aspect_ratios
        self._im_aspect_ratios = [None] * len(self._items)
        for i, img_path in enumerate(self._items):
            with Image.open(img_path) as im:
                w, h = im.size
                self._im_aspect_ratios[i] = 1.0 * w / h

        return self._im_aspect_ratios

    def _parse_image_path(self, entry):
        """How to parse image dir and path from entry.
        Parameters
        ----------
        entry : dict
            COCO entry, e.g. including width, height, image path, etc..
        Returns
        -------
        abs_path : str
            Absolute path for corresponding image.
        """
        return entry['coco_url']

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        label = self._labels[idx]
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, np.array(label).copy()

    def _load_data(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        im_aspect_ratios = []

        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO

        convert_gt_to_coco(
            self.data_path,
            self.channel,
            self.image_path,
            self.field_name,
            self.annotation_path
        )

        anno = os.path.join(self.data_path, self.annotation_path)
        _coco = COCO(anno)
        self._coco.append(_coco)
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        if not classes == self.classes:
            raise ValueError("Incompatible category names with COCO: ")
        assert classes == self.classes
        json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}
        if self.json_id_to_contiguous is None:
            self.json_id_to_contiguous = json_id_to_contiguous
            self.contiguous_id_to_json = {
                v: k for k, v in self.json_id_to_contiguous.items()}
        else:
            assert self.json_id_to_contiguous == json_id_to_contiguous

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            abs_path = self._parse_image_path(entry)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_bbox(_coco, entry)
            if not label:
                continue
            im_aspect_ratios.append(float(entry['width']) / entry['height'])
            items.append(abs_path)
            labels.append(label)
        return items, labels, im_aspect_ratios

    def _check_load_bbox(self, coco, entry):
        """Check and load ground-truth labels"""
        entry_id = entry['id']
        # fix pycocotools _isArrayLike which don't work for str in python3
        entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
        ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if xmax > xmin and ymax > ymin:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid])
        return valid_objs
