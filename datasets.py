# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import os.path as osp
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

import mmcv
from mmcv.fileio import FileClient
import json
from PIL import Image
from abc import abstractmethod
import torch.utils.data as data
import logging

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "CEPH22k":
        root = args.data_path
        dataset = ImageCephDataset(root, transform=transform)
        nb_classes = len(dataset.parser.class_to_idx.keys())
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ImageCephDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            annotation_root='/mnt/lustre/share_data/wangtianyu.vendor/dataset',
            is_training=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = ParserCephImage(root=root, annotation_root=annotation_root)
        self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class Parser:
    def __init__(self):
        pass

    @abstractmethod
    def _filename(self, index, basename=False, absolute=False):
        pass

    def filename(self, index, basename=False, absolute=False):
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        return [self._filename(index, basename=basename, absolute=absolute) for index in range(len(self))]


class ParserCephImage(Parser):
    def __init__(
            self,
            root,
            annotation_root,
            io_backend='petrel',
            **kwargs):
        super().__init__()

        self.io_backend = io_backend
        self.file_client = None
        self.kwargs = kwargs

        self.root = root  # dataset:s3://imagenet22k
        with open(osp.join(annotation_root, 'class_to_idx.json'), 'r') as f:
            self.class_to_idx = json.loads(f.read())

        with open(osp.join(annotation_root, 'label.txt'), 'r') as f:
            self.samples = f.read().splitlines()

        self._consecutive_errors = 0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        filepath, target = self.samples[index].split(' ')
        filepath = osp.join(self.root, filepath)

        try:
            img_bytes = self.file_client.get(filepath)
            img = mmcv.imfrombytes(img_bytes)[:, :, ::-1]
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {filepath}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self))
            else:
                raise e
        self._consecutive_errors = 0

        img = Image.fromarray(img)
        target = self.class_to_idx[target]
        return img, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename, _ = self.samples[index].split(' ')
        filename = osp.join(self.root, filename)
        return filename
