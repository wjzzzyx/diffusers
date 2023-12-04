import numpy as np
import os
from PIL import Image
import re
import scipy.io

from .base import TextDataset, TextInstance
from ocr.utils import strs
from ocr.transform.mixnet_transforms import Augmentation, BaseTransform


class TotalText(TextDataset):

    def __init__(
        self,
        data_root,
        mode,
        max_annotation,
        num_points,
        approx_factor,
        input_size,
        means,
        stds,
        ignore_list=None,
        load_memory=False
    ):
        super().__init__(max_annotation, num_points, approx_factor, input_size, means, stds)
        self.data_root = data_root
        self.mode = mode
        self.load_memory = load_memory

        if mode == 'train':
            self.transform = Augmentation(size=input_size, mean=means, std=stds)
        else:
            self.transform = BaseTransform(size=input_size, mean=means, std=stds)

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'train_image' if mode == 'train' else 'test_image')
        self.annotation_root = os.path.join(data_root, 'train_label_polygon' if mode == 'train' else 'test_label_polygon')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

        # self.balance_weight = True

    @staticmethod
    def parse_mat(mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = scipy.io.loadmat(mat_path + ".mat")
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        # lines = libio.read_lines(gt_path + ".txt")
        with open(gt_path + ".txt", 'rU') as f:
            lines = f.readlines()
        polygons = []
        for line in lines:
            line = strs.remove_all(line, '\xef\xbb\xbf')
            gt = line.split(',')
            xx = gt[0].replace("x: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            yy = gt[1].replace("y: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            try:
                xx = [int(x) for x in re.split(r" *", xx)]
                yy = [int(y) for y in re.split(r" *", yy)]
            except:
                xx = [int(x) for x in re.split(r" +", xx)]
                yy = [int(y) for y in re.split(r" +", yy)]
            if len(xx) < 4 or len(yy) < 4:  # too few points
                continue
            text = gt[-1].split('\'')[1]
            try:
                ori = gt[-2].split('\'')[1]
            except:
                ori = 'c'
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))
        # print(polygon)
        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = np.array(Image.open(image_path))

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        # image_id = self.image_list[item]
        # image_path = os.path.join(self.image_root, image_id)
        #
        # # Read image data
        # image = pil_load_img(image_path)
        #
        # # Read annotation
        # annotation_id = self.annotation_list[item]
        # annotation_path = os.path.join(self.annotation_root, annotation_id)
        # polygons = self.parse_mat(annotation_path)

        if self.load_memory:
            data = self.datas[item]
        else:
            data = self.load_img_gt(item)

        if self.mode == 'train':
            return self.get_training_data(data["image"], data["polygons"],
                                          image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(data["image"], data["polygons"],
                                      image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)
