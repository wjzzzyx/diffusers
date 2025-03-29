import contextlib
import io
import logging
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import pycocotools
import pycocotools.coco
import pycocotools.cocoeval

from segmentors.transform import Compose, RandomHorizontalFlip, LargeScaleJitter, ResizeLongestSide


logger = logging.getLogger(__name__)

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]


def load_coco_json(json_file, image_root):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = pycocotools.coco.COCO(json_file)
    
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [
    #   {
    #     'segmentation': [[192.81, 247.09, ... 219.03, 249.06]],
    #     'area': 1035.749,
    #     'iscrowd': 0,
    #     'image_id': 1268,
    #     'bbox': [192.81, 224.8, 74.73, 33.43],
    #     'category_id': 16,
    #     'id': 42986
    #   },
    #  ...
    # ]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    # if "minival" not in json_file:
    #     # The popular valminusminival & minival annotations for COCO2014 contain this bug.
    #     # However the ratio of buggy annotations there is tiny and does not affect accuracy.
    #     # Therefore we explicitly white-list them.
    #     ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    #     assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
    #         json_file
    #     )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    num_instances_without_valid_segmentation = 0

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            obj["bbox"][2] = obj["bbox"][0] + obj["bbox"][2]
            obj["bbox"][3] = obj["bbox"][1] + obj["bbox"][3]

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = pycocotools.mask.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            objs.append(obj)
        
        if len(objs) > 0:
            record["annotations"] = objs
            dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return cat_ids, dataset_dicts


def convert_polygons_to_mask(polygons: list[torch.Tensor], height, width):
    polygons = [p.view(-1).numpy() for p in polygons]
    rles = pycocotools.mask.frPyObjects(polygons, height, width)
    mask = pycocotools.mask.decode(rles)
    if len(mask.shape) < 3:    # ?
        mask = mask[..., None]
    mask = torch.as_tensor(mask, dtype=torch.uint8)
    mask = mask.any(dim=2)
    return mask


def get_bbox_from_mask(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(xs) > 0 and len(ys) > 0:
        return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()])
    else:
        return torch.tensor([0., 0., 0., 0.])


def filter_empty_instances(classes: torch.Tensor, bboxes: torch.Tensor, masks: torch.Tensor):
    keep = ((bboxes[:, 2] - bboxes[:, 0]) > 1e-5) & ((bboxes[:, 3] - bboxes[:, 1]) > 1e-5)
    keep &= masks.any(dim=(1, 2)).bool()
    return classes[keep], bboxes[keep], masks[keep]


class COCOInstanceDataset(Dataset):
    def __init__(self, data_dir, anno_file, mode, target_size=1024, mask_pad_value=80):
        self.coco_anno_file = anno_file
        self.cat_ids, self.data = load_coco_json(anno_file, data_dir)
        self.mode = mode
        if self.mode == "train":
            self.target_size = target_size
            self.transform = Compose([
                RandomHorizontalFlip(),
                LargeScaleJitter(target_size, mask_pad_value=mask_pad_value),
            ])
        else:
            self.transform = ResizeLongestSide(target_size)
        self.dataset_id_to_contiguous_id = {c: i for i, c in enumerate(self.cat_ids)}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info = self.data[index]
        image = Image.open(info["file_name"]).convert("RGB")
        if image.width != info["width"] or image.height != info["height"]:
            raise ValueError("Image size mismatch with info.")
        image = TF.pil_to_tensor(image)
        
        if self.mode != "train":
            self.transform.reset(image)
            image = self.transform({"image": image})["image"]
            return {
                "image_id": info["image_id"],
                "image_fname": info["file_name"],
                "height": info["height"],
                "width": info["width"],
                "image": image,
            }
        
        height, width = image.shape[-2:]
        padding_mask = torch.ones((1, height, width), dtype=torch.uint8)
        annos = info["annotations"]
        annos = [obj for obj in annos if obj.get("iscrowd", 0) == 0]    # how many bboxes and polygons if iscrowd?
        self.transform.reset(image)
        image = self.transform({"image": image})["image"]
        padding_mask = self.transform({"mask": padding_mask})["mask"]
        padding_mask = ~padding_mask.bool()
        augs = list()
        for obj in annos:
            bboxes = torch.as_tensor(obj["bbox"]).unsqueeze(0)
            raw_mask = obj["segmentation"]
            if isinstance(raw_mask, list):    # a list of polygons
                polygons = [torch.as_tensor(p).view(-1, 2) for p in raw_mask]
                augs.append(self.transform({"bboxes": bboxes, "polygons": polygons}))
                augs[-1]["mask"] = convert_polygons_to_mask(
                    augs[-1].pop("polygons"), self.target_size, self.target_size
                )
            elif isinstance(raw_mask, dict):    # rle
                mask = pycocotools.mask.decode(raw_mask)
                augs.append(self.transform({"bboxes": bboxes, "mask": mask}))
        
        classes = torch.tensor([
            self.dataset_id_to_contiguous_id[obj["category_id"]] for obj in annos
        ])
        # bboxes = torch.stack([obj["bbox"] for obj in augs])
        # transform may cause the bbox to be loose
        bboxes = torch.stack([get_bbox_from_mask(obj["mask"]) for obj in augs])
        masks = torch.stack([obj["mask"] for obj in augs])
        classes, bboxes, masks = filter_empty_instances(classes, bboxes, masks)
        
        return {
            "image_id": info["image_id"],
            "image_fname": info["file_name"],
            "height": info["height"],
            "width": info["width"],
            "image": image,
            "classes": classes,
            "bboxes": bboxes,
            "masks": masks,
            "padding_mask": padding_mask
        }


def collate_fn(batch):
    return batch


def model_output_to_coco_format(image_id, classes, scores, bboxes, masks):
    contiguous_id_to_dataset_id = dict()
    for i, cat in enumerate(COCO_CATEGORIES):
        contiguous_id_to_dataset_id[i] = cat["id"]
    
    # convert model class id to dataset class id
    classes = classes.tolist()
    classes = [contiguous_id_to_dataset_id[c] for c in classes]
    scores = scores.tolist()
    # convert from xyxy bbox to xywh bbox
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes = bboxes.tolist()
    masks = masks.cpu().numpy()
    rles = list()    # ?
    for m in masks:
        m = np.array(m[:, :, None], order="F", dtype="uint8")
        rle = pycocotools.mask.encode(m)[0]
        # "counts" is an array encoded by pycocotools as a byte-stream. Python3's
        # json writer witch always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what 
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    
    results = list()
    for i in range(len(classes)):
        results.append({
            "image_id": image_id,
            "category_id": classes[i],
            "bbox": bboxes[i],
            "score": scores[i],
            "segmentation": rles[i]
        })
    return results


def coco_eval_instance_seg(anno_file, predictions):
    with contextlib.redirect_stdout(io.StingIO()):
        coco_api = pycocotools.coco.COCO(anno_file)
    
    assert(len(predictions) > 0)

    for p in predictions:
        p.pop("bbox", None)
    
    pred_api = coco_api.loadRes(predictions)
    evaluator = pycocotools.cocoeval.COCOeval(coco_api, pred_api, "segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    results = {
        metric: float(evaluator.stats[i] * 100 if evaluator.stats[i] >= 0 else "nan")
        for i, metric in enumerate(metrics)
    }

    # precisions: shape (iou, recall, cls, area range, max dets)
    precisions = evaluator.eval["precision"]
    results_per_category = dict()
    class_names = [x["name"] for x in COCO_CATEGORIES if x["isthing"] == 1]
    assert(len(class_names) == precisions.shape[2])
    for iclass, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, iclass, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category[f"AP-{name}"] = float(ap * 100)
    
    return results, results_per_category