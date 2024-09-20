import torch
import os
import numpy as np
from datasets.cityscapes import Cityscapes
from collections import namedtuple
from tqdm import tqdm
import sys
import albumentations as A
import torch.nn.functional as F

os.chdir("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets")
sys.path.append(os.getcwd())
from Mapillary.Mapillary import Mapillary
from Mapillary.mapillary_to_cityscapes import mapillary_to_cityscapes_mapping


CityscapesClass = namedtuple(
        "CityscapesClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )
classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass(
            "rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)
        ),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass(
            "building", 11, 2, "construction", 2, False, False, (70, 70, 70)
        ),
        CityscapesClass(
            "wall", 12, 3, "construction", 2, False, False, (102, 102, 156)
        ),
        CityscapesClass(
            "fence", 13, 4, "construction", 2, False, False, (190, 153, 153)
        ),
        CityscapesClass(
            "guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)
        ),
        CityscapesClass(
            "bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)
        ),
        CityscapesClass(
            "tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)
        ),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass(
            "polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)
        ),
        CityscapesClass(
            "traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)
        ),
        CityscapesClass(
            "traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)
        ),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass(
            "license plate", -1, 255, "vehicle", 7, False, True, (0, 0, 142)
        ),
    ]
def encode_target(target):
        id_to_train_id = np.array([c.train_id for c in classes])
        target = id_to_train_id[np.array(target)]
        target[target == 255] = 19
        return target

filenames = os.listdir("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/Mapillary/Gmmseg/data/image_npfiles")
print(len(filenames))

for idx, f in tqdm(enumerate(filenames)):
    filename = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/Mapillary/Gmmseg/data/image_npfiles/{f}"
    with open(filename, "rb") as imgfile:
        img = np.load(imgfile, allow_pickle=True).item()
        smnt = img['smnt']
        pred = img['seg_map']
       
        
        smnt = torch.tensor(mapillary_to_cityscapes_mapping(smnt.squeeze().tolist()), dtype=torch.int).unsqueeze(0)
        pred = pred.squeeze().argmax(0, keepdim=True)


        #correct_predictions = (pred == smnt)

        # Step 2: Calculate the accuracy
        #accuracy = correct_predictions.float().mean().item()

        # Print the accuracy
        #print(f'Accuracy: {accuracy * 100:.2f}%')

        img['smnt'] = smnt.cuda()
        img['seg_map'] = pred.cuda()

        np.save(filename, img)
