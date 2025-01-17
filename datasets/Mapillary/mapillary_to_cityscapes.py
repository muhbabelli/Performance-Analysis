from tqdm import tqdm

import numpy as np


MAPILLARY_VISTAS_SEM_SEG_CATEGORIES = [
    {
        "color": [165, 42, 42],
        "instances": True,
        "readable": "Bird",
        "name": "animal--bird",
        "evaluate": True,
    },
    {
        "color": [0, 192, 0],
        "instances": True,
        "readable": "Ground Animal",
        "name": "animal--ground-animal",
        "evaluate": True,
    },
    {
        "color": [196, 196, 196],
        "instances": False,
        "readable": "Curb",
        "name": "construction--barrier--curb",
        "evaluate": True,
    },
    {
        "color": [190, 153, 153],
        "instances": False,
        "readable": "Fence",
        "name": "construction--barrier--fence",
        "evaluate": True,
    },
    {
        "color": [180, 165, 180],
        "instances": False,
        "readable": "Guard Rail",
        "name": "construction--barrier--guard-rail",
        "evaluate": True,
    },
    {
        "color": [90, 120, 150],
        "instances": False,
        "readable": "Barrier",
        "name": "construction--barrier--other-barrier",
        "evaluate": True,
    },
    {
        "color": [102, 102, 156],
        "instances": False,
        "readable": "Wall",
        "name": "construction--barrier--wall",
        "evaluate": True,
    },
    {
        "color": [128, 64, 255],
        "instances": False,
        "readable": "Bike Lane",
        "name": "construction--flat--bike-lane",
        "evaluate": True,
    },
    {
        "color": [140, 140, 200],
        "instances": True,
        "readable": "Crosswalk - Plain",
        "name": "construction--flat--crosswalk-plain",
        "evaluate": True,
    },
    {
        "color": [170, 170, 170],
        "instances": False,
        "readable": "Curb Cut",
        "name": "construction--flat--curb-cut",
        "evaluate": True,
    },
    {
        "color": [250, 170, 160],
        "instances": False,
        "readable": "Parking",
        "name": "construction--flat--parking",
        "evaluate": True,
    },
    {
        "color": [96, 96, 96],
        "instances": False,
        "readable": "Pedestrian Area",
        "name": "construction--flat--pedestrian-area",
        "evaluate": True,
    },
    {
        "color": [230, 150, 140],
        "instances": False,
        "readable": "Rail Track",
        "name": "construction--flat--rail-track",
        "evaluate": True,
    },
    {
        "color": [128, 64, 128],
        "instances": False,
        "readable": "Road",
        "name": "construction--flat--road",
        "evaluate": True,
    },
    {
        "color": [110, 110, 110],
        "instances": False,
        "readable": "Service Lane",
        "name": "construction--flat--service-lane",
        "evaluate": True,
    },
    {
        "color": [244, 35, 232],
        "instances": False,
        "readable": "Sidewalk",
        "name": "construction--flat--sidewalk",
        "evaluate": True,
    },
    {
        "color": [150, 100, 100],
        "instances": False,
        "readable": "Bridge",
        "name": "construction--structure--bridge",
        "evaluate": True,
    },
    {
        "color": [70, 70, 70],
        "instances": False,
        "readable": "Building",
        "name": "construction--structure--building",
        "evaluate": True,
    },
    {
        "color": [150, 120, 90],
        "instances": False,
        "readable": "Tunnel",
        "name": "construction--structure--tunnel",
        "evaluate": True,
    },
    {
        "color": [220, 20, 60],
        "instances": True,
        "readable": "Person",
        "name": "human--person",
        "evaluate": True,
    },
    {
        "color": [255, 0, 0],
        "instances": True,
        "readable": "Bicyclist",
        "name": "human--rider--bicyclist",
        "evaluate": True,
    },
    {
        "color": [255, 0, 100],
        "instances": True,
        "readable": "Motorcyclist",
        "name": "human--rider--motorcyclist",
        "evaluate": True,
    },
    {
        "color": [255, 0, 200],
        "instances": True,
        "readable": "Other Rider",
        "name": "human--rider--other-rider",
        "evaluate": True,
    },
    {
        "color": [200, 128, 128],
        "instances": True,
        "readable": "Lane Marking - Crosswalk",
        "name": "marking--crosswalk-zebra",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": False,
        "readable": "Lane Marking - General",
        "name": "marking--general",
        "evaluate": True,
    },
    {
        "color": [64, 170, 64],
        "instances": False,
        "readable": "Mountain",
        "name": "nature--mountain",
        "evaluate": True,
    },
    {
        "color": [230, 160, 50],
        "instances": False,
        "readable": "Sand",
        "name": "nature--sand",
        "evaluate": True,
    },
    {
        "color": [70, 130, 180],
        "instances": False,
        "readable": "Sky",
        "name": "nature--sky",
        "evaluate": True,
    },
    {
        "color": [190, 255, 255],
        "instances": False,
        "readable": "Snow",
        "name": "nature--snow",
        "evaluate": True,
    },
    {
        "color": [152, 251, 152],
        "instances": False,
        "readable": "Terrain",
        "name": "nature--terrain",
        "evaluate": True,
    },
    {
        "color": [107, 142, 35],
        "instances": False,
        "readable": "Vegetation",
        "name": "nature--vegetation",
        "evaluate": True,
    },
    {
        "color": [0, 170, 30],
        "instances": False,
        "readable": "Water",
        "name": "nature--water",
        "evaluate": True,
    },
    {
        "color": [255, 255, 128],
        "instances": True,
        "readable": "Banner",
        "name": "object--banner",
        "evaluate": True,
    },
    {
        "color": [250, 0, 30],
        "instances": True,
        "readable": "Bench",
        "name": "object--bench",
        "evaluate": True,
    },
    {
        "color": [100, 140, 180],
        "instances": True,
        "readable": "Bike Rack",
        "name": "object--bike-rack",
        "evaluate": True,
    },
    {
        "color": [220, 220, 220],
        "instances": True,
        "readable": "Billboard",
        "name": "object--billboard",
        "evaluate": True,
    },
    {
        "color": [220, 128, 128],
        "instances": True,
        "readable": "Catch Basin",
        "name": "object--catch-basin",
        "evaluate": True,
    },
    {
        "color": [222, 40, 40],
        "instances": True,
        "readable": "CCTV Camera",
        "name": "object--cctv-camera",
        "evaluate": True,
    },
    {
        "color": [100, 170, 30],
        "instances": True,
        "readable": "Fire Hydrant",
        "name": "object--fire-hydrant",
        "evaluate": True,
    },
    {
        "color": [40, 40, 40],
        "instances": True,
        "readable": "Junction Box",
        "name": "object--junction-box",
        "evaluate": True,
    },
    {
        "color": [33, 33, 33],
        "instances": True,
        "readable": "Mailbox",
        "name": "object--mailbox",
        "evaluate": True,
    },
    {
        "color": [100, 128, 160],
        "instances": True,
        "readable": "Manhole",
        "name": "object--manhole",
        "evaluate": True,
    },
    {
        "color": [142, 0, 0],
        "instances": True,
        "readable": "Phone Booth",
        "name": "object--phone-booth",
        "evaluate": True,
    },
    {
        "color": [70, 100, 150],
        "instances": False,
        "readable": "Pothole",
        "name": "object--pothole",
        "evaluate": True,
    },
    {
        "color": [210, 170, 100],
        "instances": True,
        "readable": "Street Light",
        "name": "object--street-light",
        "evaluate": True,
    },
    {
        "color": [153, 153, 153],
        "instances": True,
        "readable": "Pole",
        "name": "object--support--pole",
        "evaluate": True,
    },
    {
        "color": [128, 128, 128],
        "instances": True,
        "readable": "Traffic Sign Frame",
        "name": "object--support--traffic-sign-frame",
        "evaluate": True,
    },
    {
        "color": [0, 0, 80],
        "instances": True,
        "readable": "Utility Pole",
        "name": "object--support--utility-pole",
        "evaluate": True,
    },
    {
        "color": [250, 170, 30],
        "instances": True,
        "readable": "Traffic Light",
        "name": "object--traffic-light",
        "evaluate": True,
    },
    {
        "color": [192, 192, 192],
        "instances": True,
        "readable": "Traffic Sign (Back)",
        "name": "object--traffic-sign--back",
        "evaluate": True,
    },
    {
        "color": [220, 220, 0],
        "instances": True,
        "readable": "Traffic Sign (Front)",
        "name": "object--traffic-sign--front",
        "evaluate": True,
    },
    {
        "color": [140, 140, 20],
        "instances": True,
        "readable": "Trash Can",
        "name": "object--trash-can",
        "evaluate": True,
    },
    {
        "color": [119, 11, 32],
        "instances": True,
        "readable": "Bicycle",
        "name": "object--vehicle--bicycle",
        "evaluate": True,
    },
    {
        "color": [150, 0, 255],
        "instances": True,
        "readable": "Boat",
        "name": "object--vehicle--boat",
        "evaluate": True,
    },
    {
        "color": [0, 60, 100],
        "instances": True,
        "readable": "Bus",
        "name": "object--vehicle--bus",
        "evaluate": True,
    },
    {
        "color": [0, 0, 142],
        "instances": True,
        "readable": "Car",
        "name": "object--vehicle--car",
        "evaluate": True,
    },
    {
        "color": [0, 0, 90],
        "instances": True,
        "readable": "Caravan",
        "name": "object--vehicle--caravan",
        "evaluate": True,
    },
    {
        "color": [0, 0, 230],
        "instances": True,
        "readable": "Motorcycle",
        "name": "object--vehicle--motorcycle",
        "evaluate": True,
    },
    {
        "color": [0, 80, 100],
        "instances": False,
        "readable": "On Rails",
        "name": "object--vehicle--on-rails",
        "evaluate": True,
    },
    {
        "color": [128, 64, 64],
        "instances": True,
        "readable": "Other Vehicle",
        "name": "object--vehicle--other-vehicle",
        "evaluate": True,
    },
    {
        "color": [0, 0, 110],
        "instances": True,
        "readable": "Trailer",
        "name": "object--vehicle--trailer",
        "evaluate": True,
    },
    {
        "color": [0, 0, 70],
        "instances": True,
        "readable": "Truck",
        "name": "object--vehicle--truck",
        "evaluate": True,
    },
    {
        "color": [0, 0, 192],
        "instances": True,
        "readable": "Wheeled Slow",
        "name": "object--vehicle--wheeled-slow",
        "evaluate": True,
    },
    {
        "color": [32, 32, 32],
        "instances": False,
        "readable": "Car Mount",
        "name": "void--car-mount",
        "evaluate": True,
    },
    {
        "color": [120, 10, 10],
        "instances": False,
        "readable": "Ego Vehicle",
        "name": "void--ego-vehicle",
        "evaluate": True,
    },
    {
        "color": [0, 0, 0],
        "instances": False,
        "readable": "Unlabeled",
        "name": "void--unlabeled",
        "evaluate": False,
    },
]

MAPPILARY_TO_CITYSCAPES = {
    "bird": ("void", 255),
    "ground-animal": ("void", 255),
    "curb": ("sidewalk", 1),
    "fence": ("fence", 4),
    "guard-rail": ("void", 255),
    "other-barrier": ("void", 255),
    "wall": ("wall", 3),
    "bike-lane": ("void", 255),
    "crosswalk-plain": ("void", 255),
    "curb-cut": ("void", 255),
    "parking": ("void", 255),
    "pedestrian-area": ("void", 255),
    "rail-track": ("void", 255),
    "road": ("road", 0),
    "service-lane": ("void", 255),
    "sidewalk": ("sidewalk", 1),
    "bridge": ("void", 255),
    "building": ("building", 2),
    "tunnel": ("void", 255),
    "person": ("person", 11),
    "bicyclist": ("rider", 12),
    "motorcyclist": ("rider", 12),
    "other-rider": ("rider", 12),
    "crosswalk-zebra": ("road", 0),
    "general": ("road", 0),
    "mountain": ("void", 255),
    "sand": ("void", 255),
    "sky": ("sky", 10),
    "snow": ("void", 255),
    "terrain": ("terrain", 9),
    "vegetation": ("vegetation", 8),
    "water": ("void", 255),
    "banner": ("void", 255),
    "bench": ("void", 255),
    "bike-rack": ("void", 255),
    "billboard": ("void", 255),
    "catch-basin": ("void", 255),
    "cctv-camera": ("void", 255),
    "fire-hydrant": ("void", 255),
    "junction-box": ("void", 255),
    "mailbox": ("void", 255),
    "manhole": ("void", 255),
    "phone-booth": ("void", 255),
    "pothole": ("void", 255),
    "street-light": ("void", 255),
    "pole": ("pole", 5),
    "traffic-sign-frame": ("void", 255),
    "utility-pole": ("pole", 5),
    "traffic-light": ("traffic light", 6),
    "traffic-sign-back": ("void", 255),
    "traffic-sign-front": ("traffic sign", 7),
    "trash-can": ("void", 255),
    "bicycle": ("bicycle", 18),
    "boat": ("void", 255),
    "bus": ("bus", 15),
    "car": ("car", 13),
    "caravan": ("void", 255),
    "motorcycle": ("motorcycle", 17),
    "on-rails": ("train", 16),
    "other-vehicle": ("void", 255),
    "trailer": ("void", 255),
    "truck": ("truck", 14),
    "wheeled-slow": ("void", 255),
    "car-mount": ("void", 255),
    "ego-vehicle": ("void", 255),
    "unlabeled": ("void", 255)
}

def mapillary_id_to_name_dict():
    mydict = {}
    for idx, label in enumerate(MAPPILARY_TO_CITYSCAPES):
        mydict[idx] = label
    return mydict

mapillary_id_to_name = {
    0: 'bird', 
    1: 'ground-animal', 
    2: 'curb', 
    3: 'fence', 
    4: 'guard-rail', 
    5: 'other-barrier', 
    6: 'wall', 
    7: 'bike-lane', 
    8: 'crosswalk-plain', 
    9: 'curb-cut', 
    10: 'parking', 
    11: 'pedestrian-area', 
    12: 'rail-track', 
    13: 'road', 
    14: 'service-lane', 
    15: 'sidewalk', 
    16: 'bridge', 
    17: 'building', 
    18: 'tunnel', 
    19: 'person', 
    20: 'bicyclist', 
    21: 'motorcyclist', 
    22: 'other-rider', 
    23: 'crosswalk-zebra', 
    24: 'general', 
    25: 'mountain', 
    26: 'sand', 
    27: 'sky', 
    28: 'snow', 
    29: 'terrain', 
    30: 'vegetation', 
    31: 'water', 
    32: 'banner', 
    33: 'bench', 
    34: 'bike-rack', 
    35: 'billboard', 
    36: 'catch-basin', 
    37: 'cctv-camera', 
    38: 'fire-hydrant', 
    39: 'junction-box', 
    40: 'mailbox', 
    41: 'manhole', 
    42: 'phone-booth', 
    43: 'pothole', 
    44: 'street-light', 
    45: 'pole', 
    46: 'traffic-sign-frame', 
    47: 'utility-pole', 
    48: 'traffic-light', 
    49: 'traffic-sign-back', 
    50: 'traffic-sign-front', 
    51: 'trash-can', 
    52: 'bicycle', 
    53: 'boat', 
    54: 'bus', 
    55: 'car', 
    56: 'caravan', 
    57: 'motorcycle', 
    58: 'on-rails', 
    59: 'other-vehicle', 
    60: 'trailer', 
    61: 'truck', 
    62: 'wheeled-slow', 
    63: 'car-mount', 
    64: 'ego-vehicle', 
    65: 'unlabeled'
}

def mapillary_to_cityscapes_mapping(smnt):
    for i, sublist in enumerate(smnt):
        for j, label in enumerate(sublist):
            if label > 65 : 
                label = 65
            cityscapes_label = MAPPILARY_TO_CITYSCAPES[mapillary_id_to_name[label]][1]
            smnt[i][j] = cityscapes_label
    return smnt

