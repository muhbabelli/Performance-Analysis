import argparse
from DataContainerClass import DataContainer
from segmentation_module import SegmentationModel
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.cityscapes import Cityscapes
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import os
import sys
os.chdir("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets")
print("Current Working Directory:", os.getcwd())

def main(args):
    #torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SegmentationModel.load_from_checkpoint(args.ckpt)
    model.hparams.DATA.DATASET_ROOT = args.cityscapes_root
    
    
    transform=A.Compose(
                [
                    A.Resize(height=952, width=1288, always_apply=True),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                    
                ]
            )
    
    """valid_dataset = Cityscapes(
        hparams=model.hparams.DATA,
        transform=transform,
        split='val',
    )"""

    """MAPILLARY_PATH = '/datasets/mapillary-vistas-v2.0'
    sys.path.append(os.getcwd())
    from Mapillary.Mapillary import Mapillary
    from Mapillary.mapillary_to_cityscapes import mapillary_to_cityscapes_mapping
    valid_dataset = Mapillary(MAPILLARY_PATH, transform= transform, mode='val', labels_mode="v1")"""

    # Load dataset
    BDD100K_PATH = '/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/BDD/bdd100k_val_dataset'
    images_path = f"{BDD100K_PATH}/images"
    labels_path = f"{BDD100K_PATH}/labels"
    
    
    sys.path.append(os.getcwd())
    from BDD.BDD100K import BDD100KDataset
    valid_dataset = BDD100KDataset(images_path, labels_path, transform)

    dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
    
    img_names = valid_dataset.get_image_names()
    img_paths = valid_dataset.get_image_paths()
   
    output_folder = "/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/BDD/Gmmseg/data/image_npfiles"
    
   
    with torch.no_grad():
        model.to(device)
        from tqdm import tqdm
        for idx, data in tqdm(enumerate(dataloader)):
            image,smnt = data
            image = image.to(device)
            smnt = smnt.to(device)
            pred = model.inference_sample(image, smnt)
            print(pred.shape)
            if pred.shape != torch.Size([19, 952, 1288]):
                pred = pred.squeeze()
            pred = pred.argmax(0, keepdim=True)
            
            img_name, img_path = img_names[idx], img_paths[idx]
            

            element = DataContainer(img_name, img_path, smnt, pred)
        
            element.save_to_file(output_folder)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cityscapes Metrics")
    
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint",
        default= ""

    )

    parser.add_argument(
        "--cityscapes_root",
        type=str,
        help="Path to the cityscapes dataset"
    )

    parser.add_argument(
        "--devices",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(args)