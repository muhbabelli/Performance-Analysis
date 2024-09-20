import sys
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets")
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/models")
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/models/segmenter")
from pathlib import Path
import torch
from torchvision import transforms 
import myfunctions
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2


CITYSCAPES_PATH = '/datasets/cityscapes/'
MAPILLARY_PATH = '/datasets/mapillary-vistas-v2.0'
BDD100K_PATH = '/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/BDD/bdd100k_val_dataset'

def inference(dataset_name, model_name, output_folder, model_path = None, backbone = 'swin_l'):
    
    print("inference started")

    # Set device
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "Cityscapes":
        # transform ToTensor
        transform_dataset = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load dataset        
        dataset = datasets.Cityscapes(CITYSCAPES_PATH, split='val', mode='fine', target_type='semantic')
 
    
    elif dataset_name == "Mapillary":
        # transform ToTensor
        transform_dataset = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((960,1280))
        ])

        # Load dataset 
        from datasets.Mapillary.Mapillary import Mapillary
        dataset = Mapillary(MAPILLARY_PATH, transform= transform_dataset, mode='val', labels_mode="v1")

    
    elif dataset_name == "BDD100K":
        # Load dataset
        images_path = f"{BDD100K_PATH}/images"
        labels_path = f"{BDD100K_PATH}/labels"
        
        # transform ToTensor
        transform_dataset = transforms.Compose([
            transforms.ToTensor(),
        ])

        from datasets.BDD.BDD100K import BDD100KDataset
        dataset = BDD100KDataset(images_path, labels_path, transform_dataset)


    # Load the model and run inference
    if model_name == 'Segmenter':
        from segm.data.utils import STATS
        from segm.model.factory import load_model
        
        model_dir = Path(model_path).parent
        model, variant = load_model(model_path)
        model.to(mydevice).eval()
        

        # Normalization variables
        normalization_name = variant["dataset_kwargs"]["normalization"]
        normalization = STATS[normalization_name]

        # transform ToTensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        myfunctions.run_Segmenter_inference(model, variant, normalization, transform, dataset, 
                                            mydevice, output_folder)


    elif model_name == 'Oneformer':
        # transform ToTensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load the model 
        model, config = myfunctions.get_Oneformer_model(backbone)
        if dataset_name == "BDD100K":
            myfunctions.run_Oneformer_inference_BDD(model, transform, dataset, mydevice,
                                                        output_folder)
        else:
            myfunctions.run_Oneformer_inference(model, transform, dataset, mydevice,
                                                        output_folder)


    elif model_name == 'Gmmseg':
        
        from Gmmseg import segmentation_module
        model = segmentation_module.SegmentationModel.load_from_checkpoint(model_path)

        transform_dataset=A.Compose(
                    [
                        A.PadIfNeeded(
                            min_height=1036,
                            min_width=2058,
                            p=1.0,
                            mask_value=model.hparams.MODEL.IGNORE_INDEX,
                        ),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2(),
                        
                    ]
                )

        if dataset_name == "Cityscapes":
            from datasets.cityscapes import Cityscapes
            model.hparams.DATA.DATASET_ROOT = CITYSCAPES_PATH

            dataset = Cityscapes(
                hparams=model.hparams.DATA,
                transform=transform_dataset,
                split='val',
            )


        if dataset_name == "Mapillary":
            from datasets.Mapillary.Mapillary import Mapillary
            model.hparams.DATA.DATASET_ROOT = MAPILLARY_PATH

            dataset = Mapillary(
                MAPILLARY_PATH, 
                transform= transform_dataset,
                mode='val', 
                labels_mode="v1"
            )

        img_names = dataset.get_image_names()
        img_paths = dataset.get_image_paths()
            

        myfunctions.run_Gmmseg_inference(model, transform, dataset, mydevice,
                                        img_names, img_paths, output_folder)

    print(f"{model_name} inference on {dataset_name} is done!")

# Main
# -------------
def main(model_name, model_path, dataset_name, output_folder):
    
    inference(dataset_name, model_name, output_folder, model_path)


if __name__ == "__main__":
    model_name = "Gmmseg" 
    model_path = "/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/models/Gmmseg/ckpt/epoch=236-val_iou=0.83.ckpt"
    dataset_name = "Mapillary"
    output_folder = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/image_npfiles"

    main(model_name, model_path, dataset_name, output_folder)