import sys
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/models/Segmenter")
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/Mapillary_Vistas")
import torch
import numpy as np
from tqdm import tqdm
from DataContainerClass import DataContainer
import os
import torch
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from segm.model.utils import inference
from easydict import EasyDict as edict
from detectron2.checkpoint import DetectionCheckpointer
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datasets.Mapillary import mapillary_to_cityscapes


# Oneformer
def get_Oneformer_model(name='dec_layer_1'):
    sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/models/OneFormer")
    from train_net import Trainer, setup

    models_info = edict(
    swin_l=edict(
        ckpt='/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/OneFormer/model_dir/ckpt/model.pth',
        config='/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/OneFormer/model_dir/config/oneformer_swin_large_bs16_90k.yaml'
    ),
    convnext_l=edict(
        ckpt='/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/OneFormer/model_dir/ckpt/model.pth',
        config='/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/OneFormer/model_dir/config/mapillary_pretrain_oneformer_convnext_large_bs16_90k.yaml'
    ),
    convnext_xl=edict(
        ckpt='ckpts/250_16_convnext_xl_oneformer_cityscapes_90k/model.pth',
        config='configs/cityscapes/convnext/oneformer_convnext_xlarge_bs16_90k.yaml'
    ),
    dinat_l=edict(
        ckpt='ckpts/250_16_dinat_l_oneformer_cityscapes_90k/model.pth',
        config='configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml'
    ),
)
    
    config_path = models_info[name].config
    ckpt_path = models_info[name].ckpt
    
    if not os.path.exists(ckpt_path):
        ckpt_path = f"model_logs/{name}/model.pth"
        
    args = edict({'config_file': config_path, 'eval_only':True, 'opts':['MODEL.IS_TRAIN', 'False', 'MODEL.TEST.TASK', 'semantic']})
    
    config = setup(args)

    model = Trainer.build_model(config)
    DetectionCheckpointer(model, save_dir=config.OUTPUT_DIR).resume_or_load(
        ckpt_path, resume=False
    )
    model.cuda()
    _ = model.eval()
    
    return model, config
    

def get_Oneformer_logits(model, x, **kwargs):
    
    with torch.no_grad():
        out = model([{"image": x.squeeze(), 'task':'semantic'}])
        
    
    return out[0]['sem_seg']


def run_Oneformer_inference(model, transform, test_data, device, output_path,dataset_name=None):
    for idx, elm in enumerate(tqdm(test_data)):

        img, smnt = elm

        # Convert ground truth mask to tensor and process it
        #smnt = nazir.Cityscapes.encode_target(smnt)
        
        smnt = (smnt * 255).type(torch.int).squeeze()
        
        if dataset_name == "Mapillary":
            smnt = torch.tensor(mapillary_to_cityscapes.mapillary_to_cityscapes_mapping(smnt.tolist()), dtype=torch.int)
        
        smnt = smnt.to(device)

        # PreProcess Input
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        
        img = img.float().squeeze().to(device)
        

        # Make prediction
        logits = get_Oneformer_logits(model, img*255)

        pred = logits.argmax(0, keepdim=True).to(device)

        img_name = f"img_{idx}"
        img_path = test_data.images[idx]           # for mapillary
        #img_name = test_data.image_files[idx]
        #img_path = test_data.get_image_path(idx)    # for BDD  

        element = DataContainer(img_name, img_path, smnt, pred)
        
        
        element.save_to_file(output_path)


    


def run_Oneformer_inference_BDD(model, transform, test_data, device, output_path,dataset_name=None):
    for idx, elm in enumerate(tqdm(test_data)):
        print("idx: ", idx)

        img, smnt = elm

        #img_name = f"img_{idx}"
        #img_path = test_data.images[idx]           # for mapillary
        img_name = test_data.image_files[idx]
        img_path = test_data.get_image_path(idx)    # for BDD  
        
        # Convert ground truth mask to tensor and process it
        #smnt = nazir.Cityscapes.encode_target(smnt)
        
        smnt = (smnt * 255).type(torch.int).squeeze()
        
        if dataset_name == "Mapillary":
            smnt = torch.tensor(mapillary_to_cityscapes.mapillary_to_cityscapes_mapping(smnt.tolist()), dtype=torch.int)
        
        smnt = smnt.to(device)

        # PreProcess Input
        if not isinstance(img, torch.Tensor):
            img = transform(img)
        
        img = img.float().squeeze().to(device)
        

        # Make prediction
        logits = get_Oneformer_logits(model, img*255)

        pred = logits.argmax(0, keepdim=True).to(device)

        element = DataContainer(img_name, img_path, smnt, pred)
        
        element.save_to_file(output_path)

    print("Oneformer inference is done!")


def load_model_state_dict(path):
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    
    return state_dict['model']



# Segmenter 
def create_seg_map(img, model, variant, transform, device, normalization):
    # Convert image to tensor and process it
    if not isinstance(img, torch.Tensor):        
        img = transform(img)

    if img.size() == torch.Size([1, 960, 1280, 3]):
        img = img.float().squeeze().permute(2, 0, 1)
    
    img = F.normalize(img, normalization["mean"], normalization["std"])
    img = img.to(device)
    
    # Run inference, make predictions
    im_meta = dict(flip=False)
    logits = inference (
    model,
    [img],
    [im_meta],
    ori_shape=img.shape[2:4],
    window_size=variant["inference_kwargs"]["window_size"],
    window_stride=variant["inference_kwargs"]["window_stride"],
    batch_size=1,
    )

    # Store the segmentation map 
    seg_map = logits.argmax(0, keepdim=True)

    return seg_map


def run_Segmenter_inference(model, variant, normalization, transform, test_data, device, output_folder, dataset_name = None):  

    data_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=15)

    for idx, elm in enumerate(tqdm(data_loader)):
        img, smnt = elm

         # Convert ground truth mask to tensor and process it
        smnt = (smnt * 255).type(torch.int).squeeze()
        if dataset_name == "Mapillary":
            smnt = torch.tensor(mapillary_to_cityscapes.mapillary_to_cityscapes_mapping(smnt.tolist()), dtype=torch.int)
        smnt = smnt.to(device)

        pred = create_seg_map(img, model, variant, transform, device, normalization)        

        #img_name = f"img_{idx}"
        #img_path = test_data.images[idx]           # for mapillary
        img_name = test_data.image_files[idx]
        img_path = test_data.get_image_path(idx)    # for BDD   
        element = DataContainer(img_name, img_path, smnt, pred)
        
        
        element.save_to_file(output_folder)




# Gmmseg

def run_Gmmseg_inference(model, transform, dataset, device, img_names, img_paths, output_folder, dataset_name=None):
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
    
    with torch.no_grad():
        model.to(device)
        for idx, data in enumerate(dataloader):
            image,smnt = data
            image = image.to(device)

            if dataset_name == "Mapillary":
                smnt = torch.tensor(mapillary_to_cityscapes.mapillary_to_cityscapes_mapping(smnt.tolist()), dtype=torch.int)
            smnt = smnt.to(device)
            
            pred = model.inference_sample(image, smnt)
            
            img_name, img_path = img_names[idx], img_paths[idx]

            element = DataContainer(img_name, img_path, smnt, pred)
        
            element.save_to_file(output_folder)
            



# Prediction visualization 
def create_save_overlayed_imgs(img, seg_map, path, special_class = None):
    color_map = {
    0: [128, 64, 128],    # 0 : road
    1: [244, 35, 232],   # 1 : sidewalk
    2: [70, 70, 70],   # 2 : building
    3: [102, 102, 156],   # 3 : wall
    4: [190, 153, 153],  # 4 : fence
    5: [153, 153, 153],  # 5 : pole
    6: [250, 170, 30],  # 6 : traffic light
    7: [220, 220, 0],  # 7 : traffic sign
    8: [107, 142, 35],   # 8 : vegetation
    9: [152, 251, 152],   # 9 : terrain
    10: [70, 130, 180],   # 10 : sky
    11: [220, 20, 60],  # 11 : person
    12: [255, 0, 0],  # 12 : rider
    13: [0, 0, 142],  # 13 : car
    14: [0, 0, 70],  # 14 : truck
    15: [0, 60, 100],    # 15 : bus
    16: [0, 80, 100],    # 16 : train
    17: [0, 0, 230],    # 17 : motorcycle
    18: [119, 11, 32],  # 18 : bicycle
    }
    # Create an RGB image with labeled colors
    color_map[special_class] = [0, 204, 204] # type: ignore

    seg_map_np = seg_map.cpu().numpy()
    colored_map = np.zeros((seg_map_np.shape[1], seg_map_np.shape[2], 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_map[seg_map_np[0] == label] = color

    # Convert the numpy array to a PIL Image and save it
    predicted_image = Image.fromarray(colored_map)
    predicted_image = predicted_image.resize(img.size, Image.NEAREST)

    # Create an Overlayed Image with original and predicted images
    alpha = 0.35
    blended_image = Image.blend(img.convert("RGBA"), predicted_image.convert("RGBA"), alpha)
    blended_image.save(path, "PNG")


def pred_smnt_overlayed(img, smnt, seg_map, path, special_class = None):
    color_map = {
    0: [0, 0, 0],    # 0 : road
    1: [0, 0, 0],   # 1 : sidewalk
    2: [0, 0, 0],   # 2 : building
    3: [0, 0, 0],   # 3 : wall
    4: [0, 0, 0],  # 4 : fence
    5: [0, 0, 0],  # 5 : pole
    6: [0, 0, 0],  # 6 : traffic light
    7: [0, 0, 0],  # 7 : traffic sign
    8: [0, 0, 0],   # 8 : vegetation
    9: [0, 0, 0],   # 9 : terrain
    10: [0, 0, 0],   # 10 : sky
    11: [0, 0, 0],  # 11 : person
    12: [0, 0, 0],  # 12 : rider
    13: [0, 0, 0],  # 13 : car
    14: [0, 0, 0],  # 14 : truck
    15: [0, 0, 0],    # 15 : bus
    16: [0, 0, 0],    # 16 : train
    17: [0, 0, 0],    # 17 : motorcycle
    18: [0, 0, 0],  # 18 : bicycle
    }
    # Create an RGB image with all labels black except the special class
    color_map[special_class] = [255, 0, 0] # special class for pred : red
    seg_map_np = seg_map.cpu().numpy()
    colored_pred = np.zeros((seg_map_np.shape[1], seg_map_np.shape[2], 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_pred[seg_map_np[0] == label] = color



    color_map[special_class] = [0, 255, 0] # special class for ground truth : green
    smnt_np = smnt.unsqueeze(0).cpu().numpy()
    colored_smnt = np.zeros((smnt_np.shape[1], smnt_np.shape[2], 3), dtype=np.uint8)
    for label, color in color_map.items():
        colored_smnt[smnt_np[0] == label] = color

    # Convert the numpy array to a PIL Image and save it
    predicted_image = Image.fromarray(colored_pred)
    predicted_image = predicted_image.resize(img.size, Image.NEAREST)

    smnt_image = Image.fromarray(colored_smnt)
    smnt_image = smnt_image.resize(img.size, Image.NEAREST)

    # Create an Overlayed Image with original and predicted images
    alpha = 0.25
    blended_image = Image.blend(smnt_image.convert("RGBA"), predicted_image.convert("RGBA"), alpha)
    blended_image.save(path, "PNG")



# Confusion Matrix
def calculate_confusion_matrix(pred_tensor, gt_tensor, num_classes, ignore_index):
    # Flatten the tensors to 1D arrays
    pred_flat = pred_tensor.flatten().cpu()
    gt_flat = gt_tensor.flatten().cpu()
    
    # Mask out the ignore_index
    mask = (gt_flat != ignore_index)
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    return cm


def plot_confusion_matrix(model_name, dataset_name, confusion_matrix, classes, output_path, ignore_index=None, normalize=True):
    if normalize:
        # Convert to float
        confusion_matrix = confusion_matrix.astype('float')
        
        # Calculate row sums
        row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        row_sums[row_sums == 0] = epsilon
        
        # Normalize
        confusion_matrix = confusion_matrix / row_sums

    if ignore_index is not None:
        classes = [classes[c] for i, c in enumerate(classes) if i != ignore_index]
        confusion_matrix = np.delete(confusion_matrix, ignore_index, axis=0)
        confusion_matrix = np.delete(confusion_matrix, ignore_index, axis=1)

    df_cm = pd.DataFrame(confusion_matrix, classes, classes)
    
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, cmap = 'viridis', annot=False)
    plt.title(f"Confusion Matrix of {model_name} model on {dataset_name} dataset")
    plt.savefig(output_path)
    

def calculate_fp_fn_rates(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + np.diag(confusion_matrix))
    TP = np.diag(confusion_matrix)

    epsilon = 1e-7  # small constant to avoid division by zero
    FP_rate = FP / (FP + TN + epsilon)
    FN_rate = FN / (FN + TP + epsilon)

    FP_rate = np.around(FP_rate, 4)
    FN_rate = np.around(FN_rate, 4)
    
    return FP_rate, FN_rate
  

