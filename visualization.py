import sys
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/Cityscapes")
import mIoU
import os
import torch
import numpy as np
import myfunctions
from PIL import Image
from tqdm import tqdm

class_names = {
        0: 'road',
        1: 'sidewalk',
        2:'building',
        3: 'wall',
        4:'fence',
        5: 'pole',
        6: 'traffic light',
        7: 'traffic sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        }

def create_class_img_dict(input_path , output_folder_path, load_miou_data= False, miou_data_dict_path = None ):
    """
        input:
        - input_path: folder containing image .npy files
        - output_folder_path: folder in which output class_img_dict will be saved ('result files' folder)
        - load_miou_data: if False --> calls function 'compute_IoU_values', if True --> loads miou_data dict
        - miou_data_dict_path: folder from which miou_data dict will be loaded

        output:
        - classes_dict_sorted : dictionary of class numbers with corresponding IoU values, sorted based on IoU values
        - class_img_dict : {
                            (class number) :  {
                                (image name) : (image's iou value for that class)
                                ...
                                }
                            ...
                            } 

        
    """
    # Compute or load the mIoU data for each class and put them in classes_dict_sorted sorted by mIoU score
    image_npfiles_folder = input_path
    
    if not load_miou_data:
        iou_each_class = mIoU.compute_IoU_values(image_npfiles_folder)['overall_iou_for_classes']
    else:
        iou_each_class = np.load(f"{miou_data_dict_path}", allow_pickle=True).item()['overall_iou_for_classes']

    classes_dict = {}

    for num in range (0,19):
        classes_dict[num] = "{:.4f}".format(iou_each_class[num].item())

    classes_dict_sorted = dict(sorted(classes_dict.items(), key=lambda x:x[1]))
    del classes_dict # free space

    
    # img_classiou_dict = (image name) : (image's iou value for that class))
    # class_img_dict =  (class number) :  (img_classiou_dict) 
    class_img_dict = {}

    for class_ in tqdm(classes_dict_sorted):   
        img_classiou_dict = {}
        file_list = [f for f in os.listdir(image_npfiles_folder) if f.endswith('.npy')]
        for imgfile in file_list:
            imgfile = os.path.join(image_npfiles_folder, imgfile)
            img = np.load(imgfile, allow_pickle=True).item()

            if torch.any(img['smnt'] == class_):
                img_classiou_dict[img['img_name']] = img['iou_per_class'][class_].item()
        
        img_classiou_dict =  dict(sorted(img_classiou_dict.items(), key=lambda item: item[1])) # sort the images in the class according to the class iou value.

        class_img_dict[class_] = img_classiou_dict # assign the dictionary of this class as the value of the key of the larger dictionary
        

        if class_img_dict[class_] == []:
            class_img_dict.pop(class_)
            continue

    np.save(f"{output_folder_path}/class_img_dict.npy",class_img_dict)
    print("class_img_dict saved on disk ! ")

    return classes_dict_sorted, class_img_dict

def visualize_overlayed_images(image_npfiles_folder, classes_dict_sorted, class_img_dict, output_parent_folder):
    """
        input:
            - image_npfiles_folder : input folder containing image .npy files
            - classes_dict_sorted : dictionary contatining class numbers and their IoU values sorted. (output of create_class_img_dict)
            - class_img_dict: dictionary containing classes and images including these classes. (output of create_class_img_dict)
            - output_parent_folder : folder under which each class will have a folder containing the overlayed images.

        output:
            - each class will have a folder containing worst 30 images belonging to that class.
            - each image has two versions : one as real-pred overlayed, and the other is pred-smnt overlayed.
            - the function prints log info of the process  
    """
    
    # Save the worst 30 images of each class under a subdirectory for that class.
    for class_ in classes_dict_sorted:
        img_count = 1
        directory = f"{class_names[class_]}/"
        dir_path = os.path.join(output_parent_folder, directory)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for image in class_img_dict[class_]:

            if img_count > 30: # Save only the worst 30 images
                break

            filename = os.path.join(image_npfiles_folder, image)
            with open(filename, "rb") as imgfile:
                img = np.load(imgfile, allow_pickle=True).item()
                pil_image = Image.open(img['img_path'])
                smnt = img['smnt']
                seg_map = img['seg_map']
                pred_smnt_img_filename = img['img_name'] + "_abstract.png"
                overlayed_img_filename = img['img_name'] + "_real.png"
                overlayed_path = os.path.join(dir_path, overlayed_img_filename)
                pred_smnt_img_path = os.path.join(dir_path, pred_smnt_img_filename)
                myfunctions.create_save_overlayed_imgs(pil_image,seg_map, overlayed_path, special_class = class_)
                myfunctions.pred_smnt_overlayed(pil_image, smnt, seg_map, pred_smnt_img_path, special_class = class_)
                print(f"IMAGE SAVED ! IoU of class {class_names[class_]} in {img['img_name']}: {class_img_dict[class_][image]}")
                img_count += 1

            print("--------------")


# Main
# ---------------------------
            
def main(model_name, dataset_name):
    input_path = f'/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/image_npfiles'
    output_folder_path = f'/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/result_files'
    miou_data_dict_path = f'/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/result_files/miou_data.npy'
    
    classes_dict_sorted, class_img_dict = create_class_img_dict(input_path, output_folder_path, load_miou_data=True, miou_data_dict_path= miou_data_dict_path)
    
    overlayed_images_output_folder = f'/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/overlayed_imgs'
    visualize_overlayed_images(input_path, classes_dict_sorted, class_img_dict, overlayed_images_output_folder )


if __name__ == "__main__" :
    main(model_name="Gmmseg", dataset_name="Cityscapes")