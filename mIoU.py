import torch
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


CLASS_NAMES = {
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

def compute_IoU_values(input_path, num_classes, ignore_index, output_path=None):   
    
    """ 
        - input_path: image npy. files folder
        - num_classes: number of classes in the dataset. (cityscapes: 20, BDD & Mapillary: 19)
        - ignore_index: which index to be ignored (cityscapes & mapillary: 19, BDD: 255)
        - output_path: folder in which miou_data .npy file will be saved ('result files' folder)
         
        - this functions computes:
            - overall IoU of the entire dataset
            - overall IoU of for each class
            - IoU of each individual image
            - classes IoU for each image
    
        returns miou_data dictionary
    """
    
    # Set device
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Compute mIoU values
    overall_miou = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=ignore_index).to(mydevice)

    overall_iou_for_classes = MulticlassJaccardIndex(num_classes=num_classes, average='none', ignore_index= ignore_index).to(mydevice)

    file_list = [f for f in os.listdir(input_path) if f.endswith('.npy')]

    for idx, filename in tqdm(enumerate(file_list)):
        filename = os.path.join(input_path, filename)
        with open(filename, "rb") as imgfile:
            img = np.load(imgfile, allow_pickle=True).item()
            pred = img['seg_map'].float()
            smnt = img['smnt']

            # Update overall mIoU of the dataset 
            overall_miou.update(pred, smnt)

            # update mIoU value for each class
            overall_iou_for_classes.update(pred, smnt)


            # find the mIoU value per image
            iou_of_img = JaccardIndex(task='multiclass', num_classes= num_classes, ignore_index= ignore_index).to(mydevice)
            iou_of_img.update(pred, smnt)
            iou_of_img = (iou_of_img.compute().item()) * 100
            img['iou_of_img'] = iou_of_img


            # calculate the iou of each class for each image
            classes_iou_per_img = MulticlassJaccardIndex(num_classes= num_classes, average='none', ignore_index= ignore_index).to(mydevice)
            classes_iou_per_img.update(pred, smnt)
            classes_iou_per_img = (classes_iou_per_img.compute()) * 100
            img['iou_per_class'] = classes_iou_per_img
   
        np.save(filename, img)
            

    # Compute overall mIoU of the dataset  
    overall_miou = (overall_miou.compute().item()) * 100

    # Compute the mIoU for each class and put them in classes_dict_sorted sorted by mIoU score
    overall_iou_for_classes = (overall_iou_for_classes.compute()) * 100

    miou_data = {'overall_miou':overall_miou, 
                 'overall_iou_for_classes':overall_iou_for_classes.cpu()
                 }
    
    if output_path is not None:
        np.save(output_path + "/miou_data.npy", miou_data) 
    
    print(f"Computation is Complete! mIoU data is saved on disk. Overall mIoU value: {overall_miou}")

    return miou_data



def plot_iou_graphs(dataset_name, model_name ,input_path, output_path):
    """
        - dataset_name
        - model_name
        - input_path : path of file "miou_data.npy"
        - output_path : path to save the graph figures
    
    """
    iou_data_dict = np.load(input_path, allow_pickle=True).item()
    overall_miou = round(iou_data_dict['overall_miou'], 2)
    overall_iou_for_classes = iou_data_dict['overall_iou_for_classes']

    class_iou_dict = {}

    for index, class_name in enumerate(CLASS_NAMES):
        class_iou_dict[CLASS_NAMES[index]] = overall_iou_for_classes[index].item()

    class_iou_dict = dict(reversed(sorted(class_iou_dict.items(), key=lambda item: item[1], reverse=True)))

    for item in class_iou_dict:
        print(f"{item}: {class_iou_dict[item]:.2f}")


    # Extract class names and IoU values
    classes = list(class_iou_dict.keys())
    iou_scores = list(class_iou_dict.values())

    # Plotting
    plt.figure(figsize=(12, 8))
    bars = plt.bar(classes, iou_scores, color=['skyblue', 'lightgreen', 'coral', 'lightpink', 'lightskyblue',
                                            'lightcoral', 'lightgreen', 'lightpink', 'lightskyblue', 'lightcoral',
                                            'lightgreen', 'lightpink', 'lightskyblue', 'lightcoral', 'lightgreen',
                                            'lightpink', 'lightskyblue', 'lightcoral', 'lightgreen'])

    # Highlighting highest and lowest classes
    highest_class = max(class_iou_dict, key=class_iou_dict.get)
    lowest_class = min(class_iou_dict, key=class_iou_dict.get)

    # Highlighting highest and lowest bars with different colors
    for bar, class_name in zip(bars, classes):
        if class_name == highest_class:
            bar.set_color('green')
        elif class_name == lowest_class:
            bar.set_color('red')

    plt.xlabel('Class Names')
    plt.ylabel('IoU Values')
    plt.title(f'(Overall IoU of {model_name} model on {dataset_name} dataset = {overall_miou}) \n IoU of Classes:')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.grid(axis='y')

    # Displaying exact numerical values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{output_path}/IoU_Class.png')


# Main 
# --------------
if __name__ == "__main__":
    def main(dataset_name, model_name, num_classes, ignore_index, load_miou):
        input_folder = f'/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/image_npfiles' 
        output_folder = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/result_files"
        
        if not load_miou:
            print(compute_IoU_values(input_folder, num_classes, ignore_index, output_folder))

        plot_iou_graphs(dataset_name = dataset_name, model_name = model_name, input_path= output_folder + "/miou_data.npy", output_path= f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/graphs")

    

    main("Cityscapes","Gmmseg", num_classes=20, ignore_index=19, load_miou= False)
