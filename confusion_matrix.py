import sys
sys.path.append("/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/Cityscapes")
import myfunctions
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
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

def compute_conf_mat(model_name, dataset_name, input_path, plot_output_path, confmat_output_path, **kwargs):
    """
        - input_path: path of folder containing images .npy files
        - plot_output_path: path to which confmat plot figure will be saved
        - confmat_output_path: path to which confmat tensor will be saved as .npy file 

        return total_conf_mat
    """

    total_conf_mat = np.zeros((20, 20))

    file_list = [f for f in os.listdir(input_path) if f.endswith('.npy')]

    for idx, filename in tqdm(enumerate(file_list)):
        filename = os.path.join(input_path, file_list[idx])
        with open(filename, "rb") as imgfile:
            img = np.load(imgfile, allow_pickle=True).item()
            pred = img['seg_map']
            smnt = img['smnt']
            
            conf_mat = myfunctions.calculate_confusion_matrix(pred, smnt, num_classes = 20, ignore_index=19)
            img['conf_matrix'] = conf_mat
        
            fp_rate, fn_rate = myfunctions.calculate_fp_fn_rates(conf_mat)
            img['fp_rate'] = fp_rate
            img['fn_rate'] = fn_rate

            np.save(filename, img)
            
            total_conf_mat += conf_mat

    # Plot and save
    myfunctions.plot_confusion_matrix(model_name, dataset_name, total_conf_mat, class_names, f"{plot_output_path}/total_confusion_matrix.png", 19)
    
    # Save confmat 
    np.save(f'{confmat_output_path}/confusion_matrix.npy', total_conf_mat)
    
    return total_conf_mat


def compute_fp_fn_rates(confmat, output_path):
    """
        confmat : confmat tensor
        output_path : path of folder to which fp_fn_rates dictionary will be saved as .npy file 

        returns fp_fn_rates dict
    """

    total_fp_rate, total_fn_rate = myfunctions.calculate_fp_fn_rates(confmat)

    fp_fn_rates = {"total_fp_rate": total_fp_rate,
                   "total_fn_rate": total_fn_rate}
    
    
    np.save(f'{output_path}/fp_fn_rates.npy', fp_fn_rates)
    
    return fp_fn_rates



def confused_distr(confmat):
    overall_distribution = {}
    for index1, trueclass in enumerate(confmat):
        if index1 == 19:
            continue
        sum = trueclass.sum()
        fn = sum - trueclass[index1]
        class_distribution = {}
        for index2, prediction in enumerate(trueclass):
            if (index1==index2) or index2 == 19:
                continue
            class_distribution[class_names[index2]] = round(prediction / fn, 3)

        overall_distribution[class_names[index1]] = dict(reversed(sorted(class_distribution.items(), key=lambda x:x[1])))
    
    return overall_distribution
    

def plot_confused_distr(model_name, dataset_name, confused_distr, total_FN_rate ,output_path):
        # Plotting
    for idx, class_ in enumerate(confused_distr):
        classes_dict = confused_distr[class_]
        plt.figure(figsize=(12, 8))
        bars = plt.bar(classes_dict.keys(), list(np.array(list(classes_dict.values())) * 100) , color=['skyblue', 'lightgreen', 'coral', 'lightpink', 'lightskyblue',
                                                'lightcoral', 'lightgreen', 'lightpink', 'lightskyblue', 'lightcoral',
                                                'lightgreen', 'lightpink', 'lightskyblue', 'lightcoral', 'lightgreen',
                                                'lightpink', 'lightskyblue', 'lightcoral', 'lightgreen'])

        plt.xlabel('Class Names')
        plt.ylabel("Confusion Rate (% of FN)")
        plt.title(f'{model_name} - {dataset_name} \nThe model confuses the class {class_.upper()} (total FN rate: %{round((total_FN_rate[list(class_names.keys())[list(class_names.values()).index(class_)]] * 100),2)}) with the following classes: ')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.grid(axis='y')

        # Displaying exact numerical values on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

        plt.tight_layout()
        plt.show()
        plt.savefig(f'{output_path}/{idx+1}-confusions_{class_}.png')



# Main
# -------------
def main(model_name, dataset_name, load_conf_mat, load_fp_fn_rates):
    input_path = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/image_npfiles"
    tensor_output_path = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/result_files"
    plot_output_path = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/graphs"
    
    if not load_conf_mat:
        confmat = compute_conf_mat(model_name, dataset_name, input_path, plot_output_path, tensor_output_path)
    else:
        confmat_path = f"{tensor_output_path}/confusion_matrix.npy"
        confmat = np.load(confmat_path,allow_pickle=True)


    if not load_fp_fn_rates:
        fp_fn_rates_dict = compute_fp_fn_rates(confmat, tensor_output_path)
    else:
        fp_fn_rates_dict_path = f"{tensor_output_path}/fp_fn_rates.npy"
        fp_fn_rates_dict = np.load(fp_fn_rates_dict_path,allow_pickle=True).item()
    
    

    # Compute and Plot Confused_distr
    confused_distrib = confused_distr(confmat)
    fn_rates = fp_fn_rates_dict['total_fn_rate']
    
    output_path = f"/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/datasets/{dataset_name}/{model_name}/data/graphs/class_confusions"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    plot_confused_distr(model_name, dataset_name, confused_distrib, fn_rates ,output_path)


if __name__ == "__main__":
    main("Gmmseg", "Cityscapes",load_conf_mat = False, load_fp_fn_rates= False)