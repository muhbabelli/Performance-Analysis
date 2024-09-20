import torch
from torchvision import datasets
from torchvision import transforms 
import myfunctions
import warnings
warnings.filterwarnings('ignore')


def inference(backbone, output_folder):
    """
        - model_name: Oneformer model name
        - output_folder: parent folder in which image .npy files will be saved 
    """
    # Load the model 
    model, config = myfunctions.get_Oneformer_model(backbone)

    # transform ToTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the data
    test_data = datasets.Cityscapes('/datasets/cityscapes/',split='val', mode='fine', target_type='semantic')

    # Set device
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run inference and save segmentation maps into .np files in 
        
    myfunctions.run_Oneformer_inference(model, transform, test_data, mydevice, output_folder)



# Main
model_name = 'swin_l'
output_folder = '/kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/Cityscapes/OneFormer/data/image_npfiles'
inference(model_name, output_folder)