import numpy as np

class DataContainer:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    
    def __init__(self, img_name, img_path=None, smnt=None, seg_map=None, iou_of_img=None, iou_per_class=None, conf_matrix = None, fp_rate = None, fn_rate = None):
        """
            - img_name includes the .jpg/.png extension
        """
        
        self.img_name = img_name
        self.img_path = img_path
        self.smnt = smnt
        self.seg_map = seg_map
        self.iou_of_img = iou_of_img
        self.iou_per_class = iou_per_class
        self.conf_matrix = conf_matrix
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate

    def save_to_file(self, filepath):
        """
            - filepath : folder to which data_dict will be saved as .npy file
            
        """
        data_dict = {
            "img_name" : self.img_name,
            "img_path" : self.img_path,
            "smnt" : self.smnt,
            "seg_map" : self.seg_map,
            "iou_of_img" : self.iou_of_img,
            "iou_per_class" : self.iou_per_class,
            "conf_matrix" : self.conf_matrix,
            "fp_rate" : self.fp_rate,
            "fn_rate" : self.fn_rate
        }
        
        np.save(f"{filepath}/{self.img_name[:-4]}",data_dict)
