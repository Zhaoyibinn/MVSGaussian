from lib.networks.mvsgs.network import Network
from lib.config import cfg_any, args_any

import cv2
import numpy as np
import torch
import os

def load_cfg_manul(cfg_any,root_path):
    cfg_any.train_dataset.data_root = root_path.rsplit("/",1)[0]
    


root_path  = "/home/zhaoyibin/3DRE/3DGS/FatesGS/DTU/set_23_24_33/scan24"
img_root_path = os.path.join(root_path,"images")

load_cfg_manul(cfg_any,root_path)



img_path_list = os.listdir(img_root_path)
img_list = []
for img_path in img_path_list:
    img_list.append(cv2.resize(cv2.imread(os.path.join(img_root_path,img_path)),(640,480)))


img_test = torch.tensor(img_list[0].transpose(2,0,1))





print("end")