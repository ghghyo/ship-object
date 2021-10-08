#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:53:02 2021

@author: yabubaker
"""

PATH = "app/weight_vv_pre"
import glob
import os
import pandas as pd
import torch
from PIL import Image, ImageDraw
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import app.transforms as T

from skimage import io


#our files we will use and exculde unlabeled data
names=[os.path.basename(x) for x in glob.glob("app/ships/*.jpg")]
gao = [string for string in names if "Gao_ship" in string]
sen = [string for string in names if "Sen_ship" in string]
#names= gao+sen

#testing
#names=[os.path.basename(x) for x in glob.glob("/home/yabubaker/Documents/thirdproject/ship_dataset_v0/*.tif")]
#names=[string for string in names if string not in sen+ gao] #list of ships not used in training and val

#create a csv for pytorch dataloaders function
dirpath="app/ships/"
elements = ['{0}txt'.format(element[:-3]) for element in sorted(names)]

def parse_one_annot(idfile):
   data = pd.read_csv(os.path.join(dirpath,elements[idfile]),names=["0","1","2","3","4"],sep=" ",index_col=0)
   boxes_array = np.array(data[["1", "2","3", "4"]].values)
   
   return boxes_array

class SARDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(names)
        self.mode= mode
    def __getitem__(self, idx):
        # load images and bounding boxes

        img_path = os.path.join(dirpath, self.imgs[idx])
        boxes = []
        area=boxes
        #should this be RGB? if not then change backbone
        if self.mode == "test":
            im = io.imread(img_path)
            im = ((im - im.min()) * (1/(im.max() - im.min()) * 255)).astype('uint8')
            img = Image.fromarray(im).convert("RGB")
            num_objs=0
            toggle="vv" #make sure to change so that your test images are properly read in
            
        else:
            img = Image.open(img_path).convert('RGB')
            #toggle = self.imgs[idx].split("_")[2] #comment out if testing on OPENSARSHIP test set
            toggle="vv"
            box_list = parse_one_annot(idx)
            num_objs = len(box_list)
            
            for i in range(num_objs):
                #math is due to inherent data architecture
                xmin = (box_list[i][0]*256)-(box_list[i][2]*128)
                xmax = (box_list[i][0]*256)+(box_list[i][2]*128)
                ymin = (box_list[i][1]*256)-(box_list[i][3]*128)
                ymax = (box_list[i][1]*256)+(box_list[i][3]*128)
                boxes.append([xmin, ymin, xmax, ymax])
                
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
    
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target#, toggle #comment out if testing on old models
    def __len__(self):
        return len(self.imgs)


#going to need to rewrite this
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

weight="/home/yabubaker/Documents/fourthproject/app/weight_vv_pre"
idx=0



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
#change to test if you arent using test set from OPENSAR
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# move model to the right device
model.to(device)
#model.load_state_dict(torch.load(weight,map_location=torch.device('cpu') ))
#Draw one of the predictions
#put the model in evaluation mode
model.eval()
def accept_input(idx):
    
    dataset_test = SARDataset(dirpath, "train", get_transform(train=False))
    img, _ = dataset_test[idx]
    return img

def get_prediction(img):
    with torch.no_grad():
       prediction = model([img])
    prediction = ' '.join([str(elem) for elem in prediction])
    return prediction

def get_image(img):
    with torch.no_grad():
       prediction = model([img])
    image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
    draw = ImageDraw.Draw(image)
    # draw groundtruth
    trueboxes=[]
    predboxes=[]
    
    for element in range(len(prediction[0]["boxes"])):    
       boxes = prediction[0]["boxes"][element].cpu().numpy()
       score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                        decimals= 4)
       if score > 0:

          draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], 
          outline ="red", width =3)
          draw.text((boxes[0], boxes[1]), text = str(score))
          predboxes.append(boxes)
    return image
