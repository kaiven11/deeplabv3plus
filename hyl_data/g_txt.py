import pandas as pd
import numpy as np
import os
train = os.listdir('/raid/hyl/CV/DeepLabV3Plus/hyl_data/images')
train_list = [x[:-4] for x in train]

txt_writer = open('/raid/hyl/CV/DeepLabV3Plus/hyl_data/train_aug.txt','w')
val_writer = open('/raid/hyl/CV/DeepLabV3Plus/hyl_data/val_aug.txt','w')


for index ,i in enumerate(train_list):
    print(index)
    if index <35: 
        txt_writer.write(i+'\n')
    else:
        val_writer.write(i+'\n')
txt_writer.close() 
val_writer.close()