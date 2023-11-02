
import json
from labelme.utils.shape import labelme_shapes_to_label
import numpy as np
import cv2
import os
"""
根据labelme的json文件生成对应文件的mask图像
"""
def test():
    image_origin_path = r"C:\Users\martin\img\xx.jpg"
    image = cv2.imread(image_origin_path)
    # if len(image.size) == 2:
    #     shape= image.shape
    # if len(image.size) == 3:
    #     shape = image.size
    # print(w,h)

    json_path = r"C:\Users\martin\img\xx.json"
    data = json.load(open(json_path))

    lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
    print(lbl_names)
    mask=[]
    class_id=[]
    for i in range(1,len(lbl_names)): # 跳过第一个class（因为0默认为背景,跳过不取！）
        key = [k for k, v in lbl_names.items() if v == i][0]
        print(key)
        mask.append((lbl==i).astype(np.uint8)) # 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）
        class_id.append(i) # mask与class_id 对应记录保存
    print(class_id)
    # print(mask)
    # print(class_id)
    mask=np.asarray(mask,np.uint8)
    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0])
    # retval, im_at_fixed = cv2.threshold(mask[:,:,0], 0, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("mask_1111_real.png", im_at_fixed)
    # print(mask.shape)
    # for i in range(0,len(class_id)):
    #     retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY)
    #     cv2.imwrite("mask_out_{}.png".format(i), im_at_fixed)


def get_finished_json(root_dir):
    import glob
    json_filter_path = root_dir + "\*.json"
    jsons_files = glob.glob(json_filter_path)
    return jsons_files


def get_dict(json_list):
    dict_all = {}
    for json_path in json_list:
        dir,file = os.path.split(json_path)
        file_name = file.split('.')[0]
        image_path = os.path.join(dir,file_name+'.jpg')
        dict_all[image_path] = json_path
    return dict_all


def process(dict_):
    for image_path in dict_:
        
        class_id = []
        key_ = []
        image = cv2.imread(image_path)
        json_path = dict_[image_path]
        data = json.load(open(json_path))
        lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
        # print(np.sum(lbl==2),'ssss')
        # print(lbl.shape)
        # mask = np.zeros(image.shape[:2])
        
        
        # for i in range(1, len(lbl_names)):  # 跳过第一个class（因为0默认为背景,跳过不取！）
        #     key = [k for k, v in lbl_names.items() if v == i][0]
        #     # print(i,'current i')
        #     tmp = (lbl == i).astype(np.uint8) #
            
        #     mask+=tmp  # 举例：当解析出像素值为1，此时对应第一个mask 为0、1组成的（0为背景，1为对象）


        #     class_id.append(i)  # mask与class_id 对应记录保存
        #     key_.append(key)
        

        # mask1 = np.asarray(lbl, np.uint8)
        # # mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])
        # image_name = os.path.basename(image_path).split('.')[0]
        # dir_ = os.path.dirname(image_path)

              
        # image_name_ = "{}.png".format(image_name)
        # dir_path =  os.path.join(dir_, 'mask') # 构建保存缺陷的文件夹 key_[i]为缺陷名称，i为缺陷ID
        # checkpath(dir_path)
        # image_path_ = os.path.join(dir_path,image_name_)
        
        # print(np.sum(mask1==3),'mask1')

        # # retval, im_at_fixed = cv2.threshold(mask1, 0, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(image_path_, mask1)


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    root_dir = r'C:\Users\asus\Desktop\DATA\labelme_json\json'
    json_file = get_finished_json(root_dir)
    image_json = get_dict(json_file)
    process(image_json)
    # test()
