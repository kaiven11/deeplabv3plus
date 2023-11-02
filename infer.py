import torch
import cv2
from torch.nn import functional as F
import torchvision.transforms as transform
import numpy as np

 
 
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
tf = transform.Compose([transform.ToTensor(), 
                        transform.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
 
class Inference:
    def __init__(self, inp_model, threshold, re_size, classes):
        self.inp_model = inp_model
        self.inp_model.eval()
        self.size = re_size
        self.threshold = threshold
        self.classes = classes
        self.num_classes = len(classes)
 
 
    def __call__(self, img):
        masks_list = []
        img_size = img.shape[:2]
        img = cv2.resize(img, tuple(self.size))
        img = tf(img).unsqueeze(0).to(device)
 
        with torch.no_grad():
            out = self.inp_model(img)
        out = F.interpolate(out, size=img_size, mode="bicubic", align_corners=False) #上采样到图片大小
 
        w, h = out.shape[2:]
        back_matrix = torch.ones(size=(1, w, h)) * self.threshold
        back_matrix = back_matrix.to(device)
        pr = out[0]
        pr = torch.cat([pr, back_matrix], dim=0)
 
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
 
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], self.num_classes))
        for c in range(self.num_classes):
            seg_img[:, :, c] += ((pr[:, :] == c) * (255)).astype('uint8')
            num_sum = np.sum(seg_img[:, :, c])
            if num_sum > 0:
                masks_list.append([np.uint8(seg_img[:, :, c]), self.classes[c]])
        return masks_list
 
 
if __name__ == '__main__':
    from network import deeplabv3plus_mobilenet
    checkpoint = torch.load("/raid/hyl/CV/DeepLabV3Plus/checkpoints/voc.pth",map_location=torch.device('cpu'))
    model = deeplabv3plus_mobilenet(21,output_stride=16)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    
    img = cv2.imread("/raid/hyl/CV/DeepLabV3Plus/333.jpg")
    img_ori = img.copy()
    color_map = [[255,0,0],[0,0,255],[0,70,122]]
    classes = [str(x) for x in range(21)]
    model = Inference(inp_model=model, threshold=0.8, re_size=(513,513), classes=classes)
    mask_list = model(img)
    for i in range(len(mask_list)):
        mask = np.array(mask_list[i][0])
        conters,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in conters:
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(img,[cnt],0, (0, 255, 0), 4)
            else:
                box = cv2.boundingRect(cnt)
                x, y, w, h = box
                mask[y:y+h,x:x+w] = 0
        img[:, :, :][mask[:, :] > 0] = color_map[i]
    cv2.imwrite('seg.jpg',img)
    cv2.imwrite('ori.jpg',img_ori)
    # cv2.waitKey()