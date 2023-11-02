import pandas as pd
import numpy as np
import cv2
train = pd.read_csv('train.csv')

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape).T

# Test RLE functions
#assert mask2rle(rle2mask(train['EncodedPixels'].iloc[0]))==train['EncodedPixels'].iloc[0]
#assert mask2rle(rle2mask('1 1'))=='1 1'
for i in range(train.shape[0]):
    img = rle2mask(train['EncodedPixels'].iloc[i])
    print('./train_label/'+str(train['ImageId'].iloc[i]))
    cv2.imwrite('./train_label/'+str(train['ImageId'].iloc[i][:-3]+'png'),img)
    