import cv2
import numpy as np
import scipy.fftpack
from helper_geometry import order_points, line_length

def four_point_transform(image, quad):
    rect = order_points(quad)
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates

    width_a = line_length(rect[:2])
    width_b = line_length(rect[2:])
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = line_length(rect[1:3])
    height_b = line_length(rect[[0,-1]])
    max_height = max(int(height_a), int(height_b))

    
    src = np.float32(rect)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))
    
    return warped

def apply_offset(image, offset):
    return cv2.add(image, np.full(image.shape, offset, dtype=np.int8), dtype=cv2.CV_8U)
    
    
def apply_trim(image, amt):
    h,w,_ = image.shape
    new_h = int(h*amt)
    new_w = int(w*amt)
    new_h_start = int(h*(1-amt)/2)
    new_w_start = int(w*(1-amt)/2)
    return image[new_h_start:new_h_start+new_h, new_w_start:new_w_start+new_w]
    
def add_text(im, text, pos):
    fontScale = 0.5
    fontColor = (255,255,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (size, baseline) = cv2.getTextSize(text, font, fontScale, thickness)
    origin = ((int) (pos[0]-size[0]/2), (int) (pos[1]+size[1]/2))
    cv2.putText(im, text, origin, font, fontScale,fontColor, thickness=thickness)
    
def add_text_bg(im, text, pos):
    fontScale = 0.5
    fontColor = (255,255,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (size, baseline) = cv2.getTextSize(text, font, fontScale, thickness)
    origin = ((int) (pos[0]-size[0]/2), (int) (pos[1]+size[1]/2))
    top_left = ((int) (pos[0]-size[0]/2), (int) (pos[1]-size[1]/2))
    bottom_right = ((int) (pos[0]+size[0]/2), (int) (pos[1]+size[1]/2))
    cv2.rectangle(im, top_left, bottom_right, (0,0,0), -1)
    cv2.putText(im, text, origin, font, fontScale,fontColor, thickness=thickness)

def phash(im):
    imadj = cv2.resize(im, (32*4, 32*4), cv2.INTER_AREA)
    res=scipy.fftpack.dct(scipy.fftpack.dct(imadj, axis=0), axis=1)
    lowfreq = res[:32, :32]
    median = np.median(lowfreq, axis=(0,1))
    result = np.array(list(map(lambda x: x.flatten(), np.split(lowfreq > median, 3, axis = 2)))).flatten()
    return result
    
def phash_dim(img):
    imadj = cv2.resize(img, (32*4, 32*4), cv2.INTER_AREA)
    res=scipy.fftpack.dct(scipy.fftpack.dct(imadj, axis=0), axis=1)
    lowfreq = res[:32, :32]
    median = np.median(lowfreq)
    return (lowfreq > median).flatten()

    
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def applyCLAHE(im):
    lab=cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe.apply(lab[...,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
