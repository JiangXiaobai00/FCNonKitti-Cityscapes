import cv2
import numpy as np
import copy
import PIL.Image as Image
disp_path='/home/zhanghf/Desktop/aachen_000000_000019_disparity.png'
disp_path1='/home/zhanghf/Desktop/2.png'
path='/home/zhanghf/Downloads/dataset/data_semantics/training/disp_occ_0/000000_10.png'
"""
disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # disparity map: [1024, 2056]
disp[disp > 0] = (disp[disp > 0] - 1) / 256
depth = copy.copy(disp)
depth[depth > 0] = (0.209313 * 2262.52) / depth[depth > 0]
depth[depth >= 85] = 0
depth = depth.astype(np.float32)
"""
dataL=Image.open(path)
dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

dataL1=Image.open(disp_path)
dataL1 = np.ascontiguousarray(dataL1,dtype=np.float32)/256
dataL2=Image.open(disp_path1)
dataL2 = np.ascontiguousarray(dataL2,dtype=np.float32)/2
mask = (disp_true > 0)

print ("0")