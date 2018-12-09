from __future__ import print_function
import torch
from PIL import Image, ImageOps
import numpy as np
import skimage
import skimage.io
import cv2
import sys
sys.path.insert(0, '/playpen/PSMNet/dataloader')
import readpfm
import os
import os.path
#import matplotlib.pyplot as plt

predict_im = "/playpen/PSMNet/KITTI2015_test_pretrained_2015/000000_10.png"
groundtruth_im = "/playpen/KITTI2015/training/disp_occ_0/000000_10.png"

def normalizeDisparity(disp):
	(height, width) = disp.shape
	ndisp = np.asarray(disp)
	maxv = np.amax(ndisp)
	minv = np.amin(ndisp)
	print(maxv)
	print(minv)
	scale = float(256/float(maxv))
	for r in range(height):
		for c in range(width):
			disp[r,c] = scale * disp[r,c]
	return disp


def compareTwoImage(disp_p, disp_g):
	disp_p = np.ascontiguousarray(disp_p, dtype = np.float32)/256
	disp_g = np.ascontiguousarray(disp_g, dtype = np.float32)/256
	(height, width) = disp_g.shape
	black_p = 0
	for r in range(height):
		for c in range(width):
			if disp_g[r,c] == 0:
				disp_g[r,c] = disp_p[r,c]
				black_p = black_p + 1
	tpe = np.sum(np.absolute(np.subtract(disp_p, disp_g)) > 3.0)
	print("number of interested disparity point is %d"%(disp_p.size-black_p))
	print("find 3px_err %d, percentage is %0.4f"%(tpe, tpe/(disp_p.size+0.0-black_p)))
	# print("predicted disparity: "),
	# print(disp_p.shape)
	# print("groundtruth disparity:"),
	# print(disp_g.shape)
	return (tpe, disp_p.size-black_p)

def compareTwoFolders(fdisp_p, fdisp_g):
	imgns = [img for img in os.listdir(fdisp_p) if img.find('_10')>-1]
	tpe_total = 0
	disppt_total = 0
	tpe_ratio = 0.0
	index = 0
	for imgn in imgns:
		index += 1
		print("%d th image read"%index)
		disp_p = Image.open(fdisp_p+imgn)
		disp_g = Image.open(fdisp_g+imgn)
		(tpe, disppt) = compareTwoImage(disp_p,disp_g)
		tpe_total = tpe_total + tpe
		disppt_total = disppt_total + disppt
		tpe_ratio += (tpe/(disppt+0.0))
	print("three pixel error in total is %d\ndiparity points in total is %d\n the percentage is %0.4f"%(tpe_total,disppt_total,tpe_ratio/float(index)))


# (data,scale) = readpfm.readPFM("/playpen/middlebury_stereo/testing/groundtruth/disp0.pfm")
# print(data.shape)
# print(data.dtype)
# print(scale)
# data = np.multiply(data,scale)
# print(np.amax(data))
# disp_mat = np.empty([data.shape[0],data.shape[1]],dtype = np.int8)
# (height, width) = disp_mat.shape
# for r in range(height):
# 		for c in range(width):
# 			if(data[r,c] == float("inf")):
# 				disp_mat[r,c] = 0
# 			else:
# 				disp_mat[r,c] = int(data[r,c])
# cv2.imshow("image window",disp_mat)
# cv2.waitKey()
# im_test = cv2.imread("/playpen/middlebury_stereo/testing/groundtruth/disp1.png",cv2.IMREAD_GRAYSCALE)
# print(im_test.shape)

im_gray = cv2.imread(predict_im, cv2.IMREAD_GRAYSCALE)
# print(im_gray.dtype)
# print(im_gray.shape)
# cv2.imshow("image window",im_gray)
# cv2.waitKey()

test_mat = np.empty([im_gray.shape[0],im_gray.shape[1]],dtype = np.int8)
(height, width) = test_mat.shape
print(test_mat.shape)
# for r in range(height):
# 	im_gray[r,::] = int(((r+0.0)/(height+0.0))*256)
# cv2.imshow("image_test window",im_gray)
# cv2.waitKey()
fdisp_p = "/playpen/PSMNet/KITTI2012_test_pretrained_2015/"
fdisp_g = "/playpen/KITTI2012/training/disp_occ/"
compareTwoFolders(fdisp_p,fdisp_g)

im_gray = normalizeDisparity(im_gray)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_RAINBOW)
cv2.imshow("image window",im_color)
cv2.waitKey()
#read two image
disp_g = Image.open(groundtruth_im)
disp_p = Image.open(predict_im)
compareTwoImage(disp_p, disp_g)
