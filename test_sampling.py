import torch
import torch.nn as nn 
import config 
import network 
import numpy as np 
from TorchSUL import Model as M
import loss 
import cv2 
import visutil
import pickle 

def pre_process(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = np.float32(img)
	img = img / 255
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])
	img = np.transpose(img, [2, 0, 1])
	return img

def process_img(img):
	h,w = img.shape[0], img.shape[1]
	hw = max(h, w)
	canvas = np.zeros([hw, hw, 3], dtype=np.uint8)
	padh, padw = (hw-h)//2, (hw-w)//2
	canvas[padh:padh+h, padw:padw+w] = img 
	canvas = cv2.resize(img, (512, 512))
	return canvas

# initialize the network 
model_dnet = network.MultiScaleNet()
x = np.float32(np.random.random(size=[1,3,512,512]))
x = torch.from_numpy(x)
with torch.no_grad():
	model_dnet(x)
model = loss.ModelWithLoss(model_dnet)
saver = M.Saver(model)
saver.restore('./model/')
model.eval()

img = cv2.imread('000000410650.jpg')
img = process_img(img)
img = pre_process(img)
img = torch.from_numpy(img[None, ...])

sampling = network.SamplingLayer()

fmaps, hmaps = model.run(img)
for i in range(3):
	for j in range(3):
		# print(hmaps[i][j].shape)
		visutil.vis_batch(img, hmaps[i][j], './outputs/hmap_%d_%d.png'%(i,j))
		pickle.dump(hmaps[i][j].cpu().detach().numpy(), open('outputs/hmap_%d_%d.pkl'%(i,j), 'wb'))
		pickle.dump(fmaps[i][j].cpu().detach().numpy(), open('outputs/fmap_%d_%d.pkl'%(i,j), 'wb'))

all_fhmap, _, _ = sampling(hmaps[0], fmaps[0], img)
print(all_fhmap.shape)
all_fhmap, _, _ = sampling(hmaps[1], fmaps[1], img)
print(all_fhmap.shape)
all_fhmap, _, _ = sampling(hmaps[2], fmaps[2], img)
print(all_fhmap.shape)

# for f,h in zip(fmaps, hmaps):
# 	print(f[0].shape, f[1].shape, f[2].shape, h[0].shape, h[1].shape, h[2].shape)
