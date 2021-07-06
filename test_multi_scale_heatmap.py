import torch
import config 
import network 
import numpy as np 
from TorchSUL import Model as M
import loss 
import cv2 
import pickle 
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from tqdm import tqdm 

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
	return canvas, [padw, padh, hw]

def deprocess_pts(meta, pts, scale):
	pts = pts.cpu().detach().numpy().astype(np.float32)
	padw, padh, hw = meta
	pts = pts * 512 / scale
	pts = pts[:, :, :, 0] * hw / 512 
	pts[:,:,:,0] -= padw
	pts[:,:,:,1] -= padh
	return pts 

def process_img2(img):
	h,w = img.shape[0], img.shape[1]
	if h<w:
		h2 = 512
		w2 = int(np.ceil(w * h2 / h / 32) * 32)
	else:
		w2 = 512
		h2 = int(np.ceil(h * w2 / w / 32) * 32)
	img = cv2.resize(img, (w2, h2))
	return img, [w2, h2, w, h]

def deprocess_pts2(meta, pts, scale):
	pts = pts.cpu().detach().numpy().astype(np.float32)
	w2, h2, w, h = meta
	pts = pts * w2 / scale
	pts[:,:,:,0] *= w / w2 
	pts[:,:,:,1] *= h / h2
	return pts 

def draw_pts(img, pts, idx):
	for i in range(config.num_pts):
		img2 = img.copy()
		for j in range(20):
			x = pts[i,j,0]
			y = pts[i,j,1]
			img2 = cv2.circle(img2, (int(x), int(y)), 3, (0,0,255), -1)
		cv2.imwrite('outputs/%d_%d.png'%(idx, i), img2)

def get_pts(img, meta):
	fmaps, hmaps = model.run_input(img)
	centers, scales, conf = sampling.get_joint_candidates(hmaps)
	# img = cv2.imread('000000410650.jpg')
	pts_all = []
	for i in range(len(centers)):
		pts = deprocess_pts2(meta, centers[i], scales[i])[0]
		# draw_pts(img, pts, i)
		pts_all.append(pts)
	pts_all = np.concatenate(pts_all, axis=1)
	pts_all = np.transpose(pts_all, [1,0,2])
	return pts_all

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
model.cuda()
sampling = network.SamplingLayer()

# get coco 
coco = COCO('person_keypoints_val2017.json')
ids = list(coco.imgs.keys())

with torch.no_grad():
	results = {}
	for i in tqdm(ids):
		fname = './val2017/%012d.jpg'%i
		img = cv2.imread(fname)
		img, meta = process_img2(img)
		img = pre_process(img)
		img = torch.from_numpy(img[None, ...])
		img = img.cuda()

		pts = get_pts(img, meta)
		results[i] = [pts,]
	pickle.dump(results, open('coco_results.pkl', 'wb'))

