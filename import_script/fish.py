import os
import numpy as np
import torch
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from dataloaders.utils import decode_segmap

I = np.load('meanimage.npy')

numClass={'pascal':4,
'coco':21,
'cityscapes':19}
classes = ['head', 'body', 'tail', 'z_cut']
for cid, c in enumerate(classes):
    if not os.path.isdir('result/'+c):
        os.mkdir('result/' + c)
classes.pop(-1)

cuda = torch.cuda.is_available()
cuda = False
nclass = numClass['pascal']
model = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict=torch.load(r'model\model_best.pth.tar', map_location='cpu')
if cuda:
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    
    model.module.load_state_dict(weight_dict['state_dict'])
else:
    model.load_state_dict(weight_dict['state_dict'])
model.eval()


# filenames = glob.glob(r'data\1280px-Rachycentron_canadum.jpg')
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((7,7),np.uint8)

prediction = []

class fish