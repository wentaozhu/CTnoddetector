from preprocessing import full_prep_tianchitest
from config_tianchi import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas

datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    testsplit = full_prep_tianchitest(datapath,prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
else:
    testsplit = os.listdir(datapath)

nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)

bbox_result_path = config_submit['bbox_result_path'] 
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers=15,pin_memory=False,collate_fn =collate)

    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])