from __future__ import print_function
import logging
import numpy as np
import argparse
import scipy.sparse as sp
from gpu_manager import GPUManager
import time
from functools import reduce
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
#data
cmd_opt.add_argument('-dataset', type=str, default='OMIM', help='data folder name')
cmd_opt.add_argument('-pretrain', default=None, help='Pretrain data path')
cmd_opt.add_argument('-weight', type=str, default=None, help='saved model parameters')

#model param 
cmd_opt.add_argument('-emb_dim', type=int, default=64, help='Node embedding size')
cmd_opt.add_argument('-hid_dim', type=int, default=200, help='Node embedding size')
cmd_opt.add_argument('-layer_size', nargs='?', default='[64]', help='Output sizes of every layer')
cmd_opt.add_argument('-batch_size', type=int, default=128, help='Minibatch size')
cmd_opt.add_argument('-mi_batch_size', type=int, default=1000, help='Minibatch size')
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='Number of epoch')
# cmd_opt.add_argument('-inner_loop', type=int, default=100, help='Number of epoch')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='Init learning_rate')
cmd_opt.add_argument('-dropout', type=float, default=0.1, help='The dropout rate')
cmd_opt.add_argument('-reg', type=float, default=0.001, help='Regularization for user and item embeddings')
cmd_opt.add_argument('-gamma', type=float, default=0.1, help='')
cmd_opt.add_argument('-k', type=float, default=0.5, help='')
cmd_opt.add_argument('-T', type=float, default=0.01, help='')
cmd_opt.add_argument('-neg_num', type=int, default=10, help='')
cmd_opt.add_argument('-alpha', type=float, default=1, help='')
cmd_opt.add_argument('-beta', type=float, default=1, help='')
cmd_opt.add_argument('-mi_kind', type=str, default='infonce', help='')
cmd_opt.add_argument('-type', type=int, default=3, help='')

#other 
cmd_opt.add_argument('-gpu_id', type=int, default=-1, help='Which GPU to run')
cmd_opt.add_argument('-seed', type=int, default=0, help='Which GPU to run')
cmd_opt.add_argument('-show_model_param', type=bool, default=False, help='Whether to show model param')
cmd_opt.add_argument('-show_step', type=int, default=1, help='steps to show result')
cmd_opt.add_argument('-f', type=int, default=1, help='dataset')
cmd_opt.add_argument('-Ks', nargs='?', default='[1,5,10,15]', help='Output sizes of every layer')


cmd_args, _ = cmd_opt.parse_known_args()
cmd_args.Ks = eval(cmd_args.Ks)

gm = GPUManager()

logger = logging.getLogger()
logger.setLevel(logging.INFO)  
# logfile = './debug.log'
filename = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '.log'
logfile = './log/' + filename
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if cmd_args.show_model_param:
                if param.dim() > 1:
                    print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                else:
                    print(name, ':', num_param)
            total_param += num_param
    return total_param

def load_pretrain(path):
    pretrain_path = '%s/mf.npz' %(path)
    pretrain_data = np.load(pretrain_path)
    return pretrain_data

def generate_result(L):
    ret = {'precision': np.zeros(L), \
            'recall': np.zeros(L), \
            'ndcg': np.zeros(L), \
            'hit_ratio': np.zeros(L),\
            'auc': 0., \
            'predict':dict()}
    return ret

def sparse_to_tensor(X, device=None):
    X = X.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((X.row, X.col)).astype(np.int64))
    values = torch.from_numpy(X.data)
    shape = torch.Size(X.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

def get_csr_indptr(A):
    shape = A.shape
    row = A._indices()[0].detach().cpu().numpy()
    col = A._indices()[1].detach().cpu().numpy()
    coo = sp.coo_matrix(([1.]*len(row), (row, col)), shape)
    csr = coo.tocsr()
    return csr.indptr

