import numpy as np
from numpy.core.numeric import indices

import torch
import torch.nn as nn
from utilty import *
import math
import scipy.sparse as sp

"""
*********************************************************
Special function for only sparse region backpropataion layer.
"""
class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

"""
*********************************************************
Layers
"""
class Discriminator(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super(Discriminator, self).__init__()
        self.f = nn.Bilinear(hid_dim, emb_dim, 1)
        self.act = nn.Sigmoid()


    def forward(self, X, Y):
        sc = self.f(X, Y)
        sc = self.act(sc)

        return sc

class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        self.weight_size_list = eval(args.layer_size)

        self.dropout = args.dropout
        self.drop = nn.Dropout(p=self.dropout)
        self.n_layers = len(self.weight_size_list)
        self.weight_size_list = [args.emb_dim] + self.weight_size_list

        self.MLP = nn.ModuleList(\
            [nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1]) for k in range(self.n_layers)])
        self.act = nn.ReLU()

    def forward(self, A, X):
        I = sp.eye(A.shape[0])
        A_hat = A + I
        rowsum = np.array(A_hat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = A_hat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        A_hat = sparse_to_tensor(A_hat, X.device)

        inputX = X
        for i in range(self.n_layers):
            Xw = self.drop(self.MLP[i](inputX))
            outputX = self.act(torch.matmul(A_hat, Xw))
            inputX = outputX

        return outputX

class MsMPN(nn.Module):
    def __init__(self, args):
        super(MsMPN, self).__init__()

        self.weight_size_list = eval(args.layer_size)
        self.n_layers = len(self.weight_size_list)
        self.weight_size_list = [args.emb_dim] + self.weight_size_list
        self.dropout = args.dropout
        self.type = args.type

        self.MLP1s = nn.ModuleList(\
            [nn.Linear(2*self.weight_size_list[k], self.weight_size_list[k+1]) for k in range(self.n_layers)])
        self.MLP2s = nn.ModuleList(\
            [nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1]) for k in range(self.n_layers)])
        self.MLP3s = nn.ModuleList(\
            [nn.Linear(self.weight_size_list[k], self.weight_size_list[k+1]) for k in range(self.n_layers)])
        self.drop = nn.Dropout(p=self.dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.spmm = SpecialSpmm()

        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, A, sub, X):
        pre_embeddings = X
        final_embeddings = [2*pre_embeddings]
        for k in range(self.n_layers):
            all_agg_embeddings = torch.matmul(A, pre_embeddings)
            sub_agg_embeddings = self.spmm(sub[0], sub[1], torch.Size(A.shape), pre_embeddings)

            if self.type == 3:
                all_embeddings = pre_embeddings + all_agg_embeddings
                sub_embeddings = pre_embeddings * sub_agg_embeddings
                embeddings = self.sigmoid(self.alpha * self.MLP2s[k](all_embeddings)) + self.relu(self.beta * self.MLP3s[k](sub_embeddings))
            elif self.type == 2:
                embeddings = self.sigmoid(self.MLP2s[k](pre_embeddings + all_agg_embeddings + sub_agg_embeddings))
            elif self.type == 1:
                embeddings = self.sigmoid(self.MLP2s[k](pre_embeddings + sub_agg_embeddings))
            else:
                embeddings = self.sigmoid(self.MLP2s[k](pre_embeddings + all_agg_embeddings))

            pre_embeddings = self.drop(embeddings)
            norm_embeddings = nn.functional.normalize(embeddings, dim=1)
            final_embeddings += [norm_embeddings]

        final_embeddings = torch.cat(final_embeddings, 1)
        return final_embeddings

"""
*********************************************************
Model
"""
class Model(nn.Module):
    def __init__(self, data_config, args, init_embedding):
        super(Model, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.entity_embedding = nn.Embedding(self.n_users + self.n_items, args.emb_dim)
        self.entity_embedding.weight = nn.Parameter(init_embedding['entity_embedding'])

        self.k = args.k
        self.batch_size = args.mi_batch_size

        self.disc = Discriminator(args.emb_dim, args.hid_dim)
        self.msmpn = MsMPN(args)
        # self.gcn = GCN(args)
        self.fc = nn.Linear(args.emb_dim, args.hid_dim)
        self.drop = nn.Dropout(0.3)
        self.act = nn.PReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def Encode(self, id):
        return self.entity_embedding(id)

    def sort_MI(self, pos, A):
        indptr = A.tocsr().indptr
        A = sparse_to_tensor(A, self.device)
        value, indices = [], []
        value1, indices1 = [], []
        sc = torch.squeeze(torch.cat(pos, 0), -1)
        for i in range(0,len(indptr)-1):
            # ind = torch.randperm(indptr[i+1]-indptr[i])[:math.ceil(self.k*(indptr[i+1]-indptr[i]))].to(self.device)
            # val = sc[indptr[i]:indptr[i+1]][ind]
            val, ind = torch.topk(sc[indptr[i]:indptr[i+1]], math.ceil(self.k*(indptr[i+1]-indptr[i])))
            val1, ind1 = torch.topk(sc[indptr[i]:indptr[i+1]], math.ceil(1*(indptr[i+1]-indptr[i])))
            if len(val) == 0:
                continue
            val = torch.exp(val) / torch.sum(torch.exp(val))
            value.append(val)
            indices.append(A._indices()[:, indptr[i]:indptr[i+1]][:,ind])
            value1.append(val1)
            indices1.append(A._indices()[:, indptr[i]:indptr[i+1]][:,ind1])

        value = torch.cat(value, 0)
        indices = torch.cat(indices, 1)
        value1 = torch.cat(value1, 0)
        indices1 = torch.cat(indices1, 1)
        return indices, value, indices1, value1

    def _agg_sc(self, idx, pos, neg, A):
        indptr = get_csr_indptr(A)
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        if end > self.n_items + self.n_users:
            end = self.n_items + self.n_users

        new_pos, new_neg = [], []
        for i in range(start, end):
            if indptr[i+1] == indptr[i]:
                continue
            new_pos.append(torch.mean(pos[indptr[i]-indptr[start]:indptr[i+1]-indptr[start]]))
            new_neg.append(torch.mean(neg[indptr[i]-indptr[start]:indptr[i+1]-indptr[start]], 0))
    
        new_pos = torch.stack(new_pos, 0)
        new_neg = torch.squeeze(torch.stack(new_neg, 0), -1)
        return new_pos, new_neg

    def MI(self, A, X, src, pos_t, neg_t):
        X_1 = self.fc(X)
        X_2 = self.act(torch.matmul(A, X_1) + X_1)

        source = X_2.index_select(0, src)
        pos_target = X.index_select(0, pos_t)
        pos = self.act(self.disc(source, pos_target))

        neg_sc_list = []
        for i in range(len(neg_t)):
            neg_target = X.index_select(0, neg_t[i])
            neg_sc = self.disc(source, neg_target)
            neg_sc_list.append(neg_sc)
        neg = torch.stack(neg_sc_list, 1)
        neg = self.act(neg)

        return pos, neg

    def forward(self, A, A2, X, sub=None, batch=None):
        A = sparse_to_tensor(A, self.device)
        A2 = sparse_to_tensor(A2, self.device)
        X_1 = self.fc(X)
        X_2 = nn.functional.normalize(torch.matmul(A, X) + X, dim=1)

        if batch != None:
            batch_idx, src, pos_t, neg_t = batch[0], batch[1], batch[2], batch[3]
            src = src.to(self.device)
            pos_t = pos_t.to(self.device)
            neg_t = neg_t.to(self.device)
            
            # pos, neg = self.MI(A, X, src, pos_t, neg_t)

            source = X_2.index_select(0, src)
            pos_target = X_1.index_select(0, pos_t)
            pos = self.act(self.disc(source, pos_target))

            neg_sc_list = []
            for i in range(len(neg_t)):
                neg_target = X_1.index_select(0, neg_t[i])
                neg_sc = self.disc(source, neg_target)
                neg_sc_list.append(neg_sc)
            neg = torch.stack(neg_sc_list, 1)
            neg = self.act(neg)

            node_pos, node_neg = self._agg_sc(batch_idx, pos, neg, A)
            return pos, node_pos, node_neg
        
        # A = A + A2
        embeddings = self.msmpn(A, sub, X_2)

        ua_embeddings, ia_embeddings = torch.split(embeddings,\
             [self.n_users, self.n_items], 0)
        return ua_embeddings, ia_embeddings

        
"""
*********************************************************
Loss Function
"""
def BPR_loss(data_generator, user_embed, item_embed):
    Base_loss, Reg_loss = 0., 0.
    n_batch = data_generator.n_train // cmd_args.batch_size + 1
    for _ in range(n_batch):
        users, pos_items, neg_items = data_generator.generate_train_batch()
        u_e, pos_i_e, neg_i_e = user_embed[users], item_embed[pos_items], item_embed[neg_items]

        reg_loss = cmd_args.reg * \
            (torch.norm(u_e, p=2) + torch.norm(pos_i_e, p=2) + torch.norm(neg_i_e, p=2)) / cmd_args.batch_size
        pos_scores = torch.sum(torch.mul(u_e, pos_i_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e, neg_i_e), axis=1)
        base_loss = torch.mean(torch.nn.functional.softplus(-(pos_scores - neg_scores)))

        Base_loss += base_loss   
        Reg_loss += reg_loss

    bpr_loss = Base_loss + Reg_loss
    return bpr_loss

def MI_loss(pos, neg, T, mode='kl'):
    if mode == 'js':
        e_pos = torch.log(1+torch.exp(-pos))
        e_neg = torch.mean(torch.log(1+torch.exp(neg)),1)
        return (e_pos+e_neg).mean()
    elif mode == 'infonce':
        e_pos = torch.exp(pos / T)
        e_neg = torch.sum(torch.exp(neg / T), 1)
        return -(torch.log(e_pos / e_neg)).mean()
    elif mode == 'kl':
        tmp = pos - torch.mean(torch.exp(neg), 1)
        return -tmp.mean()