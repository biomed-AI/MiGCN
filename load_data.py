import collections
import numpy as np
import random as rd
import scipy.sparse as sp
from torch import dtype
from utilty import *

rd.seed(2021)
np.random.seed(2021)

class DataLoading(object):
    def __init__(self, args):
        
        self.train_path = args.dataset + "/train.txt"
        self.test_path = args.dataset + "/test.txt"
        # self.item_graph_path = args.dataset + "/item_graph.txt"
        # self.user_graph_path = args.dataset + "/user_graph.txt"

        if cmd_args.f == 0:
            self.item_graph_path = "./kg/sim/item_graph.txt"
            self.user_graph_path = "./kg/sim/user_graph.txt"
        else:
            self.item_graph_path = "./kg/test/item_graph.txt"
            self.user_graph_path = "./kg/test/user_graph.txt"


        self.batch_size = args.batch_size

        # Loading data
        self.train_data, self.train_user_dict = self.load_graph(self.train_path)
        self.test_data, self.test_user_dict = self.load_graph(self.test_path)

        self.item_graph, _ = self.load_graph(self.item_graph_path)
        self.user_graph, _ = self.load_graph(self.user_graph_path)

        self.statistic()

        # Processing data
        self.adj, self.A = self._get_adj()
        self.all_data_dict = self._get_all_data()

        self.print_data_info()



    def load_graph(self, path):
        user_dict = dict()
        inter_mat = list()

        lines = open(path, 'r').readlines()
        for l in lines:
            inters = [int(i) for i in l.strip().split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict

    def statistic(self):
        # self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        # self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0]), max(self.user_graph[:, 0]), max(self.user_graph[:, 1])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1]), max(self.item_graph[:, 0]), max(self.item_graph[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    def print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_edges]=[%d]' % (len(self.adj.tocoo().row)))

    def _get_relational_lap(self, A):
        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocsr()

        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocsr()

        lap_list = [_si_norm_lap(A_) for A_ in A]
        return sum(lap_list)

    def _get_adj(self):
        def _np_mat2sp_adj(np_mat, row_pre=0, col_pre=0):
            n_all = self.n_items + self.n_users
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        A = R + R_inv
        I, _ = _np_mat2sp_adj(self.item_graph, row_pre=self.n_users, col_pre=self.n_users)
        U, _ = _np_mat2sp_adj(self.user_graph)

        print(len(I.tocoo().col))
        print(len(U.tocoo().col))

        adj = I + U
        return adj, A

    def _get_all_data(self):
        data_dict = collections.defaultdict(list)
        lap = self.adj.tocoo()

        rows = lap.row
        cols = lap.col

        for i_id in range(len(rows)):
            head = rows[i_id]
            tail = cols[i_id]
            data_dict[head].append(tail)

        return data_dict

    def get_config(self):
        config = dict()
        config['n_users'] = self.n_users
        config['n_items'] = self.n_items
        return config

    def generate_train_batch(self):
        exist_users = self.train_user_dict.keys()
        if self.batch_size <= self.n_users:
            users = rd.sample(exist_users, self.batch_size)
        else:
            users = [rd.choice(exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def generate_neg_list(self, neg_num):
        neg_n_list = []
        A = self.adj.tocoo()
        n_all = self.n_users + self.n_items
        source = A.row
        for j in range(neg_num):
            neg_list = []
            i = 0
            while True:
                if i == len(A.row): break
                neg = np.random.randint(low=0, high=n_all,size=1)[0]

                if neg not in self.all_data_dict[source[i]]:
                    neg_list.append(neg)
                    i += 1
            neg_list = torch.LongTensor(neg_list)
            neg_n_list.append(neg_list)

        neg_n_list = torch.stack(neg_n_list, 0)
        return neg_n_list

    def generate_batch(self, batch, batch_size, neg_num):
        def sample_neg(source):
            neg_list = []
            i = 0
            while True:
                if i == len(source): break
                neg = np.random.randint(low=0, high=self.adj.shape[0],size=1)[0]
                if neg not in self.all_data_dict[source[i]]:
                    neg_list.append(neg)
                    i += 1
            return neg_list
        start = batch * batch_size
        end = (batch + 1) * batch_size
        if end > self.adj.shape[0]:
            end = self.adj.shape[0]
        indptr = self.adj.tocsr().indptr

        source = self.adj.tocoo().row[indptr[start]:indptr[end]]
        pos_target = self.adj.tocoo().col[indptr[start]:indptr[end]]
        neg_target = []
        for j in range(neg_num):
            neg_target.append(sample_neg(source))

        source = torch.LongTensor(source)
        pos_target = torch.LongTensor(pos_target)
        neg_target = torch.LongTensor(neg_target)
        return source, pos_target, neg_target
            

