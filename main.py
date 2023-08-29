from operator import mod
import time

import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn
import torch.optim as optim
from metrics import *
from utilty import *
from load_data import *
from model import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    if cmd_args.seed != -1:
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)
        # np.random.seed(cmd_args.seed)

    print(cmd_args)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    print("loading data...")
    data_generator = DataLoading(args=cmd_args)

    if cmd_args.gpu_id >= 0:
        torch.cuda.set_device(cmd_args.gpu_id)
    else:
        torch.cuda.set_device(gm.auto_choice())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cmd_args.pretrain is not None :
        print("loading pretrained data...")
        pretrain_data = load_pretrain(cmd_args.pretrain)
        user_embed = torch.from_numpy(pretrain_data['user_embed']).float().to(device)
        item_embed = torch.from_numpy(pretrain_data['item_embed']).float().to(device)
    else:
        user_embed = torch.ones([data_generator.n_users, cmd_args.emb_dim]).to(device)
        item_embed = torch.ones([data_generator.n_items, cmd_args.emb_dim]).to(device)
        torch.nn.init.xavier_uniform_(user_embed)
        torch.nn.init.xavier_uniform_(item_embed)
        print(user_embed[0][:5])


    init_embedding = dict()
    entity_embed = torch.cat((user_embed, item_embed), 0)
    init_embedding['entity_embedding'] = entity_embed

    model = Model(data_config=data_generator.get_config(), args=cmd_args, init_embedding=init_embedding).to(device)
    # torch.save(model.state_dict(), './test_model')
    # exit()
    # model.load_state_dict(torch.load('./test_model'))

    print("Loading Model..Number of Model Parameters: ", count_parameters(model))

    if cmd_args.weight is not None :
        #Testing
        #to do
        raise ValueError('Stop Testing')
    
    optimizer = optim.Adam(model.parameters(),
                           lr=cmd_args.learning_rate, 
                           amsgrad=True,
                           weight_decay=cmd_args.reg)

    stopping_step, runing_step = 30, 0
    best_rec = 0
    all_entities_id = torch.LongTensor(np.arange(data_generator.n_items + data_generator.n_users)).to(device)

    for epoch in range(cmd_args.num_epochs):
        logger.info("epoch: {}".format(epoch))
        """
        *********************************************************
        Train.
        """
        model.train()
        t1 = time.time()

        mi_loss, bpr_loss = 0, 0

        n_batch = data_generator.adj.shape[0] // cmd_args.mi_batch_size + 1
        optimizer.zero_grad()
        entity_embedding = model.Encode(all_entities_id)
        sub = None
        mi_sc = []
        for i in range(n_batch):
            source, pos_target, neg_target = data_generator.generate_batch(i, cmd_args.mi_batch_size, cmd_args.neg_num)
            sc, pos, neg = model(data_generator.adj, data_generator.A, entity_embedding, batch=(i, source, pos_target, neg_target))
            batch_mi_loss = cmd_args.gamma * MI_loss(pos, neg, cmd_args.T, cmd_args.mi_kind)
            mi_loss += batch_mi_loss
            mi_sc.append(sc)
        sub = model.sort_MI(mi_sc, data_generator.adj)

        # ind, val = sub[0], sub[1]
        # for i in range(len(val)):
        #     if ind[1][i] > 4000 and ind[1][i] < 4010:
        #         logger.info("{} {} : {}".format(ind[0][i], ind[1][i], val[i]))

        user_embed, item_embed = model(data_generator.adj, data_generator.A, entity_embedding, sub=sub)
        bpr_loss = BPR_loss(data_generator, user_embed, item_embed)
        loss = mi_loss + bpr_loss
        loss.backward()

        optimizer.step()

        t2 = time.time()
        """
        *********************************************************
        Test.
        """
        model.eval()

        ret = generate_result(len(cmd_args.Ks))

        test_users = list(data_generator.test_user_dict.keys())
        n_test_users = len(test_users)
        n_batch = n_test_users // cmd_args.batch_size + 1

        count = 0
        for id in range(n_batch):
            start = id * cmd_args.batch_size
            end = (id + 1) * cmd_args.batch_size
            user_batch = test_users[start: end]
            item_batch = range(data_generator.n_items)

            u_e, i_e = user_embed[user_batch], item_embed[item_batch]
            batch_predictions = torch.matmul(u_e, i_e.transpose(0,1)).cpu().detach().numpy()
            batch_result = batch_metrics(batch_predictions, user_batch, data_generator)
    
            count += len(batch_result)

            for re in batch_result:
                ret['precision'] += re['precision']/n_test_users
                ret['recall'] += re['recall']/n_test_users
                ret['ndcg'] += re['ndcg']/n_test_users
                ret['hit_ratio'] += re['hit_ratio']/n_test_users
                ret['auc'] += re['auc']/n_test_users
                # ret['predict'][re['predict'][0]] = np.array(re['predict'])[1:]

        assert count == n_test_users

        t3 = time.time()
        show_step = cmd_args.show_step
        if (epoch + 1) % show_step == 0:
            np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
            print('Epoch %d [%.1fs + %.1fs]: train==[%.5f = %.5f + %.5f], recall=' % (epoch, t2 - t1, t3 - t2, bpr_loss+mi_loss, bpr_loss, mi_loss) \
                , ret['recall'], ', precision=', ret['precision'] , ', auc=[%.5f], sum_recall=[%.5f]' % (ret['auc'], sum(ret['recall'])) \
            )
        
        runing_step += 1
        
        if best_rec <= ret['auc']:
            best_rec = ret['auc']
            best_all_ret = ret
            runing_step = 0
        
        if runing_step >= stopping_step:
            print('End of trainning! Best recall=', best_all_ret['recall'], ', precision=', best_all_ret['precision'] , ', auc=[%.5f]' % (best_all_ret['auc']))
            break
