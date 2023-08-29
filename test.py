import matplotlib.pylab as plt

import matplotlib
from matplotlib.pyplot import MultipleLocator
# import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
matplotlib.rcParams['savefig.dpi'] = 500
# fig, axs = plt.subplots(1,1)
matplotlib.rcParams['figure.figsize'] = [10, 7]

font0 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 13,
}

d = pd.DataFrame()
d['variant'] = ['GCN']*15 + ['Sub-Neighbor']*15 + ['Attentive']*15 + ['Informative']*15 + ['GCN']*15 + ['Sub-Neighbor']*15 + ['Attentive']*15 + ['Informative']*15
d['p'] = ['Recall@1','Recall@15','Pre@1','Precision@15', 'AUC'] * (len(d['variant']) // 5)
d['dataset'] = ['MiGCN']*60 + ['MiGCN' + '$\mathregular{_{sim}}$']*60
d['ret'] = [0.0291,0.2158,0.0324,0.0162,0.824]\
         + [0.0291+0.0025,0.2158+0.0044,0.0324+0.003,0.0162+0.0008,0.824+0.004]\
         + [0.0291-0.0025,0.2158-0.0044,0.0324-0.003,0.0162-0.0008,0.824-0.004]\
         + [0.0296,0.2142,0.0327,0.0162,0.825]\
         + [0.0296+0.003,0.2142+0.005,0.0327,0.0162+0.0009,0.825+0.005]\
         + [0.0296-0.003,0.2142-0.005,0.0327,0.0162-0.0009,0.825-0.005]\
         + [0.0320,0.2178,0.0335,0.0165,0.827]\
         + [0.0320+0.002,0.2178+0.0036,0.0335,0.0165+0.0007,0.827+0.004]\
         + [0.0320-0.002,0.2178-0.0036,0.0335,0.0165-0.0007,0.827-0.004]\
         + [0.0322,0.2194,0.0349,0.0169,0.831]\
         + [0.0322+0.0017,0.2194+0.0040,0.0349,0.0169+0.0007,0.831+0.004]\
         + [0.0322-0.0017,0.2194-0.0040,0.0349,0.0169-0.0007,0.831-0.004]\
         + [0.0293,0.1432,0.0307,0.0102,0.780]\
         + [0.0293-0.0015,0.1432+0.0021,0.0307,0.0102+0.0009,0.780+0.005]\
         + [0.0293+0.0015,0.1432-0.0021,0.0307,0.0102-0.0009,0.780-0.005]\
         + [0.0300+0.0023,0.1459-0.0025,0.0316,0.0103,0.778+0.006]\
         + [0.0300,0.1459+0.0025,0.0316,0.0103+0.0006,0.778-0.006]\
         + [0.0300-0.0023,0.1459,0.0316,0.0103-0.0006,0.778]\
         + [0.0321+0.0020,0.1563-0.0018,0.0350,0.0105+0.0007,0.782+0.004]\
         + [0.0321-0.0020,0.1563+0.0018,0.0350,0.0105-0.0007,0.782-0.004]\
         + [0.0321,0.1563,0.0350,0.0105,0.782]\
         + [0.0313,0.1608,0.0327,0.0119,0.787]\
         + [0.0313+0.0025,0.1608+0.002,0.0327,0.0119-0.0007,0.787+0.006]\
         + [0.0313-0.0025,0.1608-0.002,0.0327,0.0119+0.0007,0.787-0.006]

r1 = d[d['p'] == 'Recall@1']
r15 = d[d['p'] == 'Recall@15']
p1 = d[d['p'] == 'Pre@1']
p15 = d[d['p'] == 'Precision@15']
dauc = d[d['p'] == 'AUC']

fig = plt.figure()
ax1 = fig.add_subplot(221)
plt.tick_params(labelsize=14)
ax2 = fig.add_subplot(222)
plt.tick_params(labelsize=14)
ax3 = fig.add_subplot(223)
plt.tick_params(labelsize=14)
y_major_locator=MultipleLocator(0.005)
ax3.yaxis.set_major_locator(y_major_locator)
ax4 = fig.add_subplot(224)
plt.tick_params(labelsize=14)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

# sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=100)
plt.rc('patch',force_edgecolor=True)

sns.barplot(x='dataset',y='ret',data=r1,hue='variant',palette='OrRd',ax=ax1, capsize=.05, errwidth=1.5)
sns.barplot(x='dataset',y='ret',data=r15,hue='variant',palette='OrRd',ax=ax2, capsize=.05, errwidth=1.5)
sns.barplot(x='dataset',y='ret',data=p15,hue='variant',palette='OrRd',ax=ax3, capsize=.05, errwidth=1.5)
sns.barplot(x='dataset',y='ret',data=dauc,hue='variant',palette='OrRd',ax=ax4, capsize=.05, errwidth=1.5)

ax1.set_ylim(0.01, 0.05) 
ax1.set_ylabel('Recall@1',font2)

ax2.set_ylim(0.12, 0.23) 
ax2.set_ylabel('Recall@15',font2)

ax3.set_ylim(0.005, 0.02) 
ax3.set_ylabel('Precision@15',font2)

ax4.set_ylim(0.75, 0.85) 
ax4.set_ylabel('AUC',font2)
ax1.set_xlabel(None)
ax2.set_xlabel(None)
ax3.set_xlabel(None)
ax4.set_xlabel(None)

ax1.legend(loc='upper left',prop=font0, ncol=2)
ax2.legend(loc='best',prop=font1, ncol=1)
ax3.legend(loc='best',prop=font1, ncol=1)
ax4.legend(loc='best',prop=font1, ncol=1)
plt.savefig('./mp1.jpg', dpi=500)


# from functools import reduce

# from scipy.stats.stats import mode
# from scipy.sparse import coo_matrix
# import scipy.sparse as sp
# from numpy.random import rand
# import numpy as np
# import math


# import matplotlib.pylab as plt
# import seaborn as sns
# import pandas as pd

# import matplotlib
# from matplotlib.pyplot import MultipleLocator
# matplotlib.rcParams['figure.figsize'] = [15, 5]
# matplotlib.rcParams['savefig.dpi'] = 500
# # fig, axs = plt.subplots(1,1)



# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 15,
# }

# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 15,
# }

# font3 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 20,
# }
# plt.rc('patch',force_edgecolor=True)
# x = np.array([1, 2, 3, 4, 5])
# y1 = np.array([0.782, 0.82260, 0.83155, 0.83112, 0.81987])
# y11 = np.array([0.782-0.003, 0.82260-0.003, 0.83155-0.0035, 0.83112-0.002, 0.81987-0.002])
# y12 = np.array([0.782+0.003, 0.82260+0.003, 0.83155+0.0035, 0.83112+0.002, 0.81987+0.002])
    
# y2 = np.array([0.751, 0.7742,0.7813,0.7641,0.7566])
# y21 = np.array([0.751-0.0015, 0.7742-0.002,0.7813-0.0035,0.7641-0.0025,0.7566-0.0025])
# y22 = np.array([0.751-0.0015, 0.7742+0.002,0.7813+0.0035,0.7641+0.0025,0.7566+0.0025])

# df = pd.DataFrame()
# df['collaborative graph'] = ['MI-GCN (Relation Graph)']* 15 + ['MI-GCN* (Similarity Graph)']*15
# # ['MI-GCN (Relation Graph)','MI-GCN (Relation Graph)','MI-GCN (Relation Graph)','MI-GCN (Relation Graph)','MI-GCN (Relation Graph)','MI-GCN* (Similarity Graph)','MI-GCN* (Similarity Graph)','MI-GCN* (Similarity Graph)','MI-GCN* (Similarity Graph)','MI-GCN* (Similarity Graph)']
# df['Layer'] = [1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5]
# df['AUC'] = np.concatenate((y1,y11,y12,y2,y21,y22))

# # sns.lineplot(x="Layer", y="AUC",hue="collaborative graph", style="collaborative graph", \
# #     # markers={'MI-GCN (Relation Graph)':'o', 'MI-GCN* (Similarity Graph)':'o'},\
# #     markers=True,\
# #          dashes=True, data=df)
# # c1 = 'steelblue'
# # c2 = 'crimson'
# c1 = 'blue'
# c2 = 'red'
# plt.subplot(121)
# ax3 = plt.plot(x, y1, linestyle='-', marker='s',linewidth=2, color=c1, label='MiGCN')
# plt.errorbar(x, y1,
#              yerr=[0.006,0.006,0.007,0.004,0.004],
#              fmt='o',ecolor=c1,color=c1,elinewidth=2,capsize=4
#             )
# ax4 = plt.plot(x, y2, linestyle='--', marker='o',linewidth=2, color=c2, label='MiGCN' + '$\mathregular{_{sim}}$')
# plt.errorbar(x, y2,
#              yerr=[0.003,0.004,0.007,0.005,0.005],
#              fmt='o',ecolor=c2,color=c2,elinewidth=2,capsize=4
#             )

# for a,b in zip(x,y1):
#     if a == 5:
#         plt.text(a-0.15, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
#     else:
#         plt.text(a+0.15, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
# for a,b in zip(x,y2):
#     if a == 5:
#         plt.text(a-0.15, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
#     else:
#         plt.text(a+0.15, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
# plt.grid(True)
# x_major_locator=MultipleLocator(1)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
    
# plt.ylim(0.745, 0.86)
# plt.xlim(0.9, 5.1)
# plt.title('Model Depth ' + r'$L$',font3)
# plt.tick_params(labelsize=13)
# plt.xticks(x,['layer=0', 'layer=1', 'layer=2', 'layer=3', 'layer=4'],fontsize=14)
# plt.ylabel('AUC',font1)
# plt.legend(loc='upper right',prop=font2, ncol=2)

# ##############################################################################################################

# x = np.array([0.1,0.3,0.5,0.7,1.0,1.3,1.5,1.7,2.0])
# y1 = np.array([0.8281, 0.8323, 0.8283, 0.8315, 0.8291, 0.8294, 0.8273, 0.8261, 0.8253])
# # y11 = np.array([y1[0]-0.003, y1[1]-0.003, y1[2]-0.0035, y1[3]-0.002, y1[4]-0.002, y1[5], y1[6], y1[7], y1[8], y1[9]])
# # y12 = np.array([y1[0]+0.003, y1[1]+0.003, y1[2]+0.0035, y1[3]+0.002, y1[4]+0.002, y1[5], y1[6], y1[7], y1[8], y1[9]])
    
# y2 = np.array([0.7774, 0.7716,0.7813,0.7801,0.7711,0.7701, 0.7772, 0.7712, 0.7692])
# # y21 = np.array([y1[0]-0.003, y1[1]-0.003, y1[2]-0.0035, y1[3]-0.002, y1[4]-0.002, y1[5], y1[6], y1[7], y1[8], y1[9], y1[10]])
# # y22 = np.array([y1[0]+0.003, y1[1]+0.003, y1[2]+0.0035, y1[3]+0.002, y1[4]+0.002, y1[5], y1[6], y1[7], y1[8], y1[9], y1[10]])

# c1 = 'blue'
# c2 = 'red'

# plt.subplot(122)
# ax3 = plt.plot(x, y1, linestyle='-', marker='s',linewidth=2, color=c1, label='MI-GCN')
# # plt.errorbar(x, y1,
# #              yerr=[0.006,0.006,0.007,0.004,0.004],
# #              fmt='o',ecolor=c1,color=c1,elinewidth=2,capsize=4
# #             )
# ax4 = plt.plot(x, y2, linestyle='--', marker='o',linewidth=2, color=c2, label='MiGCN' + '$\mathregular{_{sim}}$')
# # plt.errorbar(x, y2,
# #              yerr=[0.003,0.004,0.007,0.005,0.005],
# #              fmt='o',ecolor=c2,color=c2,elinewidth=2,capsize=4
# #             )

# for a,b in zip(x,y1):
#     if a == 2.0:
#         plt.text(a-0.05, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
#     else:
#         if a in [0.3,0.7,1.3,1.7]:
#             plt.text(a+0.05, b-0.008, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
#         else:
#             plt.text(a+0.05, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)

# for a,b in zip(x,y2):
#     if a == 2.0:
#         plt.text(a-0.05, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
#     else:
#         if a in [0.3,0.7,1.3,1.7]:
#             plt.text(a+0.05, b-0.008, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
#         else:
#             plt.text(a+0.05, b+0.003, '%.3f' % b, ha='center', va= 'bottom',fontsize=12)
# plt.grid(True)
# x_major_locator=MultipleLocator(0.5)
# ax=plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
    
# plt.ylim(0.76, 0.85)
# plt.xlim(0,2.1)
# plt.title('Balancing Coefficient ' + r'$\lambda$',font3)
# # plt.xticks(x, [],fontsize=14)
# plt.ylabel('AUC',font1)
# plt.legend(loc='upper right',prop=font2, ncol=2)
# plt.tick_params(labelsize=13)



# plt.savefig('./layer.jpg', dpi=500)








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

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

print(cmd_args)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

print("loading data...")

cmd_args.dataset = './OMIM/bigraph/cross1'
cmd_args.emb_dim = 200
cmd_args.layer_size = "[200]"
cmd_args.batch_size = 512
cmd_args.learning_rate = 0.005
cmd_args.dropout = 0.3
cmd_args.reg=0.001
cmd_args.gamma=0.1
cmd_args.alpha=3
cmd_args.beta=0.1
cmd_args.neg_num=5
cmd_args.k=0.5
cmd_args.Ks = [1,5,10,15]


data_generator = DataLoading(args=cmd_args)

cnt1, cnt2 = 0, 0
for d in range(0,3209):
    print(d)
    d1_set = set(data_generator.adj.tocsr()[d].tocoo().col)
    for d1 in d1_set:
        g1_set = set(data_generator.A.tocsr()[d1].tocoo().col)
        for g1 in g1_set:
            g_set = set(data_generator.adj.tocsr()[g1].tocoo().col)
            for g in g_set:
                # print(d, d1, g1, g, '0')
                cnt1 += 1
                if data_generator.A.tocsr()[d,g] > 0:
                    cnt2 += 1
                    # print(d, d1, g1, g, '1')



print(cnt1)
print(cnt2)