import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import to_np
from stats import moving_average
from sklearn.decomposition import PCA
from model import SimpleContextNetwork_v2 as SimpleContextNetwork
from task import ContextualClassification
from torch.optim.lr_scheduler import StepLR

sns.set(style='white', palette='colorblind', context='talk')
cpal = sns.color_palette('colorblind')
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

'''init task'''
n_dim = 2
cluster_std = .3
block_size = 200
n_repeats = 10
buf_time = 10

# init task
thetas = [2 * np.pi * i for i in np.arange(0, 1, .5)]
cc = ContextualClassification(thetas, cluster_std=cluster_std)
angles = cc.rad2deg()
X, Y, task_ids = cc.get_data(block_size, n_repeats)
T = len(task_ids)
event_bonds = np.arange(0, T, block_size)[1:]

# plot data
cc.plot_all_data(X, Y)
cc.plot_data_by_task(X, Y, task_ids)
cc.plot_task_ids(task_ids, figsize=(12,3))



'''model params'''
dim_input = n_dim
dim_cinput = 8
dim_hidden = 8
dim_output = 1
w_lr = 1e-3
c_lr = 1e-3
n_iters = 1000
size = block_size * 2

'''train'''
# init model
# setting seed so that the two nets have the same init weights
torch.manual_seed(seed)
agent = SimpleContextNetwork(dim_input, dim_hidden, dim_output, dim_cinput, w_lr, c_lr)
torch.manual_seed(seed)
agent_wc = SimpleContextNetwork(dim_input, dim_hidden, dim_output, dim_cinput, w_lr, c_lr)

# train
log_loss = torch.zeros(T)
log_loss_wc = torch.zeros(T)
log_acc = torch.zeros(T)
log_acc_wc = torch.zeros(T)
log_c = torch.zeros((T, dim_hidden))
log_cr = torch.zeros((T, dim_hidden))
log_h = torch.zeros((T, dim_hidden))
log_tp_nwu = np.zeros(T, )

for t, task_id_t in enumerate(task_ids):
    yhat = agent.forward(X[t])
    yhat_wc = agent_wc.forward(X[t])
    # forward
    loss = agent.criterion(yhat, Y[t])
    loss_wc = agent_wc.criterion(yhat_wc, Y[t])
    # find context rep if at event bounds
    if (t-buf_time) in event_bonds:
        agent_wc.update_context(Y[t-buf_time:t], X[t-buf_time:t,:], n_iters=n_iters)
    if t % block_size > buf_time:
        # w update
        agent.update_weights(loss)
        agent_wc.update_weights(loss_wc)

    # logging
    log_loss[t] = loss
    log_loss_wc[t] = loss_wc
    log_acc[t] = Y[t] == torch.round(yhat)
    log_acc_wc[t] = Y[t] == torch.round(yhat_wc)
    act = agent_wc.get_context_rep(X[t], return_all=True)
    # get c
    log_cr[t], _, log_h[t] = act
    print(f'Epoch %3d | loss = %.2f, loss_wc = %.2f' % (t, loss, loss_wc))


'''plot'''

# plot the loss
alpha = .7
f, ax = plt.subplots(1,1, figsize=(12, 4))
ax.plot(to_np(log_loss), label = 'no context update', alpha=alpha)
ax.plot(to_np(log_loss_wc), label = 'with context update', alpha=alpha)
ax.plot(log_tp_nwu, color='grey', alpha=.5)

for ii, eb in enumerate(event_bonds):
    label = 'event boundaries' if ii == 0 else None
    ax.axvline(eb, linestyle='--', color='grey', label=label)
ax.set_title('Learning curves')
ax.set_xlabel('time')
ax.set_ylabel('loss')
ax.legend()
sns.despine()

# plot the loss
alpha = .7
win_size = 10
log_acc_ma = moving_average(to_np(log_acc), win_size)
log_acc_wc_ma = moving_average(to_np(log_acc_wc), win_size)
f, ax = plt.subplots(1,1, figsize=(12, 4))
ax.plot(log_acc_ma, label = 'no context update', alpha=alpha)
ax.plot(log_acc_wc_ma, label = 'with context update', alpha=alpha)
for ii, eb in enumerate(event_bonds):
    label = 'event boundaries' if ii == 0 else None
    ax.axvline(eb, linestyle='--', color='grey', label=label)
ax.set_title('Performance')
ax.set_xlabel('time')
ax.set_ylabel('accuracy')
# ax.legend()
sns.despine()


# pca the context representation
k = 2
pca = PCA(n_components=k)
log_cr_pc = pca.fit_transform(to_np(log_cr))
log_h_pc = pca.fit_transform(to_np(log_h))

# jitter the data
jitter = np.random.normal(scale=.01, size=(T,k))
log_cr_pc_j = log_cr_pc + jitter

# plot the context representation
f, ax = plt.subplots(1,1, figsize=(6, 5))
# make the mask that masks out buffer time to clean up  data analysis
mask_exclude_buffer_time = np.ones(T)
for eb in event_bonds:
    mask_exclude_buffer_time[eb: eb+buf_time] = 0
for i, task_id in enumerate(np.arange(cc.n_tasks)):
    mask = task_id == task_ids
    mask = np.logical_and(mask, mask_exclude_buffer_time)
    ax.scatter(log_cr_pc_j[mask,0], log_cr_pc_j[mask,1], c=cpal[i],
        alpha=alpha, label=f'{task_id}')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_title('The learned context representation')
ax.legend(title='task id')
drift_dist = []
for eb in [0] + list(event_bonds[:-1]):
    a,b,c,d = eb+buf_time, eb+block_size, eb+block_size+buf_time, eb+block_size*2
    # print(a,b,c,d)
    v = np.vstack([np.mean(log_cr_pc[a:b,:],axis=0), np.mean(log_cr_pc[c:d,:],axis=0)])
    ax.quiver(*v[0], *(v[1] - v[0]), scale_units='xy', angles='xy', scale=1, alpha=alpha)
    drift_dist.append(np.linalg.norm(v[0] - v[1]))
sns.despine()

f, ax = plt.subplots(1,1, figsize=(10, 4))
ax.plot(np.array(drift_dist))
ax.set_title('Context representation drift across blocks during training')
ax.set_xlabel(f'Block id (each block contains {block_size} trials)')
ax.set_ylabel('Representation drift, L2 norm')
sns.despine()


# plot the context representation
n_trial_one_loop = block_size * cc.n_tasks
data_last_loop = log_cr_pc[-n_trial_one_loop:,:]
task_ids_last_loop = task_ids[-n_trial_one_loop:]
mask_exclude_buffer_time_last_loop = mask_exclude_buffer_time[-n_trial_one_loop:]
final_mean_cr = np.zeros((cc.n_tasks,k))
f, ax = plt.subplots(1,1, figsize=(6, 5))
for i, task_id in enumerate(np.arange(cc.n_tasks)):
    mask = task_id == task_ids_last_loop
    mask = np.logical_and(mask, mask_exclude_buffer_time_last_loop)
    ax.scatter(data_last_loop[mask,0], data_last_loop[mask,1], c=cpal[i], alpha=alpha, label=f'{task_id}')
    final_mean_cr[i] = np.mean(data_last_loop[mask], axis=0)
ax.legend(title='task id')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_title(f'The learned context representation \n (last {cc.n_tasks} blocks)')
sns.despine()


f, ax = plt.subplots(1,1, figsize=(6, 5))
ax.set_title('Context representation during training')
min, max = np.min(to_np(log_cr)), np.max(to_np(log_cr))
max_minmax = np.max([np.abs(min), np.abs(max)])
vmin, vmax = - max_minmax, max_minmax

im = sns.heatmap(log_cr.data, cmap='RdYlBu', center=0, vmin=vmin, vmax=vmax)
ax.set_xlabel('context dim')
ax.set_ylabel('time')
