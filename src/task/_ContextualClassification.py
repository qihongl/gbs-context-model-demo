import numpy as np
from utils import to_pth
from sklearn.datasets import make_blobs
from task._BinaryClassification import BinaryClassification

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='talk')
N_CLASSES = 2

class ContextualClassification():

    '''task
    X: data sampled from 2 n-d gaussian distribution centered at 1 or -1
    y: in task 0: 1/-1 for center 1/-1; in task 1: the opposite
    '''
    def __init__(self, thetas, cluster_std=1):
        # constants
        self.n_classes = 2
        self.n_dim = 2
        self.cluster_std = cluster_std

        self.thetas = thetas
        self.n_tasks = len(thetas)
        self.bcs = [BinaryClassification(theta=theta, cluster_std=cluster_std) for theta in thetas]
        # log str
        self.log_str = f'cc/n_tasks-{self.n_tasks}-cluster_std-{cluster_std}'

    def get_data(self, block_size, n_repeats, to_pytorch=True):
        ''' gen n data pts for the 2 tasks
        block_size: int
        - # trials for each task
        n_repeats: int
        - each repeat loop over all task ids sequentially
        '''
        task_ids = gen_blocked_task_ids(self.n_tasks, block_size, n_repeats)
        X = []
        Y = []
        for i in range(n_repeats):

            for j in range(self.n_tasks):
                # task_id_ij = task_ids[i][j][0]
                X_ij, Y_ij = self.bcs[int(task_ids[i][j][0])].get_data(block_size)
                # append data
                X.append(X_ij)
                Y.append(Y_ij)
        # reformat data
        X = np.vstack(X)
        Y = np.vstack(Y)
        if to_pytorch:
            X, Y = to_pth(X), to_pth(Y)
        return X, Y, np.concatenate(np.concatenate(task_ids))

    def plot_task_ids(self, task_ids, figsize=(7,3)):
        f, ax = plt.subplots(1,1, figsize=figsize)
        ax.plot(task_ids)
        ax.set_xlabel('Time')
        ax.set_ylabel('task id')
        sns.despine()
        return f, ax

    def plot_all_data(self, X, Y, figsize=(6,5)):
        f, ax = plt.subplots(1,1, figsize=(6, 5))
        ax.scatter(X[:,0], X[: ,1], c=np.squeeze(Y), cmap='RdBu', alpha=.5)
        ax.set_title(f'data, ALL data')
        ax.set_xlabel('input dim 1')
        ax.set_ylabel('input dim 2')
        ax.axhline(0, ls='--', c='grey')
        ax.axvline(0, ls='--', c='grey')
        sns.despine()
        return f, ax

    def plot_data_by_task(self, X, Y, task_ids, angles=None):
        f, axes = plt.subplots(1, self.n_tasks, figsize=(self.n_tasks * 4, 4))
        for task_id in range(self.n_tasks):
            axes[task_id].scatter(
                X[task_ids==task_id,0], X[task_ids==task_id ,1],
                c=np.squeeze(Y)[task_ids==task_id], cmap='RdBu', alpha=.5
            )
            title_str = f'task id = {task_id}'
            if angles is not None:
                title_str += ', angle = %d'% (angles[task_id])
            axes[task_id].set_title(title_str)
            axes[task_id].axhline(0, ls='--', c='grey')
            axes[task_id].axvline(0, ls='--', c='grey')
            axes[task_id].set_xlabel('input dim 1')
            axes[task_id].set_ylabel('input dim 2')
            axes[task_id].set_xlim([-2, 2])
            axes[task_id].set_ylim([-2, 2])
        sns.despine()
        f.tight_layout()
        return f, axes

    def make_task_RDM(self):
        thetas_in_deg = self.rad2deg()
        thetas_in_deg = thetas_in_deg.reshape(-1,1)

        task_rdm = np.zeros((self.n_tasks, self.n_tasks))
        for i, a in enumerate(thetas_in_deg):
            for j, b in enumerate(thetas_in_deg):
                task_rdm[i, j] = angle_difference(a, b)
        return task_rdm

    def plot_RDM(self, rdm, figsize=(6,5), cmap='magma'):
        mask = np.triu(rdm)
        f, ax = plt.subplots(1,1, figsize=figsize)
        sns.heatmap(rdm, square=True, mask=mask, cmap=cmap)
        ax.set_xlabel('task id')
        ax.set_ylabel('task id')
        return f, ax


    def rad2deg(self):
        angles = np.rad2deg(self.thetas) + 90
        angles[angles>360] -= 360
        return angles

    def smallest_angle_diff(self):
        angles = self.rad2deg()
        return int(angles[1] - angles[0])

def angle_difference(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def gen_blocked_task_ids(n_tasks, block_size, n_repeats, randomize=True):
    if randomize:
        return [[np.ones(block_size) * i for i in np.random.permutation(range(n_tasks))]
                for _ in range(n_repeats)]
    return [[np.ones(block_size) * i for i in range(n_tasks)]
            for _ in range(n_repeats)]



if __name__ == '__main__':
    np.random.seed(0)

    '''how to use each bandit task'''
    cluster_std=.2
    # thetas = [2 * np.pi * i for i in np.arange(0, 1, 1/3)]
    thetas = [2 * np.pi * i for i in np.arange(0, 1, .125)]
    # thetas = [2 * np.pi * i for i in np.arange(0, 1, .25)]
    cc = ContextualClassification(thetas, cluster_std=cluster_std)
    n_tasks = cc.n_tasks
    print(cc.n_tasks)
    angles = cc.rad2deg()
    print(angles)
    print(cc.smallest_angle_diff())

    block_size = 100
    n_repeats = 10

    X, Y, task_ids = cc.get_data(block_size, n_repeats, to_pytorch=False)
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(task_ids))

    # plot data
    cc.plot_task_ids(task_ids)
    cc.plot_all_data(X, Y)
    cc.plot_data_by_task(X, Y, task_ids, angles=angles)

    task_RDM = cc.make_task_RDM()
    cc.plot_RDM(task_RDM)
    cc.thetas
