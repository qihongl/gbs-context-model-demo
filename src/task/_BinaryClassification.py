import numpy as np
from utils import to_pth

N_CLASSES = 2

class BinaryClassification():

    ''' gen a 2D binary classification problem
    given an angle theta
    define p with polar coord (1, theta) on a unit circle
    use p and -p as the two cluster center to form a classification problem
    '''

    def __init__(self, theta, cluster_std=.3):
        # constants
        self.n_classes = 2
        self.n_dim = 2
        #
        self.cluster_std = cluster_std
        self.theta = theta
        self.p = generate_random_point_on_circle(theta=theta)

    def get_data(self, sample_size=100, shuffle=True):
        half_sample_size = sample_size//2
        # gen data
        data_pos = np.random.normal(loc= self.p, scale=self.cluster_std, size=(half_sample_size, self.n_dim))
        data_neg = np.random.normal(loc=-self.p, scale=self.cluster_std, size=(half_sample_size, self.n_dim))
        X = np.vstack([data_pos, data_neg])
        Y = np.vstack([-np.ones((half_sample_size, 1)), np.ones((half_sample_size, 1))])
        # shuffle the data
        if shuffle:
            perm = np.random.permutation(len(Y))
            X = X[perm,:]
            Y = Y[perm,:]
        return X, Y


def generate_random_point_on_circle(r=1, theta=None):
    # random angle
    theta = 2 * np.pi * np.random.uniform() if theta is None else theta
    # calculating coordinates
    return np.array(r * [np.cos(theta), np.sin(theta)])



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', palette='colorblind', context='talk')
    np.random.seed(0)

    '''test - gen a classification problem '''

    sample_size=200

    for i in np.arange(0, 1, .1):
        theta = 2 * np.pi * i
        bc = BinaryClassification(theta)
        X, Y = bc.get_data(sample_size=sample_size)

        f, ax = plt.subplots(1,1, figsize=(6, 5))
        ax.scatter(X[:,0], X[: ,1], c=np.squeeze(Y), cmap='RdBu', alpha=.5)
        ax.set_title(f'i = {i}')
        ax.axhline(0, ls='--', c='grey')
        ax.axvline(0, ls='--', c='grey')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        sns.despine()
