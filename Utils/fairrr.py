import numpy as np
from Utils.utils import *
from sklearn.metrics import mutual_info_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def fairRR(args, df, mode='train', ignore_co = None, random_co=None, eps_dict = None, mean_dct = None):
    temp_df = df.copy()
    if mode == 'train':
        # calculate mutual information with label
        num_pt = df.shape[0]
        mean_dict = {}
        col_dict = []
        for col in args.feature:
            col_dict.append((col, mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col])))
            mean_dict[col] = temp_df[col].mean()
        col_dict = sorted(col_dict, key=lambda x: x[1])
        cols = np.array([col[1] for col in col_dict])
        # print(cols)
        total_mi = np.sum(cols)
        cols = (np.abs(np.cumsum(cols) - total_mi)) / (np.abs(np.cumsum(cols)) + total_mi)
        # print(cols)
        ignore_col = [col_dict[i][0] for i, col in enumerate(cols) if col > 0.5]
        randomize_col = [col for col in args.feature if col not in ignore_col]
        # randomize ignored columns
        for col in ignore_col:
            if temp_df[col].nunique() > 2:
                temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=0, valdict=args.valdict[col])
            else:
                temp_df[col] = random_bin(arr=temp_df[col].values, epsilon=0, mean=mean_dict[col])

        # randomize best columns
        total_mi = 0.0
        min_mi = 1e12
        for col in randomize_col:
            min_mi = min(min_mi, mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col]))
        for col in randomize_col:
            total_mi += np.exp(mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col]) - min_mi)
        epsilon = {}
        for col in randomize_col:
            mi = mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col])
            eps = args.tar_eps * np.exp(mi-min_mi)/total_mi
            print(col, eps)
            epsilon[col] = eps
            if temp_df[col].nunique() > 2:
                temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=eps, valdict=args.valdict[col])
            else:
                temp_df[col] = random_bin(arr=temp_df[col].values, epsilon=eps, mean=mean_dict[col])
        cols = (ignore_col, randomize_col)
        return temp_df, cols, epsilon, mean_dict
    else:
        # print(eps_dict)
        for col in ignore_co:
            if temp_df[col].nunique() > 2:
                temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=0, valdict=args.valdict[col])
            else:
                temp_df[col] = random_bin(arr=temp_df[col].values, epsilon=0, mean=mean_dct[col])

        for col in random_co:
            eps = eps_dict[col]
            if temp_df[col].nunique() > 2:
                temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=eps, valdict=args.valdict[col])
            else:
                temp_df[col] = random_bin(arr=temp_df[col].values, epsilon=eps, mean=mean_dct[col])
        return temp_df


def random_bin(arr, epsilon, mean):
    temp = np.random.uniform(low=0.0, high=1.0, size=arr.shape)
    if epsilon == 0:
        temp = (temp > 0.5).astype(int)
        alpha = mean/(0.5*mean + 0.5*(1-mean))
    else:
        p = np.exp(epsilon)/(1 + np.exp(epsilon))
        alpha = mean / (p * mean + (1-p) * (1 - mean))
        temp = (temp > p).astype(int)
    perturbed = ((temp + arr) % 2).astype(int)
    return perturbed*alpha

def random_bucket(arr, epsilon, valdict):
    pass

def cal_mi(x, y, z):
    mi_protect = []
    mi_label = []
    for i in range(x.shape[1]):
        mi_protect.append(mutual_info_score(labels_true=z, labels_pred=x[:, i]))
        mi_label.append(mutual_info_score(labels_true=y, labels_pred=x[:, i]))
    mi_protect = np.array(mi_protect)
    mi_label = np.array(mi_label)
    mi_protect = mi_protect / np.max(mi_protect)
    mi_label = mi_label / np.max(mi_label)
    mi_protect = (mi_protect - np.min(mi_protect)) / (np.max(mi_protect) - np.min(mi_protect)) * (
                np.max(mi_label) - np.min(mi_label)) + np.min(mi_label)
    return mi_protect, mi_label
