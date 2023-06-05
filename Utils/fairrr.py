import numpy as np
from Utils.utils import *
from sklearn.metrics import mutual_info_score
from optbinning import OptimalBinning

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def fairRR_v1(args, df, mode='train', ignore_co = None, random_co=None, eps_dict = None, mean_dct = None):
    temp_df = df.copy()
    if mode == 'train':
        # calculate mutual information with label
        num_pt = df.shape[0]
        mean_dict = {}
        col_dict = []
        for col in args.feature:
            col_dict.append((col, mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col])))
            mean_dict[col] = temp_df[col].mean()
            # mean_dict[col] = 1.0
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
    num_pt = arr.shape[0]
    arr = np.expand_dims(arr, axis=-1)
    pos_val = np.arange(valdict)
    pos_val = np.tile(pos_val, (num_pt, 1))
    temp = np.tile(arr, (1, valdict))
    temp = np.abs(temp - pos_val)
    kwargs = {"eps": epsilon, "val": valdict}
    res = np.apply_along_axis(random_choose, axis=1, arr=temp, **kwargs)
    return res

def random_choose(arr, eps, val):
    sigma = (val - 1)/eps
    C = np.sum(np.exp(-1*arr/sigma))
    prob = np.exp(-1*arr/sigma)/C
    # print("probability:", prob)
    res = np.random.choice(np.arange(val), size=1, replace=False, p=prob)
    return res

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


def fairRR_org(args, arr, y, z):

    r = arr.shape[1]
    num_pt = arr.shape[0]
    # print("Len y : ", len(y))
    # print("Len z : ", len(z))

    def float_to_binary(x, m=args.num_int, n=args.num_bit - args.num_int - 1):
        x_abs = np.abs(x)
        x_scaled = round(x_abs * 2 ** n)
        res = '{:0{}b}'.format(x_scaled, m + n)
        if x >= 0:
            res = '0' + res
        else:
            res = '1' + res
        return res

    # binary to float
    def binary_to_float(bstr, m=args.num_int, n=args.num_bit - args.num_int - 1):
        sign = bstr[0]
        bs = bstr[1:]
        res = int(bs, 2) / 2 ** n
        if int(sign) == 1:
            res = -1 * res
        return res

    def string_to_int(a):
        bit_str = "".join(x for x in a)
        return np.array(list(bit_str)).astype(int)

    def join_string(a, num_bit=args.num_bit, num_feat=r):
        res = np.empty(num_feat, dtype="S10")
        # res = []
        for i in range(num_feat):
            # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
            res[i] = "".join(str(x) for x in a[i * num_bit:(i + 1) * num_bit])
        return res

    float_to_binary_vec = np.vectorize(float_to_binary)
    binary_to_float_vec = np.vectorize(binary_to_float)

    if args.dataset != 'adult':
        max_val = sum([2 ** i for i in range(args.num_int)]) + sum(
            [2 ** (-1 * i) for i in range(1, args.num_bit - args.num_int)])
        min_val = 2 ** (-1 * (args.num_bit - args.num_int))

        max_ = np.max(arr)
        min_ = np.min(arr)
        arr = (arr - min_) / (max_ - min_) * (max_val - min_val) + min_val
        feat_tmp = float_to_binary_vec(arr)
        feat = np.apply_along_axis(string_to_int, 1, feat_tmp)
    else:
        feat = arr
    mi_protect, mi_label = cal_mi(x=feat, y=y, z=z)
    eps = softmax(mi_label - args.alpha * mi_protect)*args.tar_eps
    # print("Eps : ", eps)
    p = sigmoid(eps)
    p = np.expand_dims(a=p, axis=0)
    p = np.repeat(a=p, repeats=feat.shape[0], axis=0)
    # print("Shape of matrix:", p.shape, feat.shape)
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    # print(np.sum(perturb))
    perturb_feat = (perturb + feat) % 2
    if args.dataset != 'adult':
        perturb_feat = np.apply_along_axis(join_string, 1, perturb_feat)
        perturb_feat = binary_to_float_vec(perturb_feat)
    else:
        perturb_feat = perturb_feat

    # print("Perturb feat : ", np.linalg.norm(perturb_feat - feat, ord=2))
    return perturb_feat

def fairRR(args, df, mode='train', eps = None, dct = None):
    temp_df = df.copy()
    if mode == 'train':
        mean_dict = {}
        mean_sqrt_dict = {}
        val_dict = {}
        # col_dict = []
        for col in args.feature:
            # col_dict.append((col, mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col])))
            mean_dict[col] = temp_df[col].mean()
            mean_sqrt_dict[col] = temp_df[col].apply(lambda x: x**2).mean()
            val_dict[col] = temp_df[col].nunique()
        # col_dict = sorted(col_dict, key=lambda x: x[1])
        # cols = np.array([col[1] for col in col_dict])
        # total_mi = np.sum(cols)
        # cols = (np.abs(np.cumsum(cols) - total_mi)) / (np.abs(np.cumsum(cols)) + total_mi)
        # ignore_col = [col_dict[i][0] for i, col in enumerate(cols) if col > 0.5]
        # randomize_col = [col for col in args.feature if col not in ignore_col]
        # for col in ignore_col:
        #     temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=0, valdict=args.valdict[col])

        total_mi = 0.0
        for col in args.feature:
            total_mi += np.exp(mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col]))
        epsilon = {}
        for col in args.feature:
            mi = mutual_info_score(labels_true=temp_df[args.target], labels_pred=temp_df[col])
            eps = args.tar_eps * np.exp(mi) / total_mi
            epsilon[col] = eps
            temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=eps, valdict=val_dict[col])
        re_dict = (mean_dict, mean_sqrt_dict, val_dict)
        return temp_df, epsilon, re_dict
    else:
        val_dict = dct
        for col in args.feature:
            eps_ = eps[col]
            temp_df[col] = random_bucket(arr=temp_df[col].values, epsilon=eps_, valdict=val_dict[col])
        return temp_df