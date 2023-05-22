import numpy as np
from Utils.utils import *
from sklearn.metrics import mutual_info_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def fairRR(args, arr):
    r = arr.shape[1]
    num_pt = arr.shape[0]

    def float_to_binary(x, m=args.num_int, n=args.num_bit - args.num_int):
        x_abs = np.abs(x)
        x_scaled = round(x_abs * 2 ** n)
        res = '{:0{}b}'.format(x_scaled, m + n)
        # if x >= 0:
        #     res = '0' + res
        # else:
        #     res = '1' + res
        return res

    # binary to float
    def binary_to_float(bstr, n=args.num_bit - args.num_int):
        # sign = bstr[0]
        # bs = bstr[1:]
        res = int(bstr, 2) / 2 ** n
        # if int(sign) == 1:
        #     res = -1 * res
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

    max_ = np.max(arr)
    min_ = np.min(arr)
    arr = (arr - min_) / (max_ - min_) * (2 ** args.num_int - 1)

    float_to_binary_vec = np.vectorize(float_to_binary)
    binary_to_float_vec = np.vectorize(binary_to_float)

    feat_tmp = float_to_binary_vec(arr)
    feat = np.apply_along_axis(string_to_int, 1, feat_tmp)
    print(feat[0,:10])
    index_matrix = np.array(range(args.num_bit))
    index_matrix = np.tile(index_matrix, (num_pt, r))
    p = np.ones_like(index_matrix)*(np.exp(args.tar_eps/(r*args.num_bit))/(1 + np.exp(args.tar_eps/(r*args.num_bit))))
    print(p[0, :10])
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)
    perturb_feat = (perturb + feat) % 2
    print(perturb_feat[0, :10])
    perturb_feat = np.apply_along_axis(join_string, 1, perturb_feat)
    perturb_feat = binary_to_float_vec(perturb_feat)
    perturb_feat = perturb_feat / (2 ** args.num_int - 1) * (max_ - min_) + min_
    return perturb_feat


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
