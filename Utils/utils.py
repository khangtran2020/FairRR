import pickle
import random
import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from Models.models import *
from torch.utils.data import DataLoader
from Data.datasets import Data
import pandas as pd
from Utils.fairrr import fairRR, fairRR_org, fairRR_v1


def save_res(args, dct, name):
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_name(args, current_date, fold=0):
    dataset_str = f'{args.dataset}_run_{args.seed}_{fold}_{args.ratio}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}-{current_date.second}'
    model_str = f'{args.mode}_{args.submode}_{args.epochs}_{args.performance_metric}_{args.optimizer}_{args.model_type}_'
    dp_str = f'eps_{args.tar_eps}_'
    if args.mode in ['clean']:
        res_str = dataset_str + model_str + date_str
    else:
        res_str = dataset_str + model_str + dp_str + date_str
    return res_str

def init_model(args):
    if args.model_type == 'NormNN':
        return NormNN(args.input_dim, args.n_hid, args.output_dim, n_layer=args.n_layer, dropout=None)
    elif args.model_type == 'NN':
        return NN(args.input_dim, args.n_hid, args.output_dim, n_layer=args.n_layer)
    elif args.model_type == 'LR':
        return LR(args.input_dim, args.output_dim)
    elif args.model_type == 'SimpleCNN':
        print(args.input_dim, args.n_hid, args.output_dim)
        return SimpleCNN(args.input_dim, args.n_hid, args.output_dim)


def init_data(args, fold, train, test):
    train_df = train
    test_df = test

    df_train = train_df[train_df.fold != fold]
    df_valid = train_df[train_df.fold == fold]

    if args.submode == 'fairrr':
        print('=' * 10 + ' Applying FairRR ' + '=' * 10)
        # args, df, mode = 'train', ignore_co = None, random_co = None, eps_dict = None

        tr_df, epsilon, re_dict = fairRR(args=args, df=df_train, mode='train')
        mean_dict, mean_sqrt_dict, val_dict = re_dict
        va_df = fairRR(args=args, df=df_valid, mode='val', eps = epsilon, dct = val_dict)
        te_df = fairRR(args=args, df=test_df, mode='test', eps = epsilon, dct = val_dict)

        # normalizing
        for col in args.feature:
            u0 = tr_df[col].mean()
            u1 = mean_dict[col]
            u2 = tr_df[col].apply(lambda x: x**2).mean()
            u3 = mean_sqrt_dict[col]
            alpha = np.sqrt((u3 - u1**2)/(u2 - u0**2))
            beta = u1 - u0*alpha
            tr_df[col] = alpha*tr_df[col] + beta

            u0 = va_df[col].mean()
            u2 = va_df[col].apply(lambda x: x ** 2).mean()
            alpha = np.sqrt((u3 - u1 ** 2) / (u2 - u0 ** 2))
            beta = u1 - u0 * alpha
            va_df[col] = alpha * va_df[col] + beta

            u0 = te_df[col].mean()
            u2 = te_df[col].apply(lambda x: x ** 2).mean()
            alpha = np.sqrt((u3 - u1 ** 2) / (u2 - u0 ** 2))
            beta = u1 - u0 * alpha
            te_df[col] = alpha * te_df[col] + beta

        print('=' * 10 + ' Done FairRR ' + '=' * 10)

        all_data = pd.concat([tr_df, va_df, te_df], axis = 0)
        for col in args.feature:
            all_data[col] = (all_data[col] - all_data[col].mean())/(all_data[col].std() + 1e-12)

        tr_df = all_data[:tr_df.shape[0]]
        va_df = all_data[tr_df.shape[0]:tr_df.shape[0] + va_df.shape[0]]
        te_df = all_data[tr_df.shape[0] + va_df.shape[0]:]

        male_te_df = te_df[te_df[args.z] == 1].copy().reset_index(drop=True)
        female_te_df = te_df[te_df[args.z] == 0].copy().reset_index(drop=True)

        x_tr = tr_df[args.feature].values
        y_tr = tr_df[args.target].values
        z_tr = tr_df[args.z].values

        x_va = va_df[args.feature].values
        y_va = va_df[args.target].values
        z_va = va_df[args.z].values

        x_te = te_df[args.feature].values
        y_te = te_df[args.target].values
        z_te = te_df[args.z].values

        x_fem_te = female_te_df[args.feature].values
        y_fem_te = female_te_df[args.target].values
        z_fem_te = female_te_df[args.z].values

        x_mal_te = male_te_df[args.feature].values
        y_mal_te = male_te_df[args.target].values
        z_mal_te = male_te_df[args.z].values
    elif args.submode == 'fairrr_org':

        print('=' * 10 + ' Applying FairRR ' + '=' * 10)
        # args, df, mode = 'train', ignore_co = None, random_co = None, eps_dict = None, mean_dct = None
        tr_df, cols, epsilon, mean_dict = fairRR_v1(args=args, df=df_train, mode='train')
        ignore_col, random_col = cols
        args.feature = random_col
        va_df = fairRR_v1(args=args, df=df_valid, mode='val', ignore_co=ignore_col, random_co=random_col,
                          eps_dict=epsilon, mean_dct=mean_dict)
        te_df = fairRR_v1(args=args, df=test_df, mode='test', ignore_co=ignore_col, random_co=random_col,
                          eps_dict=epsilon, mean_dct=mean_dict)
        print('=' * 10 + ' Done FairRR ' + '=' * 10)

        all_data = pd.concat([tr_df, va_df, te_df], axis=0)
        for col in args.feature:
            all_data[col] = (all_data[col] - all_data[col].mean()) / (all_data[col].std() + 1e-12)

        tr_df = all_data[:tr_df.shape[0]]
        va_df = all_data[tr_df.shape[0]:tr_df.shape[0] + va_df.shape[0]]
        te_df = all_data[tr_df.shape[0] + va_df.shape[0]:]

        male_te_df = te_df[te_df[args.z] == 1].copy().reset_index(drop=True)
        female_te_df = te_df[te_df[args.z] == 0].copy().reset_index(drop=True)

        x_tr = tr_df[args.feature].values
        y_tr = tr_df[args.target].values
        z_tr = tr_df[args.z].values

        x_va = va_df[args.feature].values
        y_va = va_df[args.target].values
        z_va = va_df[args.z].values

        x_te = te_df[args.feature].values
        y_te = te_df[args.target].values
        z_te = te_df[args.z].values

        x_fem_te = female_te_df[args.feature].values
        y_fem_te = female_te_df[args.target].values
        z_fem_te = female_te_df[args.z].values

        x_mal_te = male_te_df[args.feature].values
        y_mal_te = male_te_df[args.target].values
        z_mal_te = male_te_df[args.z].values
    else:
        male_te_df = test_df[test_df[args.z] == 1].copy().reset_index(drop=True)
        female_te_df = test_df[test_df[args.z] == 0].copy().reset_index(drop=True)

        x_tr = df_train[args.feature].values
        y_tr = df_train[args.target].values
        z_tr = df_train[args.z].values

        x_va = df_valid[args.feature].values
        y_va = df_valid[args.target].values
        z_va = df_valid[args.z].values

        x_te = test_df[args.feature].values
        y_te = test_df[args.target].values
        z_te = test_df[args.z].values

        x_fem_te = female_te_df[args.feature].values
        y_fem_te = female_te_df[args.target].values
        z_fem_te = female_te_df[args.z].values

        x_mal_te = male_te_df[args.feature].values
        y_mal_te = male_te_df[args.target].values
        z_mal_te = male_te_df[args.z].values

        # print('=' * 10 + ' Applying FairRR ' + '=' * 10)
        # # (args, arr, y, z)
        # x_tr = fairRR_org(args=args, arr=x_tr, y=y_tr, z=z_tr)
        # x_va = fairRR_org(args=args, arr=x_va, y=y_va, z=z_va)
        # x_te = fairRR_org(args=args, arr=x_te, y=y_te, z=z_te)
        # x_mal_te = fairRR_org(args=args, arr=x_mal_te, y=y_mal_te, z=z_mal_te)
        # x_fem_te = fairRR_org(args=args, arr=x_fem_te, y=y_fem_te, z=z_fem_te)
        # print('=' * 10 + ' Done FairRR ' + '=' * 10)

    # get numpy


    # Defining DataSet

    ## train
    train_dataset = Data(X=x_tr, y=y_tr, ismale=z_tr)
    ## valid
    valid_dataset = Data(X=x_va, y=y_va, ismale=z_va)

    ## test
    test_male_dataset = Data(X=x_mal_te, y=y_mal_te, ismale=z_mal_te)
    test_female_dataset = Data(X=x_fem_te, y=y_fem_te, ismale=z_fem_te)
    test_dataset = Data(X=x_te, y=y_te, ismale=z_te)

    tr_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True,
                           pin_memory=True, drop_last=True, )

    va_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                           pin_memory=True, drop_last=False)

    te_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False,
                           pin_memory=True, drop_last=False)
    te_mal_loader = DataLoader(test_male_dataset, batch_size=args.batch_size, pin_memory=True,
                               drop_last=False, shuffle=False, num_workers=0)
    te_fem_loader = DataLoader(test_female_dataset, batch_size=args.batch_size, pin_memory=True,
                               drop_last=False, shuffle=False, num_workers=0)

    args.n_batch = len(tr_loader)
    args.bs_male = args.batch_size
    args.bs_female = args.batch_size
    args.bs = args.batch_size
    args.num_val_male = len(male_te_df)
    args.num_val_female = len(female_te_df)

    tr_info = tr_loader
    va_info = va_loader
    te_info = (te_loader, te_mal_loader, te_fem_loader)
    return tr_info, va_info, te_info


def init_history():
    history = {
        'tr_loss': [],
        'tr_acc': [],
        'va_loss': [],
        'va_acc': [],
        'demo_parity': [],
        'acc_parity': [],
        'equal_opp': [],
        'equal_odd': [],
        'te_loss': [],
        'te_acc': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_acc_parity': 0,
        'best_equal_opp': 0,
        'best_equal_odd': 0,
    }
    return history
