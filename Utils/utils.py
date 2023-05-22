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
from Utils.fairrr import fairRR


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
    elif args.model_type == 'SimpleCNN':
        print(args.input_dim, args.n_hid, args.output_dim)
        return SimpleCNN(args.input_dim, args.n_hid, args.output_dim)


def init_data(args, fold, train, test):
    mal_tr_df, fem_tr_df = train
    test_df, mal_te_df, fem_te_df = test

    df_train_mal = mal_tr_df[mal_tr_df.fold != fold]
    df_train_fem = fem_tr_df[fem_tr_df.fold != fold]
    df_val_mal = mal_tr_df[mal_tr_df.fold == fold]
    df_val_fem = fem_tr_df[fem_tr_df.fold == fold]
    df_train = pd.concat([df_train_mal, df_train_fem], axis=0).sample(frac=1.0).reset_index(drop=True)
    df_valid = pd.concat([df_val_mal, df_val_fem], axis=0).sample(frac=1.0).reset_index(drop=True)

    # get numpy
    x_tr = df_train[args.feature].values
    y_tr = df_train[args.target].values
    z_tr = df_train[args.z].values

    x_va = df_valid[args.feature].values
    y_va = df_valid[args.target].values
    z_va = df_valid[args.z].values

    x_te = test_df[args.feature].values
    y_te = test_df[args.target].values
    z_te = test_df[args.z].values

    x_fem_te = fem_te_df[args.feature].values
    y_fem_te = fem_te_df[args.target].values
    z_fem_te = fem_te_df[args.z].values

    x_mal_te = mal_te_df[args.feature].values
    y_mal_te = mal_te_df[args.target].values
    z_mal_te = mal_te_df[args.z].values

    if args.submode == 'fairrr':
        print('=' * 10 + ' Applying FairRR ' + '=' * 10)
        x_tr = fairRR(args=args, arr=x_tr)
        x_va = fairRR(args=args, arr=x_va)
        x_te = fairRR(args=args, arr=x_te)
        x_fem_te = fairRR(args=args, arr=x_fem_te)
        x_mal_te = fairRR(args=args, arr=x_mal_te)
        print('=' * 10 + ' Done FairRR ' + '=' * 10)

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
    args.num_val_male = len(mal_te_df)
    args.num_val_female = len(fem_te_df)

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
