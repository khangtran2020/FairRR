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


def save_res(fold, args, dct, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)


def get_name(args, current_date, fold):
    return '{}_{}_fold_{}_metric_{}_eps_{}_epochs_{}_{}-{}-{}_{}-{}-{}'.format(args.dataset,
                                                                               args.mode, fold,
                                                                               args.performance_metric,
                                                                               args.tar_eps,
                                                                               args.epochs,
                                                                               current_date.day,
                                                                               current_date.month,
                                                                               current_date.year,
                                                                               current_date.hour,
                                                                               current_date.minute,
                                                                               current_date.second)


def init_model(args):
    if args.model_type == 'NormNN':
        return NormNN(args.input_dim, args.n_hid, args.output_dim)
    elif args.model_type == 'NN':
        return NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
    elif args.model_type == 'Logit':
        return NormLogit(args.input_dim, args.n_hid, args.output_dim)
    elif args.model_type == 'LogitSmooth':
        return Logit(args.input_dim, args.n_hid, args.output_dim)
    elif args.model_type == 'CNN':
        return CNN(args.input_dim, args.n_hid, args.output_dim)


def init_data(args, fold, train_df, test_df):
    train_df, test_df = normalize_data(args=args, train_df=train_df, test_df=test_df)
    male_df = train_df[train_df[args.z] == 1]
    female_df = train_df[train_df[args.z] == 0]

    df_train = train_df[train_df.fold != fold]
    df_valid = train_df[train_df.fold == fold]

    train_idx = list(df_train.index)
    valid_idx = list(df_train.index)
    train_mal_idx = list(male_df[male_df.fold != fold].index)
    valid_mal_idx = list(male_df[male_df.fold == fold].index)
    train_fem_idx = list(female_df[female_df.fold != fold].index)
    valid_fem_idx = list(female_df[female_df.fold == fold].index)


    x_train, y_train, z_train = (train_df[args.feature].values, train_df[args.target].values, train_df[args.z].values)
    x_test, y_test, z_test = (test_df[args.feature].values, test_df[args.target].values, test_df[args.z].values)

    if args.mode == 'fairrr':
        print('='*10 + ' Applying FairRR ' + '='*10)
        x_train = fairRR(args=args, arr=x_train, y=y_train, z=z_train)
        x_test = fairRR(args=args, arr=x_test, y=y_test, z=z_test)
        print('=' *10 + ' Done FairRR ' + '=' * 10)

    # Defining DataSet
    train_male_dataset = Data(
        X=x_train[train_mal_idx, :],
        y=y_train[train_mal_idx],
        ismale=z_train[train_mal_idx]
    )

    train_female_dataset = Data(
        X=x_train[train_fem_idx, :],
        y=y_train[train_fem_idx],
        ismale=z_train[train_fem_idx]
    )

    valid_male_dataset = Data(
        X=x_train[valid_mal_idx, :],
        y=y_train[valid_mal_idx],
        ismale=z_train[valid_mal_idx]
    )

    valid_female_dataset = Data(
        X=x_train[valid_fem_idx, :],
        y=y_train[valid_fem_idx],
        ismale=z_train[valid_fem_idx]
    )

    train_dataset = Data(
        X=x_train[train_idx, :],
        y=y_train[train_idx],
        ismale=z_train[train_idx]
    )

    valid_dataset = Data(
        X=x_train[valid_idx, :],
        y=y_train[valid_idx],
        ismale=z_train[valid_idx]
    )

    test_dataset = Data(
        X=x_test,
        y=y_test,
        ismale=z_test
    )

    # Defining DataLoader with BalanceClass Sampler
    # sampler_male = torch.utils.data.RandomSampler(train_male_dataset, replacement=False)
    train_male_loader = DataLoader(
        train_male_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    # sampler_female = torch.utils.data.RandomSampler(train_female_dataset, replacement=False)
    train_female_loader = DataLoader(
        train_female_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    valid_male_loader = torch.utils.data.DataLoader(
        valid_male_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    valid_female_loader = torch.utils.data.DataLoader(
        valid_female_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    args.n_batch = len(train_male_loader)
    args.bs_male = int(args.sampling_rate * len(train_male_dataset))
    args.bs_female = int(args.sampling_rate * len(train_female_dataset))
    args.bs = int(args.sampling_rate * len(train_dataset))
    args.num_val_male = len(valid_mal_idx)
    args.num_val_female = len(valid_fem_idx)
    return train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader


def get_grad_vec(model, device, requires_grad=False):
    size = 0
    for name, layer in model.named_parameters():
        if name == 'decoder.weight':
            continue
        size += layer.view(-1).shape[0]
    if device.type == 'cpu':
        sum_var = torch.FloatTensor(size).fill_(0)
    else:
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        if name == 'decoder.weight':
            continue
        sum_var[size:size + layer.view(-1).shape[0]] = (layer.grad).view(-1)
        size += layer.view(-1).shape[0]

    return sum_var


def normalize_data(args, train_df, test_df):
    all_data = pd.concat([train_df, test_df], axis=0)
    if args.dataset != 'adult':
        minmax_scaler = MinMaxScaler()
        for col in args.feature:
            all_data[col] = minmax_scaler.fit_transform(all_data[col].values)
    train_df = all_data[:train_df.shape[0]]
    test_df = all_data[train_df.shape[0]:]
    return train_df, test_df
