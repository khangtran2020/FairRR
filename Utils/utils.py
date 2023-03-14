import pickle
import random
import os
import numpy as np
import torch
from sklearn.metrics import log_loss
from Models.models import *
from torch.utils.data import DataLoader
from Data.datasets import Data
import pandas as pd


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

def init_data(args, fold, train_df, test_df, male_df, female_df):
    if args.mode == 'clean':
        df_train = train_df[train_df.fold != fold]
        df_valid = train_df[train_df.fold == fold]

        # Defining DataSet
        train_dataset = Data(
            X=df_train[args.feature].values,
            y=df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            X=df_valid[args.feature].values,
            y=df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            X=test_df[args.feature].values,
            y=test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=4
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, valid_loader, test_loader
    elif args.mode == 'fairrr':
        df_train = train_df[train_df.fold != fold]
        df_valid = train_df[train_df.fold == fold]
        x_train = FairRR(df_train[args.feature].values)
        # Defining DataSet
        train_dataset = Data(
            X=x_train,
            y=df_train[args.target].values,
            ismale=df_train[args.z].values
        )

        valid_dataset = Data(
            X=df_valid[args.feature].values,
            y=df_valid[args.target].values,
            ismale=df_valid[args.z].values
        )

        test_dataset = Data(
            X=test_df[args.feature].values,
            y=test_df[args.target].values,
            ismale=test_df[args.z].values
        )

        # Defining DataLoader with BalanceClass Sampler
        sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            num_workers=0
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
        return train_loader, valid_loader, test_loader

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
