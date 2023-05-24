import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from Data.datasets import Data
from sklearn.model_selection import train_test_split


def read_adult(args):
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
              'income']
    label_dict = {
        ' <=50K': '<=50K',
        ' >50K': '>50K',
        ' <=50K.': '<=50K',
        ' >50K.': '>50K'
    }
    train_df = pd.read_csv('Data/Adult/adult.data', header=None)
    test_df = pd.read_csv('Data/Adult/adult.test', skiprows=1, header=None)
    all_data = pd.concat([train_df, test_df], axis=0)
    all_data.columns = header

    def hour_per_week(x):
        if x <= 19:
            return '0'
        elif (x > 19) & (x <= 29):
            return '1'
        elif (x > 29) & (x <= 39):
            return '2'
        elif x > 39:
            return '3'

    def age(x):
        if x <= 24:
            return '0'
        elif (x > 24) & (x <= 34):
            return '1'
        elif (x > 34) & (x <= 44):
            return '2'
        elif (x > 44) & (x <= 54):
            return '3'
        elif (x > 54) & (x <= 64):
            return '4'
        else:
            return '5'

    def country(x):
        if x == ' United-States':
            return 0
        else:
            return 1

    all_data['hours-per-week'] = all_data['hours-per-week'].map(lambda x: hour_per_week(x))
    all_data['age'] = all_data['age'].map(lambda x: age(x))
    all_data['native-country'] = all_data['native-country'].map(lambda x: country(x))
    all_data = all_data.drop(
        ['fnlwgt', 'education-num', 'marital-status', 'occupation', 'relationship', 'capital-gain', 'capital-loss'],
        axis=1)
    temp = pd.get_dummies(all_data['age'], prefix='age')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('age', axis=1)
    temp = pd.get_dummies(all_data['workclass'], prefix='workclass')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('workclass', axis=1)
    temp = pd.get_dummies(all_data['education'], prefix='education')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('education', axis=1)
    temp = pd.get_dummies(all_data['race'], prefix='race')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('race', axis=1)
    temp = pd.get_dummies(all_data['hours-per-week'], prefix='hour')
    all_data = pd.concat([all_data, temp], axis=1)
    all_data = all_data.drop('hours-per-week', axis=1)
    all_data['income'] = all_data['income'].map(label_dict)
    lb = LabelEncoder()
    all_data['sex'] = lb.fit_transform(all_data['sex'].values)
    lb = LabelEncoder()
    all_data['income'] = lb.fit_transform(all_data['income'].values)
    feature_cols = list(all_data.columns)
    feature_cols.remove('income')
    feature_cols.remove('sex')
    label = 'income'
    z = 'sex'
    for col in feature_cols: all_data[col] = (all_data[col] - all_data[col].mean())/(all_data[col].std() + 1e-12)
    print(all_data.head())
    train_df = all_data[:train_df.shape[0]].reset_index(drop=True)
    test_df = all_data[train_df.shape[0]:].reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
    train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z

def read_bank(args):
    # 3305
    df = pd.read_csv('Data/Bank/formated_bank.csv')
    feature_cols = list(df.columns)
    feature_cols.remove('y')
    feature_cols.remove('z')
    feature_cols.remove('label')
    feature_cols.remove('is_train')
    feature_cols.remove('intercept')
    label = 'y'
    z = 'z'
    for col in feature_cols: df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-12)
    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
    train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z

def read_abalone(args):
    # 1436
    df = pd.read_csv('Data/Abalone/formated_abalone.csv')
    # print(df.head())
    feature_cols = list(df.columns)
    feature_cols.remove('y')
    feature_cols.remove('label')
    feature_cols.remove('z')
    feature_cols.remove('is_train')
    label = 'y'
    z = 'z'
    for col in feature_cols: df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-12)
    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
    train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z

def read_utk(args):
    df = pd.read_csv('Data/UTK/feat.zip', compression='zip')
    feature_cols = list(df.columns)
    feature_cols.remove('ethnicity')
    feature_cols.remove('gender')
    feature_cols.remove('is_train')
    label = 'gender'
    z = 'ethnicity'
    for col in feature_cols: df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-12)
    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[df['is_train'] == 0].reset_index(drop=True)
    male_tr_df = train_df[train_df[z] == 1].copy().reset_index(drop=True)
    female_tr_df = train_df[train_df[z] == 0].copy().reset_index(drop=True)
    male_te_df = test_df[test_df[z] == 1].copy().reset_index(drop=True)
    female_te_df = test_df[test_df[z] == 0].copy().reset_index(drop=True)
    fold_separation(male_tr_df, args.folds, feature_cols, label)
    fold_separation(female_tr_df, args.folds, feature_cols, label)
    if args.submode == 'ratio':
        male_tr_df, female_tr_df = choose_data(args=args, df_0=male_tr_df, df_1=female_tr_df)
    train_df = pd.concat([male_tr_df, female_tr_df], axis=0).sample(frac=1).reset_index(drop=True)
    return train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z

def fold_separation(train_df, folds, feat_cols, label):
    skf = StratifiedKFold(n_splits=folds)
    train_df['fold'] = np.zeros(train_df.shape[0])
    for i, (idxT, idxV) in enumerate(skf.split(train_df[feat_cols], train_df[label])):
        train_df.at[idxV, 'fold'] = i

def minmax_scale(df, cols):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for col in cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df

def choose_data(args, df_0, df_1):
    # print(len(df_0),len(df_1))
    if len(df_0) > len(df_1):
        df = df_1.copy()
        df_1 = df_0.copy()
        df_0 = df.copy()
        del (df)

    df_0 = df_0.reset_index(drop=True)
    df_1 = df_1.reset_index(drop=True)

    num_pt = int(args.ratio * len(df_0))
    if num_pt > len(df_1):
        print('Can not achieve that rate between group0 and group1')
        args.can_ratio = False
        return df_0.reset_index(drop=True), df_1.reset_index(drop=True)
    else:
        idx = np.random.choice(np.arange(len(df_1)), size=num_pt, replace=False)
        df_1 = df_1.iloc[idx, :].copy()
        return df_0.reset_index(drop=True), df_1.reset_index(drop=True)