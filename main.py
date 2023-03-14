from config import parse_args
from Data.read_data import *
import datetime
from Runs.run_clean import *
from Utils.utils import *
import warnings
import torch

warnings.filterwarnings("ignore")


def run(args, current_time, device):
    # read data
    print('Using noise scale: {}, clip: {}'.format(args.ns, args.clip))

    if args.dataset == 'adult':
        train_df, test_df, feature_cols, label, z = read_adult(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
    elif args.dataset == 'bank':
        train_df, test_df, feature_cols, label, z = read_bank(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        # print(train_df.max(axis=0), train_df.min(axis=0))
    elif args.dataset == 'abalone':
        train_df, test_df, feature_cols, label, z = read_abalone(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        # print(train_df.max(axis=0), train_df.min(axis=0))
    elif args.dataset == 'utk':
        train_df, test_df, feature_cols, label, z = read_utk(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        # print(train_df.max(axis=0), train_df.min(axis=0))
    print(train_df[args.target].value_counts())
    print(test_df[args.target].value_counts())
    print(test_df[args.z].value_counts())
    print(f'Running with dataset {args.dataset}: {len(train_df)} train, {len(test_df)} test')

    if args.debug:
        run_fair_eval(fold=0, train_df=train_df, test_df=test_df, args=args,
                 device=device,
                 current_time=current_time)
    else:
        for i in range(args.folds):
            run_fair_eval(fold=i, train_df=train_df, test_df=test_df, args=args,
                          device=device,
                          current_time=current_time)

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)