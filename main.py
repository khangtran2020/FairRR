import warnings
import torch
import datetime
from Runs.run_clean import *
from Utils.utils import *
from config import parse_args
from Data.read_data import *

warnings.filterwarnings("ignore")


def run(args, current_time, device):
    # read data
    if args.dataset == 'adult': read_dat = read_adult
    elif args.dataset == 'abalone': read_dat = read_abalone
    elif args.dataset == 'bank': read_dat = read_bank
    else: read_dat = read_utk
    train_df, test_df, male_tr_df, female_tr_df, male_te_df, female_te_df, feature_cols, label, z = read_dat(args)
    args.feature = feature_cols
    args.target = label
    args.z = z
    args.input_dim = len(feature_cols)
    args.output_dim = 1

    print(f'Running with dataset {args.dataset}: {len(train_df)} train, {len(test_df)} test')
    print(f'{len(male_tr_df)} male, {len(female_tr_df)} female, {len(feature_cols)} features')
    print('Train label counts:', train_df[args.target].value_counts())
    print('Test label counts:', test_df[args.target].value_counts())

    train = (male_tr_df, female_tr_df)
    test = (test_df, male_te_df, female_te_df)
    tr_info, va_info, te_info = init_data(args=args, fold=0, train=train, test=test)
    name = get_name(args=args, current_date=current_time, fold=0)

    run_dict = {
        'clean': run_clean,
    }

    run = run_dict[args.mode]
    run(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, name=name, device=device)

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)