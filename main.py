import warnings
import datetime
from Runs.run_clean import *
from Runs.run_sklearn import *
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
    train_df, test_df, feature_cols, label, z = read_dat(args)
    args.feature = feature_cols
    args.target = label
    args.z = z
    args.input_dim = len(feature_cols)
    args.output_dim = 1

    print(f'Running with dataset {args.dataset}: {len(train_df)} train, {len(test_df)} test, {len(feature_cols)} features')
    print('Train label counts:', train_df[args.target].value_counts())
    print('Test label counts:', test_df[args.target].value_counts())

    train = train_df
    test = test_df
    if args.mode == 'clean': tr_info, va_info, te_info = init_data(args=args, fold=0, train=train, test=test)
    else:
        df_train = train_df.copy()

        print('=' * 10 + ' Applying FairRR ' + '=' * 10)
        # args, df, mode = 'train', ignore_co = None, random_co = None, eps_dict = None

        tr_df, cols, epsilon, mean_dct = fairRR(args=args, df=df_train, mode='train')
        for col in args.feature:
            print(col, df_train[col].std(), tr_df[col].std())
        ignore_col, random_col = cols
        te_df = fairRR(args=args, df=test_df, mode='test', ignore_co=ignore_col, random_co=random_col,
                       eps_dict=epsilon, mean_dct=mean_dct)
        print('=' * 10 + ' Done FairRR ' + '=' * 10)

        tr_info = tr_df
        va_info = None
        te_info = te_df

    name = get_name(args=args, current_date=current_time, fold=0)

    run_dict = {
        'clean': run_clean,
        'sklearn': run_sklearn
    }

    run = run_dict[args.mode]
    run(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, name=name, device=device)

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)