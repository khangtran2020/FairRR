import argparse

def add_general_group(group):
    group.add_argument("--save_path", type=str, default="results/models/", help="dir path for saving model file")
    group.add_argument("--res_path", type=str, default="results/dict/", help="dir path for output file")
    group.add_argument("--plot_path", type=str, default="results/plot/", help="dir path for plots file")
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--mode", type=str, default='clean', help="Mode of running ['clean']")
    group.add_argument("--submode", type=str, default='fairrr', help="Mode of running ['clean', 'fairrr']")
    group.add_argument("--performance_metric", type=str, default='acc', help="['acc', 'f1', 'auc', 'pre']")


def add_data_group(group):
    group.add_argument('--dataset', type=str, default='adult', help="name of dataset")
    group.add_argument('--folds', type=int, default=5, help='number of folds for cross-validation')
    group.add_argument('--ratio', type=float, default=0.0, help="ratio group0/group1")
    group.add_argument('--max_val', type=float, default=1.0, help="maximum value for dataset")
    group.add_argument('--min_val', type=float, default=0.0, help="minimum value for dataset")
    group.add_argument('--num_class', type=int, default=2, help="label space")
    group.add_argument('--n_comp', type=int, default=20, help="pca dimension")
    group.add_argument('--n_bin', type=int, default=5, help="max # of bucket")

def add_model_group(group):
    group.add_argument("--model_type", type=str, default='NN', help="Model type")
    group.add_argument("--lr", type=float, default=0.02, help="learning rate")
    group.add_argument('--batch_size', type=int, default=512, help="batch size for training process")
    group.add_argument('--n_hid', type=int, default=16, help='number hidden embedding dim')
    group.add_argument('--n_layer', type=int, default=2, help='number of layer')
    group.add_argument("--alpha", type=float, default=0.05)
    group.add_argument("--optimizer", type=str, default='adam')
    group.add_argument("--epochs", type=int, default=100, help='training step')
    group.add_argument("--patience", type=int, default=8, help='early stopping')
    group.add_argument("--debug", type=bool, default=True)
    group.add_argument("--ns", type=float, default=1.0, help='noise scale for dp')
    group.add_argument("--num_workers", type=int, default=0)
    group.add_argument("--dropout", type=float, default=0.2, help='dropout')

def add_fairrr_group(group):
    group.add_argument("--num_bit", type=int, default=10, help='clipping gradient bound')
    group.add_argument("--num_int", type=int, default=1, help="targeted epsilon")
    group.add_argument("--tar_eps", type=float, default=1.0, help="targeted epsilon")

def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    general_group = parser.add_argument_group(title="General configuration")
    fairrr_group = parser.add_argument_group(title="FairRR configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_general_group(general_group)
    add_fairrr_group(fairrr_group)

    return parser.parse_args()
