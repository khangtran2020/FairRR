import matplotlib.pyplot as plt
import numpy as np
from Utils.utils import *

def print_history_fair(fold, history, num_epochs, args, current_time):
    # plt.figure(figsize=(15,5))
    name = get_name(args=args, current_date=current_time, fold=fold)
    save_name = args.plot_path + '{}.jpg'.format(name)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    axs[0].plot(
        np.arange(num_epochs),
        history['train_history_acc'],
        '-o',
        label='Train ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_history_acc'],
        '--o',
        label='Val ACC',
        color='#1f77b4'
    )

    # axs[0].plot(
    #     np.arange(num_epochs),
    #     history['test_history_acc'],
    #     ':*',
    #     label='Test ACC',
    #     color='deeppink'
    # )

    # x = np.argmax(history['val_history_acc'])
    # y = np.max(history['val_history_acc'])
    #
    # xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    # ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]
    #
    # # axs[0].scatter(x, y, s=200, color='#1f77b4')
    # #
    # # axs[0].text(
    # #     x - 0.03 * xdist,
    # #     y - 0.13 * ydist,
    # #     'max acc\n%.2f' % y,
    # #     size=14
    # # )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}', size=18)
    axs[0].legend()

    # plt2 = plt.gca().twinx()

    axs[1].plot(
        np.arange(num_epochs),
        history['demo_parity'],
        '-o',
        label='Demographic parity',
        color='#2ca02c'
    )

    axs[1].set_ylabel('Demographic Parity', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}', size=18)

    axs[1].legend()

    axs[2].plot(
        np.arange(num_epochs),
        history['equal_opp'],
        '-o',
        label='Equality of Opportunity',
    )
    axs[2].set_ylabel(r'Equality of Opportunity', size=14)
    axs[2].set_xlabel('Epochs', size=14)
    axs[2].set_title(f'FOLD {fold + 1}', size=18)
    plt.savefig(save_name)

def theorem_3_4(history, args):

    print(f"SEED #{args.seed}, EPS {args.tar_eps}, DATASET {args.dataset}")
    
    print("DP : ", history['demo_parity'])
    yy = history['demo_parity'][-1]
    xx = (np.exp(args.tar_eps - 1) / np.exp(args.tar_eps + 1)) ** 2
    print(f"DP : {yy} <= {xx}")
    
    print("EQUAL OPP : ", history['equal_opp'])
    yy = history['equal_opp'][-1]
    xx = (np.exp(args.tar_eps - 1) / np.exp(args.tar_eps + 1)) ** 2
    print(f"Equal Opp : {yy} <= {xx}")

    print("EQUAL ODD : ", history['equal_odd'])
    yy = history['equal_odd'][-1]
    xx = (np.exp(args.tar_eps - 1) / np.exp(args.tar_eps + 1)) ** 2
    print(f"Equal Odd : {yy} <= {xx}")