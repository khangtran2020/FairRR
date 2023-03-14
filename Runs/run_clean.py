from Models.train_eval import *
from Utils.plottings import *
from Utils.metrics import *
from tqdm import tqdm

def run_fair_eval(fold, train_df, test_df, args, device, current_time):
    name = get_name(args=args, current_date=current_time, fold=fold)
    model_name = '{}.pt'.format(name)
    train_loader, train_male_loader, train_female_loader, valid_male_loader, valid_female_loader, valid_loader, test_loader = init_data(
        args=args, fold=fold, train_df=train_df, test_df=test_df)

    # Defining Model for specific fold
    model = init_model(args=args)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Defining LR SCheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.1, patience=15, verbose=True,
                                                           threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-4, eps=1e-08)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'demo_parity': [],
        'acc_parity': [],
        'equal_odd': [],
        'equal_opp': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_acc_parity': 0,
        'best_equal_odd': 0,
        'best_equal_opp': 0,
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        # print('Epoch {}:'.format(epoch), male_loss, female_loss)
        train_loss, train_out, train_targets = train_fn(train_loader, model, criterion, optimizer, device,
                                                        scheduler=None)
        val_loss, outputs, targets = eval_fn(valid_loader, model, criterion, device)
        test_loss, test_outputs, test_targets = eval_fn(test_loader, model, criterion, device)
        _, _, demo_p = demo_parity(male_loader=valid_male_loader, female_loader=valid_female_loader,
                                   model=model, device=device)
        _, _, equal_odd = equality_of_odd(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=model, device=device)
        _, _, equal_opp = equality_of_opp(male_loader=valid_male_loader,
                                          female_loader=valid_female_loader, model=model, device=device)
        acc_par = acc_parity(male_loader=valid_male_loader,
                             female_loader=valid_female_loader, model=model, device=device)
        train_acc = performace_eval(args, train_targets, train_out)
        test_acc = performace_eval(args, test_targets, test_outputs)
        acc_score = performace_eval(args, targets, outputs)

        scheduler.step(acc_score)
        # scheduler_male.step(male_acc_score)
        # scheduler_female.step(female_acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_ACC_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_ACC_SCORE=acc_score)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(acc_score)
        history['demo_parity'].append(demo_p)
        history['acc_parity'].append(acc_par)
        history['equal_odd'].append(equal_odd)
        history['equal_opp'].append(equal_opp)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)

        es(epoch=epoch, epoch_score=acc_score, model=model, model_path=args.save_path + model_name)
        if es.early_stop:
            print('Maximum Patience {} Reached , Early Stopping'.format(args.patience))
            break

    # print_history_fair(fold, history, epoch + 1, args, current_time)
    save_res(fold=fold, args=args, dct=history, current_time=current_time)