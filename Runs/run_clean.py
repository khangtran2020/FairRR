from Models.train_eval import *
from Utils.plottings import *
from Utils.metrics import *
from tqdm import tqdm

def run_clean(args, tr_info, va_info, te_info, name, device):
    model_name = '{}.pt'.format(name)
    tr_loader = tr_info
    va_loader = va_info
    te_loader, te_mal_loader, te_fem_loader = te_info

    # Defining Model for specific fold
    model = init_model(args=args)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = init_history()

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        tr_loss, tr_out, tr_tar = train_fn(tr_loader, model, criterion, optimizer, device, scheduler=None)
        va_loss, va_out, va_tar = eval_fn(va_loader, model, criterion, device)
        te_loss, te_out, te_tar = eval_fn(te_loader, model, criterion, device)
        _, _, demo_p = demo_parity(male_loader=te_mal_loader, female_loader=te_fem_loader, model=model, device=device)
        _, _, equal_opp, equal_odd = eq_opp_odd(male_loader=te_mal_loader, female_loader=te_fem_loader,
                                                model=model, device=device)
        tr_acc = performance_eval(args, tr_tar, tr_out)
        te_acc = performance_eval(args, te_tar, te_out)
        va_acc = performance_eval(args, va_tar, va_out)

        tk0.set_postfix(loss=tr_loss, acc=tr_acc, val_loss=va_loss, val_acc=va_acc, te_acc=te_acc,
                        demo=demo_p, eqopp=equal_opp, eqodd=equal_odd)

        history['tr_loss'].append(tr_loss)
        history['tr_acc'].append(tr_acc)
        history['va_loss'].append(va_loss)
        history['va_acc'].append(va_acc)
        history['demo_parity'].append(demo_p)
        history['equal_opp'].append(equal_opp)
        history['equal_odd'].append(equal_odd)
        history['te_loss'].append(te_loss)
        history['te_acc'].append(te_acc)

        es(epoch=epoch, epoch_score=va_acc, model=model, model_path=args.save_path + model_name)

    model.load_state_dict(torch.load(args.save_path + model_name))
    te_loss, te_out, te_tar = eval_fn(te_loader, model, criterion, device)
    te_acc = performance_eval(args, te_tar, te_out)
    _, _, demo_p = demo_parity(male_loader=te_mal_loader, female_loader=te_fem_loader, model=model, device=device)
    _, _, equal_opp, equal_odd = eq_opp_odd(male_loader=te_mal_loader, female_loader=te_fem_loader,
                                            model=model, device=device)
    history['best_test'] = te_acc
    history['best_demo_parity'] = demo_p
    history['best_equal_opp'] = equal_opp
    history['best_equal_odd'] = equal_odd
    print("=" * 100)
    print(f"Best result: acc {te_acc}, demo {demo_p}, opp {equal_opp}, odd {equal_odd}")
    print("=" * 100)
    save_res(args=args, dct=history, name=name)