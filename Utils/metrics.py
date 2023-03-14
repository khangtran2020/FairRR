import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score
from copy import deepcopy

def demo_parity(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    female_outputs = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, _, _ = d
            features = features.to(device, dtype=torch.float)
            outputs = model(features)
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)

        for bi, d in enumerate(female_loader):
            features, _, _ = d
            features = features.to(device, dtype=torch.float)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)

    male_outputs = np.round(np.array(male_outputs))
    female_outputs = np.round(np.array(female_outputs))
    prob_male = np.sum(male_outputs) / len(male_outputs)
    prob_female = np.sum(female_outputs) / len(female_outputs)
    return prob_male, prob_female, np.abs(prob_male - prob_female)

def equality_of_odd(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    male_target = []
    female_outputs = []
    female_target = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)
            male_target.extend(target.cpu().detach().numpy().astype(int).tolist())

        for bi, d in enumerate(female_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)
            female_target.extend(target.cpu().detach().numpy().astype(int).tolist())

    male_outputs = np.round(np.array(male_outputs))
    tn, fp, fn, tp = confusion_matrix(male_target, male_outputs).ravel()
    male_tpr = tp / (tp + fn)
    male_fpr = fp / (fp + tn)
    female_outputs = np.round(np.array(female_outputs))
    tn, fp, fn, tp = confusion_matrix(female_target, female_outputs).ravel()
    female_tpr = tp / (tp + fn)
    female_fpr = fp / (fp + tn)
    return male_tpr, female_tpr, max(np.abs(male_tpr - female_tpr), np.abs(male_fpr - female_fpr))

def equality_of_opp(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    male_target = []
    female_outputs = []
    female_target = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)
            male_target.extend(target.cpu().detach().numpy().astype(int).tolist())

        for bi, d in enumerate(female_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)
            female_target.extend(target.cpu().detach().numpy().astype(int).tolist())

    male_outputs = np.round(np.array(male_outputs))
    tn, fp, fn, tp = confusion_matrix(male_target, male_outputs).ravel()
    male_tpr = tp / (tp + fn)
    female_outputs = np.round(np.array(female_outputs))
    tn, fp, fn, tp = confusion_matrix(female_target, female_outputs).ravel()
    female_tpr = tp / (tp + fn)
    return male_tpr, female_tpr, np.abs(male_tpr - female_tpr)

def disperate_impact(male_loader, female_loader, global_model, male_model, female_model, num_male, num_female, device):
    global_model.to(device)
    male_model.to(device)
    female_model.to(device)

    glob_male_out = []
    glob_female_out = []
    male_outputs = []
    female_outputs = []

    global_model.eval()
    male_model.eval()
    female_model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, _, _ = d

            features = features.to(device, dtype=torch.float)

            glob_out = global_model(features)
            male_out = male_model(features)

            glob_out = torch.squeeze(glob_out, dim=-1)
            glob_out = glob_out.cpu().detach().numpy()
            glob_male_out.extend(glob_out)

            male_out = torch.squeeze(male_out, dim=-1)
            male_out = male_out.cpu().detach().numpy()
            male_outputs.extend(male_out)

        for bi, d in enumerate(female_loader):
            features, _, _ = d

            features = features.to(device, dtype=torch.float)

            glob_out = global_model(features)
            female_out = female_model(features)

            glob_out = torch.squeeze(glob_out, dim=-1)
            glob_out = glob_out.cpu().detach().numpy()
            glob_female_out.extend(glob_out)

            female_out = torch.squeeze(female_out, dim=-1)
            female_out = female_out.cpu().detach().numpy()
            female_outputs.extend(female_out)

    male_outputs = np.array(male_outputs)
    glob_male_out = np.array(glob_male_out)
    female_outputs = np.array(female_outputs)
    glob_female_out = np.array(glob_female_out)

    male_norm = np.sum(np.abs(male_outputs - glob_male_out))
    female_norm = np.sum(np.abs(female_outputs - glob_female_out))
    return male_norm / num_male, female_norm / num_female

def disperate_impact_smooth(male_loader, female_loader, global_model, male_model, female_model, num_male, num_female, device, num_draws):
    global_model.to(device)
    male_model.to(device)
    female_model.to(device)

    glob_male_out = []
    glob_female_out = []
    male_outputs = []
    female_outputs = []

    global_model.eval()
    male_model.eval()
    female_model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, _, _ = d

            features = features.to(device, dtype=torch.float)

            glob_out = 0.0
            for i in range(num_draws):
                new_model = deepcopy(global_model).to(device)
                state_dict = new_model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key] + torch.normal(mean=0.0, std=0.1, size=state_dict[key].size(),
                                                                     requires_grad=False).to(device)
                new_model.load_state_dict(state_dict)
                glob_out = glob_out + new_model(features)
            glob_out = glob_out / num_draws

            male_out = 0.0
            for i in range(num_draws):
                new_model = deepcopy(male_model).to(device)
                state_dict = new_model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key] + torch.normal(mean=0.0, std=0.1, size=state_dict[key].size(),
                                                                     requires_grad=False).to(device)
                new_model.load_state_dict(state_dict)
                male_out = male_out + new_model(features)
            male_out = male_out / num_draws

            glob_out = torch.squeeze(glob_out, dim=-1)
            glob_out = glob_out.cpu().detach().numpy()
            glob_male_out.extend(glob_out)

            male_out = torch.squeeze(male_out, dim=-1)
            male_out = male_out.cpu().detach().numpy()
            male_outputs.extend(male_out)

        for bi, d in enumerate(female_loader):
            features, _, _ = d

            features = features.to(device, dtype=torch.float)

            glob_out = 0.0
            for i in range(num_draws):
                new_model = deepcopy(global_model).to(device)
                state_dict = new_model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key] + torch.normal(mean=0.0, std=0.1, size=state_dict[key].size(),
                                                                     requires_grad=False).to(device)
                new_model.load_state_dict(state_dict)
                glob_out = glob_out + new_model(features)
            glob_out = glob_out / num_draws

            female_out = 0.0
            for i in range(num_draws):
                new_model = deepcopy(female_model).to(device)
                state_dict = new_model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key] + torch.normal(mean=0.0, std=0.1, size=state_dict[key].size(),
                                                                     requires_grad=False).to(device)
                new_model.load_state_dict(state_dict)
                female_out = female_out + new_model(features)
            female_out = female_out / num_draws

            glob_out = torch.squeeze(glob_out, dim=-1)
            glob_out = glob_out.cpu().detach().numpy()
            glob_female_out.extend(glob_out)

            female_out = torch.squeeze(female_out, dim=-1)
            female_out = female_out.cpu().detach().numpy()
            female_outputs.extend(female_out)

    male_outputs = np.array(male_outputs)
    glob_male_out = np.array(glob_male_out)
    female_outputs = np.array(female_outputs)
    glob_female_out = np.array(glob_female_out)

    male_norm = np.sum(np.abs(male_outputs - glob_male_out))
    female_norm = np.sum(np.abs(female_outputs - glob_female_out))
    return male_norm / num_male, female_norm / num_female

def acc_parity(male_loader, female_loader, model, device):
    model.to(device)
    male_outputs = []
    male_target = []
    female_outputs = []
    female_target = []

    model.eval()
    with torch.no_grad():

        for bi, d in enumerate(male_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            male_outputs.extend(outputs)
            male_target.extend(target.cpu().detach().numpy().astype(int).tolist())

        for bi, d in enumerate(female_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            female_outputs.extend(outputs)
            female_target.extend(target.cpu().detach().numpy().astype(int).tolist())

    male_outputs = np.round(np.array(male_outputs))
    male_acc = accuracy_score(y_true=male_target, y_pred=male_outputs)
    female_outputs = np.round(np.array(female_outputs))
    female_acc = accuracy_score(y_true=female_target, y_pred=female_outputs)
    return np.abs(male_acc - female_acc)

def group_acc_parity(group_loader, model, device):
    model.to(device)
    group_acc = {}

    model.eval()
    with torch.no_grad():
        for key, loader in group_loader.items():
            output_arr = []
            target_arr = []
            for bi, d in enumerate(loader):
                features, target, _ = d

                features = features.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                outputs = model(features)
                # if outputs.size(dim=0) > 1:
                outputs = torch.squeeze(outputs, dim=-1)
                outputs = outputs.cpu().detach().numpy()
                output_arr.extend(outputs)
                target_arr.extend(target.cpu().detach().numpy().astype(int).tolist())
            output_arr = np.round(np.array(output_arr))
            acc = accuracy_score(y_true=target_arr, y_pred=output_arr)
            group_acc[key] = acc
    return group_acc

def group_disperate_impact(group_loader, global_model, group_model, group_num, device):
    global_model.to(device)
    global_model.eval()
    for key, model in group_model.items():
        model.to(device)
        model.eval()

    group_norm = {}
    with torch.no_grad():
        for key, loader in group_loader.items():
            model = group_model[key]
            glob_out_arr = []
            group_out_arr = []
            for bi, d in enumerate(loader):
                features, _, _ = d

                features = features.to(device, dtype=torch.float)

                glob_out = global_model(features)
                group_out = model(features)

                glob_out = torch.squeeze(glob_out, dim=-1)
                glob_out = glob_out.cpu().detach().numpy()
                glob_out_arr.extend(glob_out)

                group_out = torch.squeeze(group_out, dim=-1)
                group_out = group_out.cpu().detach().numpy()
                group_out_arr.extend(group_out)

            group_out_arr = np.array(group_out_arr)
            glob_out_arr = np.array(glob_out_arr)
            norm = np.sum(np.abs(group_out_arr - glob_out_arr))/group_num[key]
            group_norm[key] = norm
    return group_norm

def performace_eval(args, y_true, y_pred):
    if args.performance_metric == 'acc':
        return accuracy_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'f1':
        return f1_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif args.performance_metric == 'pre':
        return precision_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))