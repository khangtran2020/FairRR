import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.special import erf


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


def eq_opp_odd(male_loader, female_loader, model, device):
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
    return male_tpr, female_tpr, np.abs(male_tpr - female_tpr), \
        0.5 * np.abs(male_tpr - female_tpr) + 0.5 * np.abs(male_fpr - female_fpr)