import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score
from collections import defaultdict
from torch.autograd import Variable

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        if self.run_mode == 'func' and epoch < self.skip_ep:
            return
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score

class ReduceOnPlatau:
    def __init__(self, mode="max", delta=1e-4, verbose=False, args=None, min_lr=5e-5):
        self.patience = args.lr_patience
        self.counter = 0
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.args = args
        self.min_lr = min_lr
        self.step = args.lr_step
        self.best_score = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                if self.args.lr - self.step < self.min_lr:
                    self.args.lr = self.min_lr
                else:
                    self.args.lr -= self.step
                print("Reduce learning rate to {}".format(self.args.lr))
                self.counter = 0
        else:
            self.best_score = score
            self.counter = 0
        return self.args


def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0

    for bi, d in enumerate(dataloader):
        features, target, _ = d

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        # num_data_point += features.size(dim=0)
        optimizer.zero_grad()

        output = model(features)
        output = torch.squeeze(output)

        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return train_loss, train_outputs, train_targets


def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss = 0
    # num_data_point = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, target, _ = d

            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            # num_data_point += features.size(dim=0)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            loss += loss_eval.item()
            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss, fin_outputs, fin_targets

def train_smooth_classifier(dataloader, model, model_, criterion, optimizer, device, scheduler, alpha):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0
    params_ = model_.state_dict()

    for bi, d in enumerate(dataloader):
        features, target, _ = d

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        # num_data_point += features.size(dim=0)
        optimizer.zero_grad()
        output = model(features)
        output = torch.squeeze(output)
        l2_norm = 0.0
        for p in model.named_parameters():
            l2_norm += torch.norm(p[1] - params_[p[0]], p=2)
        # print(l2_norm.item())
        loss = criterion(output, target) + alpha * l2_norm / 2
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return train_loss, train_outputs, train_targets

def pgd_attack(model, group_loader, y, epsilon, clip_max, clip_min, num_steps, step_size, device):
    group_acc = {}
    m = torch.distributions.Uniform(torch.tensor([-epsilon]), torch.tensor([epsilon]))
    for key, loader in group_loader.items():
        print('=' * 10 + 'attacking {}'.format(key) + '=' * 10)
        output_arr = []
        target_arr = []
        for bi, d in enumerate(loader):
            features, targets, _ = d
            X_random = m.rsample(sample_shape=features.size())
            X_random = torch.squeeze(X_random, dim=-1)
            # print(features.size(), X_random.size())
            X_pgd = torch.clip(input=features + X_random, min=0, max=1.0)
            X_pgd = X_pgd.to(device, dtype=torch.float)
            features = features.to(device, dtype=torch.float)
            X_pgd.requires_grad = True
            target = torch.ones(X_pgd.size(0)) - targets
            target = target.to(device, dtype=torch.float)
            for i in range(num_steps):
                pred = model(X_pgd)
                loss = torch.nn.BCELoss()(torch.squeeze(pred, dim=-1), target)
                loss.backward()

                eta = step_size * X_pgd.grad.data.sign()

                X_pgd = X_pgd + eta
                eta = torch.clamp(X_pgd.data - features.data, -epsilon, epsilon)

                X_pgd = features.data + eta
                X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
                X_pgd = X_pgd.detach()
                X_pgd.requires_grad_()
                X_pgd.retain_grad()
            outputs = model(X_pgd)
            outputs = torch.squeeze(outputs, dim=-1)
            outputs = outputs.cpu().detach().numpy()
            output_arr.extend(outputs)
            target_arr.extend(targets.cpu().detach().numpy().astype(int).tolist())
        output_arr = np.round(np.array(output_arr))
        acc = accuracy_score(y_true=target_arr, y_pred=output_arr)
        group_acc[key] = 1 - acc
    return group_acc

def train_pgd(args, dataloader, model, criterion, optimizer, device, scheduler):
    model.to(device)
    model.train()

    train_targets = []
    train_outputs = []
    train_loss = 0

    for bi, d in enumerate(dataloader):
        features, target, _ = d

        features = features.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)
        x_adv = projected_gradient_descent(model, features, target, criterion,
                                           num_steps=args.num_steps_pgd, step_size=args.step_size_pgd,
                                           eps=args.eps_pgd, eps_norm='inf',
                                           step_norm='inf')
        # num_data_point += features.size(dim=0)
        optimizer.zero_grad()

        output = model(x_adv)
        output = torch.squeeze(output)

        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        output = output.cpu().detach().numpy()

        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(output)

    return train_loss, train_outputs, train_targets

def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, step_norm, eps, eps_norm,
                               clamp=(0, 1), y_target=None):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    # print(x_adv.max(), x_adv.min())
    targeted = y_target is not None
    num_channels = x.shape[1]

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        prediction = torch.squeeze(prediction)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            if step_norm == 'inf':
                gradients = _x_adv.grad.sign() * step_size
            else:
                # Note .view() assumes batched image data as 4D tensor
                gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1) \
                    .norm(step_norm, dim=-1) \
                    .view(-1, num_channels, 1, 1)

            if targeted:
                # Targeted: Gradient descent with on the loss of the (incorrect) target label
                # w.r.t. the image data
                x_adv -= gradients
            else:
                # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                # the model parameters
                x_adv += gradients

        # Project back into l_norm ball and correct range
        if eps_norm == 'inf':
            # Workaround as PyTorch doesn't have elementwise clip
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        else:
            delta = x_adv - x

            # Assume x and x_adv are batched tensors where the first dimension is
            # a batch dimension
            mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps

            scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
            scaling_factor[mask] = eps

            # .view() assumes batched images as a 4D Tensor
            delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            x_adv = x + delta

        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()