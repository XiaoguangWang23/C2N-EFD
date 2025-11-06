import random
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from models.layers import EFD
from utils.loss_utils import BalLossL2
from utils.metrics_utils import evaluate
import os
import random
import numpy as np




def setup_seed(seed=42):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def data_split(dataset_name):
    ChinaFSD_dicts = {0: {'train': [2014, 2016], 'valid': [2017], 'test': [2018]},
                      1: {'train': [2014, 2017], 'valid': [2018], 'test': [2019]},
                      2: {'train': [2014, 2018], 'valid': [2019], 'test': [2020]},
                      3: {'train': [2014, 2019], 'valid': [2020], 'test': [2021]},
                      4: {'train': [2014, 2020], 'valid': [2021], 'test': [2022]}
    }

    USFSD_dicts = {0: {'train': [1991, 1999], 'valid': [2000, 2001], 'test': [2003]},
                   1: {'train': [1991, 2000], 'valid': [2001, 2002], 'test': [2004]},
                   2: {'train': [1991, 2001], 'valid': [2002, 2003], 'test': [2005]},
                   3: {'train': [1991, 2002], 'valid': [2003, 2004], 'test': [2006]},
                   4: {'train': [1991, 2003], 'valid': [2004, 2005], 'test': [2007]},
                   5: {'train': [1991, 2004], 'valid': [2005, 2006], 'test': [2008]}
    }

    CreditCard_dicts = {0: {'seed': 42, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                        1: {'seed': 52, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                        2: {'seed': 62, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                        3: {'seed': 72, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                        4: {'seed': 82, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
    }

    Bank_dicts = {0: {'seed': 42, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                  1: {'seed': 52, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                  2: {'seed': 62, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                  3: {'seed': 72, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                  4: {'seed': 82, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
    }

    Student_dicts = {0: {'seed': 42, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                     1: {'seed': 52, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                     2: {'seed': 62, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                     3: {'seed': 72, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                     4: {'seed': 82, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
    }

    Stroke_dicts = {0: {'seed': 42, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                    1: {'seed': 52, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                    2: {'seed': 62, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                    3: {'seed': 72, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
                    4: {'seed': 82, 'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2},
    }


    if dataset_name == 'ChinaFSD':
        return ChinaFSD_dicts
    elif dataset_name == 'USFSD':
        return USFSD_dicts
    elif dataset_name == 'CreditCard':
        return CreditCard_dicts
    elif dataset_name == 'Bank':
        return Bank_dicts
    elif dataset_name == 'Student':
        return Student_dicts
    elif dataset_name == 'Stroke':
        return Stroke_dicts
    else:
        return 'wrong'


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=100, delta=0, path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 100
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, epoch, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        self.val_loss_min = val_loss
        if self.path:
            torch.save(model.state_dict(), self.path)



def train_valid_test(config,
                    train_loader, valid_cx,valid_nx,valid_y, test_cx,test_nx,test_y, Nu, categories,
                    repeat, best_model_save_path=False):
    setup_seed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nc = len(categories)
    model = EFD(categories, Nc, Nu, d=config['dim_emb'],
                cross_layers=config['cross_layers'],
                mlp_hidden_dims=config['mlp_hidden_dims']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_func = BalLossL2(alpha=config['alpha'], lambda_reg=config['lambda_reg'])

    scores_train_list = []
    scores_valid_list = []
    scores_test_list = []
    params = config
    early_stopping = EarlyStopping(patience=config['patience'], path=best_model_save_path)
    for epoch in range(config['epochs']):
        # train ============================================
        start_time = time.time()
        model.train()
        loss_list = []
        classify_loss_list = []
        l2_reg_loss_list = []
        for cx, nx, labels in train_loader:
            cx, nx, labels = cx.to(device), nx.to(device), labels.to(device)
            optimizer.zero_grad()
            y_probs,c2d_scalar = model(cx, nx)
            loss, classify_loss, l2_reg_loss = loss_func(y_probs.squeeze(), labels, model)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            classify_loss_list.append(classify_loss)
            l2_reg_loss_list.append(l2_reg_loss)
        loss_train = np.mean(loss_list)
        scores_train = evaluate(labels.detach().cpu().numpy().astype(int),
                                y_probs.detach().cpu().numpy().squeeze(),
                                epo=epoch,
                                loss=loss_train,
                                classify_loss=np.mean(classify_loss_list),
                                l2_reg_loss=np.mean(l2_reg_loss_list),
                                params=params)

        end_time = time.time()
        cost_time = end_time - start_time
        scores_train['cost_time'] = cost_time
        scores_train_list.append(scores_train)

        # valid ============================================
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            cx, nx, labels = valid_cx.to(device), valid_nx.to(device), valid_y.to(device)
            y_probs,c2d_scalar = model(cx, nx)
            loss_valid, classify_loss, l2_reg_loss = loss_func(y_probs.squeeze(), labels, model)
            scores_valid = evaluate(labels.detach().cpu().numpy().astype(int),
                                    y_probs.detach().cpu().numpy().squeeze(),
                                    epo=epoch,
                                    loss=loss_valid.item(),
                                    classify_loss=classify_loss,
                                    l2_reg_loss=l2_reg_loss,
                                    params=params)

        end_time = time.time()
        cost_time = end_time - start_time
        scores_valid['cost_time'] = cost_time
        scores_valid_list.append(scores_valid)
        early_stopping(classify_loss, epoch, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # test ============================================
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        cx, nx, labels = test_cx.to(device), test_nx.to(device), test_y.to(device)
        y_probs,c2d_scalar = model(cx, nx)
        loss_test, classify_loss, l2_reg_loss = loss_func(y_probs.squeeze(), labels, model)
        scores_test = evaluate(labels.detach().cpu().numpy().astype(int),
                               y_probs.detach().cpu().numpy().squeeze(),
                               epo=epoch,
                               loss=loss_test.item(),
                               classify_loss=classify_loss,
                               l2_reg_loss=l2_reg_loss,
                               params=params)

    end_time = time.time()
    cost_time = end_time - start_time
    scores_test['cost_time'] = cost_time
    scores_test_list.append(scores_test)
    print(epoch)

    scores_test['repeat'] = repeat
    scores_test['Dataset'] = config['dataset_name']

    return scores_test











