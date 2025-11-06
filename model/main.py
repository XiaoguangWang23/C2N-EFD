import pandas as pd
import numpy as np
from utils.data_utils import load_data
from utils.train_utils import setup_seed,train_valid_test,data_split
import random
import torch
import torch.nn as nn




config = {
    'dim_emb':4,
    'cross_layers': 1,
    'mlp_hidden_dims': [128, 16],
    'weight_decay':0,

    'epochs': 150,
    'batch_size': 1024,
    'lr': 0.01,
    'alpha': 12,
    'lambda_reg': 0.003,
    'patience': 20
}


if __name__ == '__main__':
    setup_seed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    dataset_name = 'ChinaFSD'
    data_path = '../data/ChinaFSD/ChinaFSD.csv'
    config['dataset_name'] = dataset_name
    config['data_path'] = data_path
    split_dicts = data_split(dataset_name)

    for repeat in [0]:
        splits = split_dicts[repeat]
        train_loader, (valid_cx, valid_nx, valid_y), (test_cx, test_nx, test_y), Nu, categories = \
            load_data(config, splits)
        df_result_list = []
        best_model_save_path = False
        df_result = train_valid_test(config=config,
                                     train_loader=train_loader,
                                     valid_cx=valid_cx,
                                     valid_nx=valid_nx,
                                     valid_y=valid_y,
                                     test_cx=test_cx,
                                     test_nx=test_nx,
                                     test_y=test_y,
                                     Nu=Nu,
                                     categories=categories,
                                     repeat=repeat,
                                     best_model_save_path=best_model_save_path
                                     )
        df_result_list.append(df_result)
        data_results = pd.concat(df_result_list, axis=0)
        print(data_results)



