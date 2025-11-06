import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import copy
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer




def load_ChinaFSD(config, splits):
    data = pd.read_csv(config['data_path'])
    data.drop(columns=['nodeID'], inplace=True)

    columns = data.columns
    cat_feats = ['IndustryCodeC', 'IsCocurP', 'InefficInvestSign', 'ConcurrentPosition', 'OneControlMany']
    num_feats = [col for col in columns if col not in cat_feats + ['Year', 'Label']]
    data = data[num_feats + cat_feats + ['Year', 'Label']]
    categories = [len(np.unique(data[col].values)) for col in cat_feats]
    Nu = len(num_feats)
    Nc = len(cat_feats)

    data[num_feats] = StandardScaler().fit_transform(data[num_feats])
    for col in cat_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    trainset = copy.deepcopy(data[(data['Year'] >= splits['train'][0]) & (data['Year'] <= splits['train'][-1])])
    validset = copy.deepcopy(data[(data['Year'] >= splits['valid'][0]) & (data['Year'] <= splits['valid'][-1])])
    testset = copy.deepcopy(data[(data['Year'] >= splits['test'][0]) & (data['Year'] <= splits['test'][-1])])

    train_y = trainset['Label'].values
    valid_y = validset['Label'].values
    test_y = testset['Label'].values

    trainset.drop(columns=['Year', 'Label'], inplace=True)
    validset.drop(columns=['Year', 'Label'], inplace=True)
    testset.drop(columns=['Year', 'Label'], inplace=True)

    train_cx = torch.tensor(trainset[cat_feats].values, dtype=torch.long)
    train_nx = torch.tensor(trainset[num_feats].values, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    valid_cx = torch.tensor(validset[cat_feats].values, dtype=torch.long)
    valid_nx = torch.tensor(validset[num_feats].values, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)

    test_cx = torch.tensor(testset[cat_feats].values, dtype=torch.long)
    test_nx = torch.tensor(testset[num_feats].values, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = Data.TensorDataset(train_cx, train_nx, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)

    return train_loader, (valid_cx,valid_nx,valid_y), (test_cx,test_nx,test_y), Nu, categories


def load_USFSD(config, splits):
    data = pd.read_csv(config['data_path'])
    data.drop(columns=['gvkey'], inplace=True)
    data.drop_duplicates(inplace=True)
    data.index = range(data.shape[0])

    for col in data.columns:
        if data[col].isnull().any():
            mean_val = data[col].mean()
            data[col].fillna(mean_val, inplace=True)

    columns = data.columns
    cat_feats = ['issue']
    num_feats = [col for col in columns if col not in cat_feats + ['Year', 'Label']]
    data = data[num_feats + cat_feats + ['Year', 'Label']]
    categories = [len(np.unique(data[col].values)) for col in cat_feats]
    Nu = len(num_feats)
    Nc = len(cat_feats)

    data[num_feats] = StandardScaler().fit_transform(data[num_feats])
    for col in cat_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    trainset = copy.deepcopy(data[(data['Year'] >= splits['train'][0]) & (data['Year'] <= splits['train'][-1])])
    validset = copy.deepcopy(data[(data['Year'] >= splits['valid'][0]) & (data['Year'] <= splits['valid'][-1])])
    testset = copy.deepcopy(data[(data['Year'] >= splits['test'][0]) & (data['Year'] <= splits['test'][-1])])

    train_y = trainset['Label'].values
    valid_y = validset['Label'].values
    test_y = testset['Label'].values

    trainset.drop(columns=['Year', 'Label'], inplace=True)
    validset.drop(columns=['Year', 'Label'], inplace=True)
    testset.drop(columns=['Year', 'Label'], inplace=True)

    train_cx = torch.tensor(trainset[cat_feats].values, dtype=torch.long)
    train_nx = torch.tensor(trainset[num_feats].values, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    valid_cx = torch.tensor(validset[cat_feats].values, dtype=torch.long)
    valid_nx = torch.tensor(validset[num_feats].values, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)

    test_cx = torch.tensor(testset[cat_feats].values, dtype=torch.long)
    test_nx = torch.tensor(testset[num_feats].values, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = Data.TensorDataset(train_cx, train_nx, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)

    return train_loader, (valid_cx,valid_nx,valid_y), (test_cx,test_nx,test_y), Nu, categories


def load_CreditCard(config, splits):
    data = pd.read_csv(config['data_path'])
    data.drop(columns=['ID'], inplace=True)
    data.drop_duplicates(inplace=True)
    data.index = range(data.shape[0])

    num_feats = ['LIMIT_BAL', 'AGE',
                 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    cat_feats = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    data = data[num_feats + cat_feats + ['Label']]
    categories = [len(np.unique(data[col].values)) for col in cat_feats]
    Nu = len(num_feats)
    Nc = len(cat_feats)

    data[num_feats] = StandardScaler().fit_transform(data[num_feats])
    for col in cat_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop(columns=['Label'])
    y = data['Label'].values

    trainset, tempset, train_y, temp_y = train_test_split(
        X, y, random_state=splits['seed'],
        test_size=splits['val_size'] + splits['test_size']
    )
    validset, testset, valid_y, test_y = train_test_split(
        tempset, temp_y, random_state=splits['seed'],
        test_size=splits['test_size'] / (splits['val_size'] + splits['test_size'])
    )

    train_cx = torch.tensor(trainset[cat_feats].values, dtype=torch.long)
    train_nx = torch.tensor(trainset[num_feats].values, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    valid_cx = torch.tensor(validset[cat_feats].values, dtype=torch.long)
    valid_nx = torch.tensor(validset[num_feats].values, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)

    test_cx = torch.tensor(testset[cat_feats].values, dtype=torch.long)
    test_nx = torch.tensor(testset[num_feats].values, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = Data.TensorDataset(train_cx, train_nx, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)

    return train_loader, (valid_cx,valid_nx,valid_y), (test_cx,test_nx,test_y), Nu, categories


def load_Bank(config, splits):
    data = pd.read_csv(config['data_path'])
    num_feats = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    cat_feats = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    data['Label'] = data['Label'].map({'no': 0, 'yes': 1})
    data = data[num_feats + cat_feats + ['Label']]
    categories = [len(np.unique(data[col].values)) for col in cat_feats]
    Nu = len(num_feats)
    Nc = len(cat_feats)

    data[num_feats] = StandardScaler().fit_transform(data[num_feats])
    for col in cat_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop(columns=['Label'])
    y = data['Label'].values

    trainset, tempset, train_y, temp_y = train_test_split(
        X, y, random_state=splits['seed'],
        test_size=splits['val_size'] + splits['test_size']
    )
    validset, testset, valid_y, test_y = train_test_split(
        tempset, temp_y, random_state=splits['seed'],
        test_size=splits['test_size'] / (splits['val_size'] + splits['test_size'])
    )

    train_cx = torch.tensor(trainset[cat_feats].values, dtype=torch.long)
    train_nx = torch.tensor(trainset[num_feats].values, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    valid_cx = torch.tensor(validset[cat_feats].values, dtype=torch.long)
    valid_nx = torch.tensor(validset[num_feats].values, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)

    test_cx = torch.tensor(testset[cat_feats].values, dtype=torch.long)
    test_nx = torch.tensor(testset[num_feats].values, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = Data.TensorDataset(train_cx, train_nx, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)

    return train_loader, (valid_cx,valid_nx,valid_y), (test_cx,test_nx,test_y), Nu, categories


def load_Student(config, splits):
    data = pd.read_csv(config['data_path'],sep=';')
    num_feats = ['Age at enrollment',
                 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
                 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
                 'Unemployment rate', 'Inflation rate', 'GDP']
    cat_feats = ['Marital status', 'Application mode', 'Application order', 'Course', 'evening attendance',
                 'Previous qualification',
                 'Nacionality', "Mother\'s qualification", 'Father\'s qualification', 'Mother\'s occupation',
                 'Father\'s occupation',
                 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender',
                 'Scholarship holder',
                 'International'
                 ]
    data.rename(columns={'Output': 'Label'}, inplace=True)
    data['Label'] = data['Label'].map({'Graduate': 0, 'Enrolled':0, 'Dropout': 1})
    data = data[num_feats + cat_feats + ['Label']]
    categories = [len(np.unique(data[col].values)) for col in cat_feats]
    Nu = len(num_feats)
    Nc = len(cat_feats)

    data[num_feats] = StandardScaler().fit_transform(data[num_feats])
    for col in cat_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop(columns=['Label'])
    y = data['Label'].values

    trainset, tempset, train_y, temp_y = train_test_split(
        X, y, random_state=splits['seed'],
        test_size=splits['val_size'] + splits['test_size']
    )
    validset, testset, valid_y, test_y = train_test_split(
        tempset, temp_y, random_state=splits['seed'],
        test_size=splits['test_size'] / (splits['val_size'] + splits['test_size'])
    )

    train_cx = torch.tensor(trainset[cat_feats].values, dtype=torch.long)
    train_nx = torch.tensor(trainset[num_feats].values, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    valid_cx = torch.tensor(validset[cat_feats].values, dtype=torch.long)
    valid_nx = torch.tensor(validset[num_feats].values, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)

    test_cx = torch.tensor(testset[cat_feats].values, dtype=torch.long)
    test_nx = torch.tensor(testset[num_feats].values, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = Data.TensorDataset(train_cx, train_nx, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)

    return train_loader, (valid_cx,valid_nx,valid_y), (test_cx,test_nx,test_y), Nu, categories



def load_Stroke(config, splits):
    data = pd.read_csv(config['data_path'])
    num_feats = ['age','avg_glucose_level', 'bmi']
    cat_feats = ['gender','hypertension','heart_disease','ever_married',
                    'work_type', 'Residence_type','smoking_status']
    data.drop(columns=['id'], inplace=True)
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())
    data.rename(columns={'stroke': 'Label'}, inplace=True)
    data = data[num_feats + cat_feats + ['Label']]
    categories = [len(np.unique(data[col].values)) for col in cat_feats]
    Nu = len(num_feats)
    Nc = len(cat_feats)

    data[num_feats] = StandardScaler().fit_transform(data[num_feats])
    for col in cat_feats:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop(columns=['Label'])
    y = data['Label'].values

    trainset, tempset, train_y, temp_y = train_test_split(
        X, y, random_state=splits['seed'],
        test_size=splits['val_size'] + splits['test_size']
    )
    validset, testset, valid_y, test_y = train_test_split(
        tempset, temp_y, random_state=splits['seed'],
        test_size=splits['test_size'] / (splits['val_size'] + splits['test_size'])
    )

    train_cx = torch.tensor(trainset[cat_feats].values, dtype=torch.long)
    train_nx = torch.tensor(trainset[num_feats].values, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    valid_cx = torch.tensor(validset[cat_feats].values, dtype=torch.long)
    valid_nx = torch.tensor(validset[num_feats].values, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)

    test_cx = torch.tensor(testset[cat_feats].values, dtype=torch.long)
    test_nx = torch.tensor(testset[num_feats].values, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_dataset = Data.TensorDataset(train_cx, train_nx, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)

    return train_loader, (valid_cx,valid_nx,valid_y), (test_cx,test_nx,test_y), Nu, categories







def load_data(config, splits=None):
    if config['dataset_name'] == 'ChinaFSD':
        return load_ChinaFSD(config, splits)
    elif config['dataset_name'] == 'USFSD':
        return load_USFSD(config, splits)
    elif config['dataset_name'] == 'CreditCard':
        return load_CreditCard(config, splits)
    elif config['dataset_name'] == 'Bank':
        return load_Bank(config, splits)
    elif config['dataset_name'] == 'Student':
        return load_Student(config, splits)
    elif config['dataset_name'] == 'Stroke':
        return load_Stroke(config, splits)
    else:
        raise ValueError("data_name is wrong!")













