import pandas as pd
import numpy as np



def fetch_ucr_dataset(dataset,path,variable_length=False):

    if variable_length:
        path += '{}/'.format(dataset)
        with open(path + "{}_TRAIN.tsv".format(dataset),'r') as f:
            train = f.readlines()
        train = [train_line.replace('\n','') for train_line in train]
        labels_train = []
        ts_train = []
        for train_line in train:
            val = train_line.split('\t')
            labels_train.append(int(val[0]))
            ts_train.append(np.array([float(v) for v in val[1:]]))
            ts_train[-1] = ts_train[-1][~np.isnan(ts_train[-1])]

        with open(path + "{}_TEST.tsv".format(dataset),'r') as f:
            test = f.readlines()
        test = [test_line.replace('\n','') for test_line in test]
        labels_test = []
        ts_test = []
        for test_line in test:
            val = test_line.split('\t')
            labels_test.append(int(val[0]))
            ts_test.append(np.array([float(v) for v in val[1:]]))
            ts_test[-1] = ts_test[-1][~np.isnan(ts_test[-1])]
        return {'data_train':ts_train,'target_train':np.array(labels_train), 'data_test':ts_test, 'target_test':np.array(labels_test)}

    else:
        path += '{}/'.format(dataset)
        train_data = pd.read_csv(path + "{}_TRAIN.tsv".format(dataset),sep='\t',header=None)
        target_train = np.array(train_data[0].values)
        train_data = train_data.drop(0,axis=1)
        train_data = train_data.fillna(0)
        data_train = np.array(train_data.values)

        test_data = pd.read_csv(path + "{}_TEST.tsv".format(dataset),sep='\t',header=None)
        target_test = np.array(test_data[0].values)
        test_data = test_data.drop(0,axis=1)
        test_data = test_data.fillna(0)
        data_test = np.array(test_data.values)
        return {'data_train':data_train,'target_train':target_train, 'data_test':data_test, 'target_test':target_test}