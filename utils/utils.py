import pandas as pd
import numpy as np



def fetch_ucr_dataset(dataset):

    path = "/Users/pboniol/Desktop/datasets/UCRArchive_2018/{}/".format(dataset)

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