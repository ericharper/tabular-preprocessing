from dask.distributed import Client, LocalCluster
import dask
import subprocess
import dask.dataframe as dd
from dask import delayed
import pandas as pd
from dask_ml.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import Categorizer, OrdinalEncoder, MinMaxScaler
from dask_ml.preprocessing import Categorizer, DummyEncoder, MinMaxScaler
from dask_ml.impute import SimpleImputer
import pickle
import time
import json
from joblib import dump, load
import pyarrow.parquet as pq

import shutil
import os
import glob


def launch_cluster():
    """

    :return:
    """
    cmd = "hostname --all-ip-addresses"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    ip = str(output.decode()).split()[0]

    cluster = LocalCluster(ip=ip, n_workers=8)
    client = Client(cluster)
    return client


def csv_to_parquet(in_path, out_path, num_files, shuffle=False, random_state=123):
    """

    :param in_path:
    :param out_path:
    :param num_files:
    :param shuffle:
    :param random_state:
    :return:
    """
    df = dd.read_csv(in_path, dtype=str)

    if shuffle:
        df = df.sample(frac=1.0, random_state=random_state)

    df = df.repartition(npartitions=num_files)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    df.to_parquet(out_path)  # writes one parquet file for each partition


def impute_zeros(train_paths, test_paths, out_path):
    # Don't need to fit train for constant imputer.....
    train = dd.read_parquet(train_paths)
    test = dd.read_parquet(test_paths)

    zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
    train = zero_imputer.fit_transform(train)
    test = zero_imputer.transform(test)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    train.to_parquet(out_path + '/train')
    test.to_parquet(out_path + '/test')


def load_chunk(file_list):
    df = pq.ParquetDataset(file_list).read_pandas()
    df = df.to_pandas()
    return df

def main():

    IN_PATH = './data/iris/iris.data'

    OUT_ROOT = './data/iris/preprocessing'

    NUM_FILES = 10

    SHUFFLE = True

    RANDOM_STATE = 123

    TRAIN_SPLIT = .8

    NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    CAT_COLS = ['species']

    NUMERIC_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    client = launch_cluster()
    print(client)

    csv_to_parquet(IN_PATH, OUT_ROOT + '/csv_to_parquet', NUM_FILES, SHUFFLE, random_state=RANDOM_STATE)

    file_list = glob.glob(OUT_ROOT + '/csv_to_parquet/*.parquet')

    df = load_chunk(file_list)

    print(df.head())

    train_idx = int(TRAIN_SPLIT * len(file_list))
    train_files = file_list[0:train_idx]
    test_files = file_list[train_idx:]

    print(train_files)
    print(test_files)

    impute_zeros(train_files, test_files, OUT_ROOT + '/impute_zeros')


if __name__ == '__main__':
    main()
