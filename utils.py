from dask.distributed import Client, LocalCluster
import dask
import subprocess
import dask.dataframe as dd
from dask import delayed
import pandas as pd
from dask_ml.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import Categorizer, OrdinalEncoder, StandardScaler
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
    cmd = "hostname --all-ip-addresses"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    ip = str(output.decode()).split()[0]

    cluster = LocalCluster(ip=ip, n_workers=8)
    client = Client(cluster)
    return client


def csv_to_parquet(in_path, out_path, num_files, shuffle=False, random_state=123):
    df = dd.read_csv(in_path, dtype=str)

    if shuffle:
        df = df.sample(frac=1.0, random_state=random_state)

    df = df.repartition(npartitions=num_files)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    df.to_parquet(out_path)  # writes one parquet file for each partition


def split_numerics_cats(in_paths, out_path, numeric_cols, cat_cols):
    df = dd.read_parquet(in_paths)

    numerics = df[numeric_cols].astype('float32')
    cats = df[cat_cols]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    numerics.to_parquet(out_path + '/numerics')
    cats.to_parquet(out_path + '/cats')


def impute_zeros(in_paths, out_path):
    df = dd.read_parquet(in_paths)

    zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
    df = zero_imputer.fit_transform(df)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    df.to_parquet(out_path)


def impute_missing(in_paths, out_path):
    df = dd.read_parquet(in_paths)

    missing_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
    df = missing_imputer.fit_transform(df)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    df.to_parquet(out_path)


def scale_numerics(train_paths, test_paths, out_path):
    train = dd.read_parquet(train_paths)
    test = dd.read_parquet(test_paths)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    train.to_parquet(out_path + '/train')
    test.to_parquet(out_path + '/test')

    # TODO: save StandardScaler


def ordinal_encode(train_paths, test_paths, out_path):
    train = dd.read_parquet(train_paths)
    test = dd.read_parquet(test_paths)

    categorizer = Categorizer()
    train = categorizer.fit_transform(train)

    train_categories = categorizer.categories_

    # replace categories in test that are not in train with 'MISSING
    for col, category in train_categories.items():
        test[col] = test[col].mask(~test[col].isin(category.categories.values), 'MISSING')
    test = categorizer.transform(test)


    ordinal_encoder = OrdinalEncoder()
    train = ordinal_encoder.fit_transform(train)

    test = ordinal_encoder.transform(test)

    train = train.astype('int64')
    test = test.astype('int64')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    train.to_parquet(out_path + '/train')
    test.to_parquet(out_path + '/test')

    # TODO: save Ordinal Encoder
    # TODO: save maxes (for categorical embedding later)
    # TODO: Deal with categories in test that are not in train


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

    split_numerics_cats(file_list, OUT_ROOT + '/split_numerics_cats', NUMERIC_COLS, CAT_COLS)

    numeric_file_list = glob.glob(OUT_ROOT + '/split_numerics_cats/numerics/*.parquet')

    impute_zeros(numeric_file_list, OUT_ROOT + '/impute_zeros')

    numeric_file_list = glob.glob(OUT_ROOT + '/impute_zeros/*.parquet')

    numerics = load_chunk(numeric_file_list)

    print(numerics.head())

    train_idx = int(TRAIN_SPLIT * len(file_list))

    numeric_train_files = numeric_file_list[0:train_idx]
    numeric_test_files = numeric_file_list[train_idx:]

    print(numeric_train_files)
    print(numeric_test_files)

    scale_numerics(numeric_train_files, numeric_test_files, OUT_ROOT + '/scale_numerics')

    numeric_train_files = glob.glob(OUT_ROOT + '/scale_numerics/train/*.parquet')
    numeric_test_files = glob.glob(OUT_ROOT + '/scale_numerics/test/*.parquet')

    numeric_train = load_chunk(numeric_train_files)
    numeric_test = load_chunk(numeric_test_files)

    print('Numeric Train:')
    print(numeric_train.head())
    print(numeric_train.dtypes)
    print('Numeric Test:')
    print(numeric_test.head())
    print(numeric_test.dtypes)

    cat_file_list = glob.glob(OUT_ROOT + '/split_numerics_cats/cats/*.parquet')

    cats = load_chunk(cat_file_list)

    print(cats.head())

    cat_train_files = cat_file_list[0:train_idx]
    cat_test_files = cat_file_list[train_idx:]

    print(cat_train_files)
    print(cat_test_files)

    ordinal_encode(cat_train_files, cat_test_files, OUT_ROOT + '/ordinals')

    ordinal_train_files = glob.glob(OUT_ROOT + '/ordinals/train')
    ordinal_test_files = glob.glob(OUT_ROOT + '/ordinals/test')

    ordinal_train = load_chunk(ordinal_train_files)
    ordinal_test = load_chunk(ordinal_test_files)

    print('Ordinal Train:')
    print(ordinal_train.head())
    print(ordinal_train.dtypes)
    print('Ordinal Test:')
    print(ordinal_test.head())
    print(ordinal_test.dtypes)






if __name__ == '__main__':
    main()
