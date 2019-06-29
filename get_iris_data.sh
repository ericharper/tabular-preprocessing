#!/bin/bash

mkdir -p ./data/iris

if [ ! -f ./data/iris/iris.data ]; then
    # column names
    echo 'sepal_length,sepal_width,petal_length,petal_width,species' > ./data/iris/iris.data
    #wget -P ./data/iris/ 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' -O - >> ./data/iris/iris.data
fi
