"""spark_numpy_vectors, pyspark utilities on Windows."""
from __future__ import print_function


def spark_os_env():
    """Setup spark environment for Windows OS."""
    import os
    import sys
    spark_path = "C:/opt/spark"
    os.environ['SPARK_HOME'] = spark_path
    os.environ['HADOOP_HOME'] = spark_path
    sys.path.append(spark_path + "/bin")
    sys.path.append(spark_path + "/python")
    sys.path.append(spark_path + "/python/pyspark/")
    sys.path.append(spark_path + "/python/lib")
    sys.path.append(spark_path + "/python/lib/pyspark.zip")
    sys.path.append(spark_path + "/python/lib/py4j-0.10.4-src.zip")


spark_os_env()

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

import numpy as np
import collections

from operator import add


Params = collections.namedtuple('Params', ['key', 'weight'])
ChunkData = collections.namedtuple('ChunkData', ['uri', 'params'])
DataRow = collections.namedtuple('DataRow', ['id', 'data', 'params'])


def compute_tensor_data(uri, params):
    """Generate some tensor data of shape height x width x depth """
    return np.arange(4 * 3 * 3).reshape(4, 3, 3) * params.weight


def compute_task(data):
    """Spark task computing RDD results from input data chunk """
    tensor = compute_tensor_data(data.uri, data.params)
    vectors = tensor.reshape(tensor.shape[0]*tensor.shape[1], tensor.shape[2])

    results = []
    for vector_id, vector in enumerate(vectors):
        results.append(DataRow(vector_id, vector, data.params))
    return results


def rdd_file_exists(sc, path):
    """Check if RDD file exists."""
    rdd = sc.pickleFile(path)
    try:
        rdd.first()
        return True
    except:
        return False


def get_group_file(id):
    return 'group_' + str(id)


def test_sort_and_group(rdd):
    print ("collected vector data: ")
    results = rdd.collect()
    for result in results:
        print(result.id, ": ", result.data)

    print ("sort vectors by id: ")
    sorted = rdd.map(lambda x: (x[0], x[1])).sortByKey()
    results = sorted.collect()
    for result in results:
        print(result)

    print ("group vectors by id: ")
    groups = rdd.map(lambda x: (x[0], x[1])).reduceByKey(lambda x, y: [x, y])
    results = groups.collect()
    for result in results:
        print(result)
    return


def test_group_by_id_and_save(rdd):
    print("get groups by id: ")
    groups = rdd.map(lambda x: (x[0], x[1])).reduceByKey(lambda x, y: [x, y])
    num_ids = groups.count()

    print("save individual picked files for ", num_ids, " groups")
    for id in range(num_ids):
        group_file = get_group_file(id)
        if not rdd_file_exists(sc, group_file):
            print ("group file: ", group_file)
            groups.filter(lambda x: x[0] == id).saveAsPickleFile(group_file)

    for id in range(num_ids):
        group_file = get_group_file(id)
        group = sc.pickleFile(group_file)
        results = group.collect()
        print("group: ", id)
        for result in results:
            print(result)
    return


if __name__ == "__main__":
    print("spark numpy vector test")
    spark = SparkSession\
        .builder\
        .appName("spark_numpy_vectors")\
        .getOrCreate()

    sc = spark.sparkContext

    uris = ['asset001', 'assert002', 'asset003']
    weights = [1, 0, 2]
    chunks = []
    for uri, weight in zip(uris, weights):
        chunks.append(ChunkData(uri, Params('key_test1', weight)))  

    filename = 'tmp\\test.' + '.results'
    if not rdd_file_exists(sc, filename):
        print("compute vector file ...")
        sc.parallelize(chunks).flatMap(compute_task).saveAsPickleFile(filename)
    else:
        print('vectors file already exists ...')
    rdd = sc.pickleFile(filename)

    test_sort_and_group(rdd)
    test_group_by_id_and_save(rdd)

    spark.stop()

