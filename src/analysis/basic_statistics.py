import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark import SparkConf
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation, Summarizer, ChiSquareTest
from pyspark.sql import SparkSession, DataFrame, functions as F

sty = "seaborn"
mpl.style.use(sty)

logging.basicConfig(filename='../../logs/analysis/ali2018-basic-statistics.log',
                    level=logging.INFO, format='%(asctime)s %(message)s')


def get_basic_statistics(cluster_trace: DataFrame, method: str):
    """
    :param cluster_trace: the raw cluster trace
    :param method:  the correlation method
    :return:
    """
    logging.info("\n ======= START SUMMARY RUN ======= \n")
    cluster_trace = _densify_vectors(cluster_trace)
    # _summarize(cluster_trace)
    # this requires categorical features
    # _hypothesis_test(cluster_trace)
    _plot_correlation(cluster_trace, method)


def _densify_vectors(cluster_trace: DataFrame):
    """
    :return: the cluster trace with a dense feature vector
    """
    # impute missing values to densify the feature vector
    # same logic as in model training (../model/*.scala)
    numeric_types = ["int", "double"]
    numeric_cols = [i[0] for i in cluster_trace.dtypes if
                    i[1] in numeric_types and i[0] in FEATURES and i[0] != "labels"]

    # list comprehension to add suffix to all imputed cols
    imputed_cols = [j + "_imp" for j in numeric_cols]

    imputer = Imputer(inputCols=numeric_cols, outputCols=imputed_cols)
    imputer = imputer.fit(cluster_trace)
    cluster_trace = imputer.transform(cluster_trace)

    # add feature vector
    logging.info(f"the final features: \n {imputed_cols}")
    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols(imputed_cols)
    vecAssembler.setHandleInvalid('skip')
    cluster_trace = vecAssembler.transform(cluster_trace)

    return cluster_trace


def _hypothesis_test(cluster_trace: DataFrame):
    """
    :param cluster_trace: the cluster trace with feature and label vector
    """
    r = ChiSquareTest.test(cluster_trace, "task_type", "labels")
    r.show(truncate=False)
    logging.info(r.toPandas().to_records(index=False))
    pass


def _summarize(cluster_trace: DataFrame):
    # create summarizer for multiple metrics "mean" and "count"
    summarizer = Summarizer.metrics("count", "max", "mean", "min", "normL1", "normL2", "numNonZeros", "std", "sum",
                                    "variance")
    # compute statistics for multiple metrics without weight
    logging.info("statistics: count, max, mean, min, normL1, normL2, numNonZeros, std, sum, variance")
    features_summary = cluster_trace.select(summarizer.summary(cluster_trace.features).alias("summary"))
    features_summary = features_summary.select(F.col("summary.count").alias("count"),
                                               F.col("summary.mean").alias("mean"),
                                               F.col("summary.min").alias("min"),
                                               F.col("summary.normL1").alias("normL1"),
                                               F.col("summary.normL2").alias("normL2"),
                                               F.col("summary.numNonZeros").alias("numNonZeros"),
                                               F.col("summary.std").alias("std"),
                                               F.col("summary.variance").alias("variance"))
    features_summary.show(truncate=False)
    features_sum = features_summary.toPandas().to_records(index=False)
    logging.info(features_sum)
    pass


def _plot_correlation(cluster_trace: DataFrame, method: str):
    """
    check for correlations among features
    :param cluster_trace: the cluster trace with a dense feature vector
    :param method the correlation method ("pearson/spearman")
    """
    corr_df = Correlation.corr(cluster_trace, "features", method)
    corr_arr = np.array(corr_df.head()[0].toArray())
    # set the feature names as columns
    corr_pd = pd.DataFrame(corr_arr, columns=FEATURES)
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    #plt.title(f"{method} correlation matrix of the cleaned Alibaba 2018 cluster trace")

    # Generate a mask for the upper triangle bc the spearman matrix is symmetric
    mask = np.triu(np.ones_like(corr_arr, dtype=bool))

    sns.heatmap(
        corr_pd,
        cmap="coolwarm",
        mask=mask,
        linewidths=.5,  # this sepeartes each cell
        center=0,  # 0 corr is the middle
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .7}
    )
    plt.yticks(ticks=corr_pd.index, labels=FEATURES, rotation=20)
    plt.xticks(ticks=corr_pd.index, labels=FEATURES, rotation=20)
    # this overwrites
    plt.savefig(f"../../out/analysis/stats/{dataset_name}-{method}.pdf", format="pdf")
    fig.show()
    # the final features (from numbers) are in the logs
    pass


if __name__ == '__main__':
    seed = 61
    dataset_name = "batch_jobs_clean_03inst_1task_00015S_1F"
    # taken from GBTCVTaskFailBinPred.scala
    FEATURES = ["instance_num",  # number of instances for the task
                "task_type",  # values 1-12, meaning unknown (ordinal categ. feature)
                "plan_cpu",  # number of cpus needed by the task, 100 is 1 core
                "plan_mem",  # normalized memory size, [0, 100]
                # from batch instances
                "seq_no",
                # "labels",
                # custom fields I added in trace.py
                "map_reduce",
                # whether this task is a map "m" or reduce "r" operation (nominal categ. feature) -> 1hot encode
                "sched_intv",  # the schedule interval (in ?) if this value is set, its a recurring job (batch job)
                "job_exec",  # the number of execution of this task
                # "logical_job_name"
                ]
    config = SparkConf().setAll([
        ('spark.driver.memory', '5g')
    ])

    spark = SparkSession.builder \
        .master("local[3]") \
        .appName("Alibaba 2018 feature correlation basic statistics") \
        .config(conf=config) \
        .getOrCreate()

    # read in the data
    cluster_trace = spark.read.csv(
        f"../../out/clean/{dataset_name}/*.csv.gz",
        header=True,
        inferSchema=True
    )

    cluster_trace = cluster_trace.withColumn("map_reduce", F.when(F.col("map_reduce") == "m", 1).otherwise(0))
    get_basic_statistics(cluster_trace, method="pearson")
    get_basic_statistics(cluster_trace, method="spearman")
