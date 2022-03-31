from os.path import exists

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as T
from pyspark_test import assert_pyspark_df_equal

from src.analysis.tc_sensitivity_analysis import _add_td, _add_tc, _add_reduce_co, _add_lower_quart_co, \
    _add_upper_quart_co, _add_adaptive_co, _add_co, \
    _plot_agg_co_per_tc_model, _get_agg_co_per_tc_model, _plot_agg, _plot_check_distr_model, \
    _plot_check_distr_model_over_time


@pytest.fixture
def spark_session():
    master = "local"  # use 4 cores
    app_name = "test sensitivity analysis of the checkpoint overhead of the eval cluster trace data"
    spark_session = SparkSession.builder \
        .master(master) \
        .appName(app_name) \
        .getOrCreate()
    return spark_session


class TestTcSensitivityAnalysis:
    # SA constants
    TW_STEPS = [5, 10, 20, 50, 100]
    IO_RATIO = 16.418926
    TP = 0.04

    def test__add_tc(self, spark_session):
        fixture_path = "fixtures/got/add_tc.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _add_tc(prediction_output, self.TW_STEPS, self.IO_RATIO)
        new_col_set = {"tc"}

        assert new_col_set.issubset(got.columns)
        assert (len(prediction_output.columns) + len(new_col_set)) == len(got.columns)

    def test__add_td(self, spark_session):
        # Note: all fixtures are derived from: BinaryTaskfailClassifier-trace-Alibaba2018-size-1-mem-3072-testfrac-0.2-sampleseed-66-fittime-1800-nrows-3000000-cvK-5-chunk.csv
        fixture_path = "fixtures/got/add_td.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _add_td(prediction_output)
        new_col_set = {"td"}

        expected_fixture_path = "fixtures/expected/add_td.csv"
        expected = spark_session.read.csv(expected_fixture_path, header=True, inferSchema=True)

        assert new_col_set.issubset(got.columns)
        assert (len(prediction_output.columns) + len(new_col_set)) == len(got.columns)
        assert_pyspark_df_equal(got, expected, order_by=["task_name"])

    def test__add_reduce_co(self, spark_session):
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        # cast the `tc` column to ArrayType because it cant be inferred from csv
        prediction_output = prediction_output.withColumn("tc", F.col("tc").cast(T.StringType()))
        prediction_output = prediction_output.withColumn("tc", F.split(F.col("tc"), ";", -1))

        got = _add_reduce_co(prediction_output)
        new_col_set = {"reduce_co"}

        expected_reduce_co = [262.435181565379, 267.7397081830153, 273.04423480065157, 278.3487614182878]
        reduce_co = got.select("reduce_co").filter(F.col("reduce_co").isNotNull()).collect()[0][0]
        assert reduce_co == expected_reduce_co

    def test__add_lower_quart_co(self, spark_session):
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        # cast the `tc` column to ArrayType because it cant be inferred from csv
        prediction_output = prediction_output.withColumn("tc", F.col("tc").cast(T.StringType()))
        prediction_output = prediction_output.withColumn("tc", F.split(F.col("tc"), ";", -1))

        got = _add_lower_quart_co(prediction_output)
        new_col_set = {"second_quant_co"}

        assert new_col_set.issubset(got.columns)
        assert (len(prediction_output.columns) + len(new_col_set)) == len(got.columns)
        assert got.select("second_quant_co").distinct().count() == 3

    def test__add_upper_quart_co(self, spark_session):
        # replica of the above test
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        # cast the `tc` column to ArrayType because it cant be inferred from csv
        prediction_output = prediction_output.withColumn("tc", F.col("tc").cast(T.StringType()))
        prediction_output = prediction_output.withColumn("tc", F.split(F.col("tc"), ";", -1))

        got = _add_upper_quart_co(prediction_output)
        new_col_set = {"third_quant_co"}

        assert new_col_set.issubset(got.columns)
        assert (len(prediction_output.columns) + len(new_col_set)) == len(got.columns)
        assert got.select("third_quant_co").distinct().count() == 3

    def test__add_adaptive_co(self, spark_session):
        TP = 4 # in min
        # replica of the above test
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        # cast the `tc` column to ArrayType because it cant be inferred from csv
        prediction_output = prediction_output.withColumn("tc", F.col("tc").cast(T.StringType()))
        prediction_output = prediction_output.withColumn("tc", F.split(F.col("tc"), ";", -1))

        got = _add_adaptive_co(prediction_output, TP)
        new_col_set = {"adaptive_co"}

        assert new_col_set.issubset(got.columns)
        assert (len(prediction_output.columns) + len(new_col_set)) == len(got.columns)
        assert got.select("adaptive_co").distinct().count() == 6

    def test__get_agg_co_per_tc_model(self, spark_session):
        TP = 4
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        # cast the `tc` column to ArrayType because it cant be inferred from csv
        prediction_output = prediction_output.withColumn("tc", F.col("tc").cast(T.StringType()))
        prediction_output = prediction_output.withColumn("tc", F.split(F.col("tc"), ";", -1))
        prediction_output = _add_co(prediction_output, tp=TP)

        got = _get_agg_co_per_tc_model(prediction_output, metric="mean", acc=1000)

        expected_fixture_path = "fixtures/expected/get_avg_co_per_tc_model.csv"
        expected = spark_session.read.csv(expected_fixture_path, header=True, inferSchema=True)
        got = got.withColumn("zip_tc", F.col("zip_tc").cast(T.DoubleType()))
        assert_pyspark_df_equal(got, expected, order_by=["zip_tc"], check_dtype=False)

    def test__plot_agg_co_per_tc_model(self, spark_session):
        fixture_path = "fixtures/expected/get_avg_co_per_tc_model.csv"
        avg_co_per_tc_model = pd.read_csv(fixture_path, header=0)
        _plot_agg_co_per_tc_model(avg_co_per_tc_model, filename="test", metric="avg")
        assert exists("../../out/analysis/sa/plots/bar/test_avg_co.pdf")

    def test__plot_agg(self, spark_session):
        TP = 4
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        # cast the `tc` column to ArrayType because it cant be inferred from csv
        prediction_output = prediction_output.withColumn("tc", F.col("tc").cast(T.StringType()))
        prediction_output = prediction_output.withColumn("tc", F.split(F.col("tc"), ";", -1))
        prediction_output = _add_co(prediction_output, tp=TP)

        _plot_agg(prediction_output, filename="test", metric="median", acc=1000)
        assert exists("../../out/analysis/sa/plots/bar/test_median_co.pdf")

    def test_plot_check_model_distr(self, spark_session):
        fixture_path = "fixtures/got/add_co.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        _plot_check_distr_model(prediction_output, filename="test")
        assert exists("../../out/analysis/sa/plots/bar/test_perc_check.pdf")

    def test_plot_check_distr_model_over_time(self, spark_session):
        # a stratified sample ({0: 0.01, 1: 0.5}) of the 1% cluster experiment from 31.12.2021 (19k rows)
        fixture_path = "fixtures/got/plot_check_distr_model_over_time.csv"
        prediction_output = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        days = 8
        label = 0
        filename="test"
        _plot_check_distr_model_over_time(prediction_output, filename, days, label)
        assert exists("../../out/analysis/sa/plots/bar/test_perc_check_label_0_8_days.pdf")
