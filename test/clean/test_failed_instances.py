from os.path import exists

import pytest
from pyspark import SparkConf
from pyspark.sql import SparkSession

from src.clean.failed_instances import FailedInstances


@pytest.fixture
def spark_session():
    master = "local[2]"
    config = SparkConf().setAll([
        ('spark.driver.memory', '3g'),
    ])
    app_name = "test clean failed Alibaba 2018 cluster trace events"
    spark_session = SparkSession.builder \
        .master(master) \
        .appName(app_name) \
        .config(conf=config) \
        .getOrCreate()
    return spark_session


@pytest.fixture
def fi(spark_session):
    return FailedInstances(
        spark_session=spark_session,
        # created using zcat batch_instance.tar.gz | head -n 10000
        # M7_6 ,j_2745 (no compute metrics), and M3,j_13297 (compute metrics) are fake
        batch_instance_path="fixtures/got/batch_instance_10k.csv",
        batch_task_path="fixtures/got/batch_task_10k.csv",
        write_path="../../out/clean/batch_fail_test",
        name_node=""
    )


class TestFailedInstances:
    def test_run(self, fi, spark_session):
        # given
        test_write = fi.get_write_path()
        no_part = 1
        fi.run(no_part=no_part)
        spark_session.stop()
        assert exists(test_write)
