import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark_test import assert_pyspark_df_equal

from src.clean.schema import batch_task_schema
from src.clean.trace import _add_map_reduce, _edit_recurring, _add_mtts, _add_ttr, _add_labels, \
    _add_reduce_checkpoint_baseline, _add_quantiles_checkpoint_baseline, _join_instance_data


@pytest.fixture
def spark_session():
    master = "local[4]"  # use 4 cores
    app_name = "test clean Alibaba 2018 cluster trace"
    spark_session = SparkSession.builder \
        .master(master) \
        .appName(app_name) \
        .getOrCreate()
    return spark_session


class TestCleanTrace:

    def test__edit_recurring_4_jobs(self, spark_session):
        """
        How the fixture was extracted:
        `batch_tasks.groupby("task_name").agg(F.approx_count_distinct("status").alias("status_count")).sort("status_count", ascending=False)`
        from this list, we get the first 4 `job_name` values (by start_time) to select the full job and not just this one task
        `batch_tasks.filter(F.col("job_name").isin(""j_3353366", "j_3794102", "j_455604", "j_3364660")).sort("task_id_num").show(150)`
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_job_exec_4_jobs.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _edit_recurring(batch_tasks)

        expected_path = "fixtures/expected/add_job_exec_4_jobs.csv"
        expected = spark_session.read.csv(expected_path, header=True, inferSchema=True)

        new_col_set = {"earliest", "sched_intv", "job_exec", "logical_job_name"}
        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        assert_pyspark_df_equal(got, expected, check_columns_in_order=False, check_dtype=False, order_by=["task_name"])

    def test__edit_recurring(self, spark_session):
        """
        the fixture is job_name = j_3985826 (part of add_job_exec_4_jobs.csv)
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_job_exec_j_3353366.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _edit_recurring(batch_tasks)
        new_col_set = {"earliest", "sched_intv", "job_exec", "logical_job_name"}

        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        # assert no values for a single execution of a job in the three added columns
        assert got.select(F.col("earliest")).distinct().collect()[0][0] == 88863
        assert got.select(F.col("sched_intv")).distinct().collect()[0][0] is None
        assert got.select(F.col("logical_job_name")).distinct().collect()[0][0] is None
        assert got.select(F.col("job_exec")).distinct().collect()[0][0] is None

    def test__add_map_reduce(self, spark_session):
        """
        the fixture is job_name = j_3985826 (see Section 4.1 of Tian et al. 2018)
        you can check your result against Figure 4
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_map_reduce_j_3985826.csv"
        batch_task = spark_session.read.csv(fixture_path, header=True, schema=batch_task_schema)

        got = _add_map_reduce(batch_task)
        assert "map_reduce", "task_id_num" in got.columns
        assert (len(batch_task.columns) + 2) == len(got.columns)

        expected_path = "fixtures/expected/add_map_reduce_j_3985826.csv"
        expected = spark_session.read.csv(expected_path, header=True, inferSchema=True)

        assert_pyspark_df_equal(got, expected, check_columns_in_order=False, check_dtype=False, order_by=["task_name"])

    def test__add_mtts(self, spark_session):
        """
        the fixture is the same as add_map_reduce_j_3985826.csv because it includes the preceeding steps
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_mtts_4_jobs.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _add_mtts(batch_tasks)
        new_col_set = {"latest", "task_duration", "tts_task", "mtts_task"}

        expected_path = "fixtures/expected/add_mtts_4_jobs.csv"
        expected = spark_session.read.csv(expected_path, header=True, inferSchema=True)

        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        assert_pyspark_df_equal(got, expected, check_columns_in_order=False, check_dtype=False, order_by=["task_name"])

    def test__add_ttr(self, spark_session):
        """
        the fixture consists of 4 job runs where task_name J7_1_2_6 Terminated, Failed, Terminated, Failed
        te 4 job_name (ordered by start_time): j_1661, j_2085966 (Failed), j_105678, j_132937 (Failed)
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_ttr_4_jobs_1_recovery.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _add_ttr(batch_tasks)
        new_col_set = {"ttf_task", "ttr_task"}

        expected_path = "fixtures/expected/add_ttr_4_jobs_1_recovery.csv"
        expected = spark_session.read.csv(expected_path, header=True, inferSchema=True)

        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        assert_pyspark_df_equal(got, expected, check_columns_in_order=False, check_dtype=False, order_by=["task_name"])

    def test__add_labels(self, spark_session):
        """
        reuses the fixture `add_ttr_4_jobs_1_recovery.csv` to distinguish between the label for different statuses
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_ttr_4_jobs_1_recovery.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)

        got = _add_labels(batch_tasks)
        new_col_set = {"labels"}

        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        # assert only "Terminated", "Waiting", "Running" and "Failed"
        assert got.select("status").distinct().count() == 4
        # because we do binary classification there can only be two labels
        assert got.select("labels").distinct().count() == 2
        # 1 is the only accepted label
        assert got.filter(F.col("status") == "Failed").select("labels").distinct().count() == 1

    def test__add_reduce_checkpoint_baseline(self, spark_session):
        """
        the fixture contains 2 "r" tasks
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_checkpoint_baselines_j_3985826.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        got = _add_reduce_checkpoint_baseline(batch_tasks)
        new_col_set = {"reduce_checkpoint"}

        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        assert got.select("reduce_checkpoint").filter(F.col("reduce_checkpoint") == 1).count() == 1

    def test__add_quantiles_checkpoint_baseline(self, spark_session):
        """
        the fixture `got/add_checkpoint_baselines_j_3985826.csv` is a copy of `expected/add_map_reduce_j_3985826.csv`
        it contains a DAG of 16 tasks
        :param spark_session: the spark_session fixture
        """
        fixture_path = "fixtures/got/add_checkpoint_baselines_j_3985826.csv"
        batch_tasks = spark_session.read.csv(fixture_path, header=True, inferSchema=True)
        got = _add_quantiles_checkpoint_baseline(batch_tasks)
        new_col_set = {"second_quant_checkpoint", "third_quant_checkpoint"}

        assert new_col_set.issubset(got.columns)
        assert (len(batch_tasks.columns) + len(new_col_set)) == len(got.columns)
        # there can only be 1 of each quantiles per job execution
        assert got.select("second_quant_checkpoint").filter(F.col("second_quant_checkpoint") == 1).count() == 1
        assert got.select("third_quant_checkpoint").filter(F.col("second_quant_checkpoint") == 1).count() == 1

    def test__join_instance_data(self, spark_session):
        """
        assures that all values stay after joining
        :param spark_session: the spark_session fixture
        """
        task_fixture_path = "fixtures/got/batch_task_j_138474_M10.csv"
        instance_fixture_path = "fixtures/got/batch_instance_j_138474_M10.csv"
        batch_tasks = spark_session.read.csv(task_fixture_path, header=True, inferSchema=True)
        batch_instances = spark_session.read.csv(instance_fixture_path, header=True, inferSchema=True)
        got = _join_instance_data(batch_tasks, batch_instances)
        new_col_set = set(batch_instances.columns)

        assert new_col_set.issubset(got.columns)
        # we assert that `instance_num` rows were added
        assert got.count() == batch_tasks.select("instance_num").collect()[0][0]
