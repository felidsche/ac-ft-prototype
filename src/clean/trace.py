from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

from src.clean.schema import *


def clean(batch_tasks: DataFrame, batch_instances: DataFrame) -> DataFrame:
    """
    :param batch_tasks: the raw cluster trace tasks
    :param batch_instances: the raw cluster trace instances of a task
    :return: the clean cluster trace
    """
    # _add_map_reduce() is called first because we need the field `task_id_num` in _add_job_exec()
    batch_tasks = _add_map_reduce(batch_tasks)
    batch_tasks = _edit_recurring(batch_tasks)
    batch_tasks = _add_mtts(batch_tasks)
    batch_tasks = _add_ttr(batch_tasks)
    batch_tasks = _add_reduce_checkpoint_baseline(batch_tasks)
    batch_tasks = _add_quantiles_checkpoint_baseline(batch_tasks)
    batch_instances = _add_labels(batch_instances)
    batch_jobs = _join_instance_data(batch_tasks, batch_instances)

    return batch_jobs


def _edit_recurring(batch_tasks: DataFrame, sched_intv_dev: float = 0.05) -> DataFrame:
    """
    see Section 6.2 of Tian et al. 2018
    required for _add_mtts(), _add_ttrs()
    :type sched_intv_dev: the deviation of a scheduling interval (in%) that is accepted
    :param batch_tasks: the raw cluster trace
    :return: the cluster trace with 4 additional columns for recurring batch jobs
    """
    # TODO: refactor this in 3 smaller functions
    batch_tasks = (
        batch_tasks
            # add start time of the earlierst (start_time) task_name per job_name
            .withColumn(
            "earliest",
            F.first(F.col("start_time")).over(Window.partitionBy("job_name", "task_name").orderBy("task_id_num"))
        )
            # obtain scheduling interval using the earliest start time
            # NOTE: null means this job was executed only once (see test__add_job_exec)
            .withColumn(
            "sched_intv",
            F.when(
                # Check if there is a next job_name
                F.isnull(F.lead("earliest").over(Window.partitionBy("task_name").orderBy("earliest"))),
                # if not, use the previous job_name to calculate `sched_intv` as below
                F.abs(F.lag("earliest").over(Window.partitionBy("task_name").orderBy("earliest")) - F.col("earliest"))
            ).otherwise(
                # calculate `sched_intv` as the difference of the earliest start times of two consecutive jobs
                F.abs(F.lead("earliest").over(Window.partitionBy("task_name").orderBy("earliest")) - F.col("earliest"))
            )
        )
            .withColumn(
            "recurrent",
            F.when(
                # Check if there is a previous value of `sched_intv`
                (
                    F.isnull(F.lag("sched_intv").over(Window.partitionBy("task_name").orderBy("earliest")))
                ) & (
                    # if not, use the next job_name to determine `recurrent` as below
                        (F.lead("sched_intv").over(Window.partitionBy("task_name").orderBy("earliest")) / F.col(
                            "sched_intv") / 100) <= sched_intv_dev
                ),
                1
            ).otherwise(
                F.when(
                    # if the previous value of `sched_intv` is <= {sched_intv_dev} it is a periodic schedule
                    (F.lag("sched_intv").over(Window.partitionBy("task_name").orderBy("earliest")) / F.col(
                        "sched_intv") / 100) <= sched_intv_dev,
                    1
                ).otherwise(None)
            )
        )
            .withColumn("job_exec", F.sum("recurrent").over(Window.partitionBy("task_name").orderBy("earliest")))
            # artificially create one `logical_job_name` value over all executions of the same batch job for MTTS and TTRS
            .withColumn(
            "logical_job_name",
            F.when(
                F.col("recurrent") == 1,
                F.concat(F.lit("L_"), F.first("job_name").over(Window.partitionBy("task_name")))
            ).otherwise(None)
        )
            .drop("recurrent")
    )
    return batch_tasks


def _add_map_reduce(batch_tasks: DataFrame) -> DataFrame:
    """
    add `map_reduce` column: _"m"_ for Map-task, _"r"_ for Reduce-Task
    see Algorithm 1 of Tian et al. 2018
    :return:
    """
    # create helper columns
    batch_tasks = (
        batch_tasks
            # create an array of the task's DAG
            .withColumn("task_dag", F.split("task_name", r"\_", limit=-1))
            # count earlier tasks in the DAG, that this task is dependent on, as parents
            .withColumn("parents", F.size(F.col("task_dag")) - 1)
            .withColumn("task_id", F.col("task_dag")[0])
            # get only the digits from the first part
            .withColumn("task_id_num", F.regexp_extract(F.col("task_id"), r"\d+", 0))
            # we need to cast to be able to order by this column
            .withColumn("task_id_num", F.col("task_id_num").cast(T.IntegerType()))
            # creates an array of the task's DAG and the next task's DAG
            .withColumn("job_dag",
                        F.collect_list("task_dag").over(Window.partitionBy("job_name").orderBy("task_id_num")))
            # adds the full job DAG to each task's row
            .withColumn("job_dag", F.last("job_dag").over(Window.partitionBy("job_name")))
            # remove one level in the nested list to get a level1 list
            .withColumn("job_dag", F.flatten("job_dag"))
            # cross product of all rows with each element in the job's DAG
            .withColumn("job_dag_exploded", F.explode("job_dag"))
            # check for each row if it's a match
            .withColumn("job_dag_exploded", (F.col("job_dag_exploded") == F.col("task_id_num")).cast(T.IntegerType()))
            # go back to the normal aggregation level
            .groupBy(
            ["task_name", "task_id_num", "instance_num", "job_name", "task_type", "status", "start_time", "end_time",
             "plan_cpu",
             "plan_mem", "parents"])
            # count the matches
            # (Note: there are no "self-counts" b.c. we look for matches in "task_id_num" and not in "task_id"
            .agg(F.sum("job_dag_exploded").alias("children"))
    )

    batch_tasks = (
        batch_tasks
            .withColumn(
            "map_reduce",
            F.when(
                (F.col("parents") < 2) |
                (
                        (F.col("parents") != 0) & (F.col("children") >= 2)
                ),
                "m"
            ).when(
                (F.col("children") < 2) |
                (
                        (F.col("children") != 0) & (F.col("parents") >= 2)
                ),
                "r"
            ).when(F.col("task_name").contains(F.lit("task_")), None).otherwise(None)
        )
    )

    batch_tasks = batch_tasks.drop("parents", "children")
    # TODO: check the distribution of "m" and "r" tasks
    return batch_tasks


def _add_mtts(batch_tasks: DataFrame) -> DataFrame:
    """
    calculate MTTS: _"Mean Time To Success"_ of a task over multiple executions in different jobs
    :return: the cluster trace with 4 additional columns for MTTS
    """
    batch_tasks = (
        batch_tasks
            .withColumn(
            "latest",
            F.max("end_time").over(Window.partitionBy("task_name", "job_name").orderBy("task_id_num"))
        )
            .withColumn("task_duration", F.col("latest") - F.col("earliest"))
            # prevent negative `task_duration` and `tts_task` by setting it to None
            .withColumn(
            "task_duration",
            F.when(
                (
                        (F.col("status") == "Running") | (F.col("status") == "Terminated")
                ) &
                (F.col("end_time") == 0),
                None
            ).otherwise(F.col("task_duration"))
        )
            .withColumn("tts_task", F.when(F.col("status") == "Terminated", F.col("task_duration")).otherwise(None))
            .withColumn(
            "mtts_task",
            F.mean("tts_task").over(Window.partitionBy("logical_job_name", "task_name"))
        )
    )
    return batch_tasks


def _add_ttr(batch_tasks: DataFrame) -> DataFrame:
    """
    add TTR: _"Time To Recover"_ of a task over multiple executions in different jobs
    (logical)_job_name with many failed task executions: L_j_1661 (failed jobs: j_2085966, j_4119616, j_132937, j_3442245)  (task: J7_1_2_6), L_j_30451, L_j_10678
    :return:
    """
    batch_tasks = (
        batch_tasks
            .withColumn("ttf_task", F.when(F.col("status") == "Failed", F.col("task_duration")).otherwise(None))
            .withColumn(
            "ttr_task",
            F.when(
                # previous task execution "Failed"
                (F.lag("status").over(
                    Window.partitionBy("logical_job_name", "task_name").orderBy("earliest")) == "Failed") &
                # this task execution "Terminated"
                (F.col("status") == "Terminated"),
                # sum the task duration of the "Failed" and "Terminated" task execution
                F.lag("ttf_task").over(Window.partitionBy("logical_job_name", "task_name").orderBy("earliest")) + F.col(
                    "tts_task")
            ).otherwise(None)  # task not recovered
        )
    )

    return batch_tasks


def _add_labels(batch_instances: DataFrame) -> DataFrame:
    """
    adds the `labels` column with the binary task classification label to the DataFrame
    :param batch_instances: the batch instances data
    :return: the cleaned cluster trace with the label
    """
    batch_instances = (
        batch_instances
            .withColumn("labels", F.when(F.col("status") == "Failed", 1).otherwise(0))
    )
    return batch_instances


def _add_reduce_checkpoint_baseline(batch_tasks: DataFrame) -> DataFrame:
    """
    adds the `reduce_checkpoint` (int) column: 1 -> checkpoint, 0 -> no checkpoint
    baselines:
        "reduce" -> checkpoint baseline that checkpoints the first "reduce" task per job exec
        "dag" -> checkpoint baseline that checkpoints 1 task in the first 1/3, and 1 task in the 2/3 of the job DAG
    :param batch_tasks: the cleaned cluster trace
    :return: the cluster trace with the `reduce_checkpoint` column
    """
    batch_tasks = (
        batch_tasks
            # helper column for reduce_count, we use None to ignore it later
            .withColumn("is_reduce", F.when(F.col("map_reduce") == "r", 1).otherwise(None))
            # helper column for reduce_checkpoint
            .withColumn(
            "reduce_no",
            F.sum(F.col("is_reduce")).over(Window.partitionBy("logical_job_name", "job_name").orderBy("task_id_num"))
        )
            .withColumn(
            "reduce_checkpoint",
            F.when(
                # since each group has only 1 reduce, we want to set the 1 to this row
                (F.col("reduce_no") == 1) &
                (F.col("is_reduce") == 1),
                1
            ).otherwise(None)
        )
            .drop("is_reduce", "reduce_no")

    )
    return batch_tasks


def _add_quantiles_checkpoint_baseline(batch_tasks: DataFrame) -> DataFrame:
    """
    adds the {no}_quant_checkpoint` (int) columns: 1 -> checkpoint, 0 -> no checkpoint
    baselines:
        "quantiles" -> checkpoint baseline that checkpoints 1 task at the 25% quantile, and 1 task at the 75% quantile of the job DAG
    :param batch_tasks: the cleaned cluster trace
    :return: the cluster trace with the `dag_checkpoint` column
    """
    batch_tasks = (
        batch_tasks
            .withColumn(
            "quantiles",
            F.percentile_approx(F.col("task_id_num"), percentage=[0.25, 0.75]).over(
                Window.partitionBy("logical_job_name", "job_name"))
        )
            # explode the array column
            .withColumn("second_quantile", F.col("quantiles")[0])
            .withColumn("third_quantile", F.col("quantiles")[1])
            # mark the quantile rows
            .withColumn("second_quant_checkpoint",
                        F.when(F.col("task_id_num") == F.col("second_quantile"), 1).otherwise(None))
            .withColumn("third_quant_checkpoint",
                        F.when(F.col("task_id_num") == F.col("third_quantile"), 1).otherwise(None))
            .drop("quantiles", "second_quantile", "third_quantile")
    )
    return batch_tasks


def _join_instance_data(batch_tasks: DataFrame, batch_instances: DataFrame) -> DataFrame:
    # rename duplicate cols
    batch_instances = (
        batch_instances
            .withColumnRenamed("status", "instance_status")
            .withColumnRenamed("task_type", "instance_task_type")
            .withColumnRenamed("start_time", "instance_start_time")
            .withColumnRenamed("end_time", "instance_end_time")
    )
    # batch_tasks is too big for broadcast join
    batch_jobs = (
        batch_tasks
            .join(batch_instances,
                  on=["task_name", "job_name"],
                  how="inner")
    )
    return batch_jobs


if __name__ == "__main__":
    # final Spark job: spark-c52b91cea91142caa30b8593007a2430
    # final clean Dataset: batch_jobs_clean_03inst_1task_00015S_1F
    name_node = "ip:port"

    batch_task_path = f"hdfs://{name_node}/spark/alibaba2018-trace/batch_task.tar.gz"
    batch_instance_path = f"hdfs://{name_node}/spark/alibaba2018-trace/batch_instance.tar.gz"
    app_name = "clean Alibaba 2018 cluster trace"

    spark_session = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

    # sample_seed = 55
    # sample_seed = 36
    sample_seed = 61

    batch_tasks = spark_session.read.csv(
        path=batch_task_path,
        header=False,
        schema=batch_task_schema
    )

    batch_instances = spark_session.read.csv(
        path=batch_instance_path,
        header=False,
        schema=batch_instance_schema
    )

    batch_jobs = clean(batch_tasks, batch_instances)
    # sample for X% failed task instance executions and Y% of the successful ones
    # this should match the success/fail rate and lead to more balanced classes
    # the stratified sample is according to tab:traceComparison
    # 0.006 / 4 bc right now 1 is 25%
    batch_jobs = batch_jobs.sampleBy("labels", fractions={0: 0.0015, 1: 1}, seed=sample_seed)
    # expect 720MB -> 6 partitions at 128MB
    batch_jobs.repartition(6).write.csv(
        f"hdfs://{name_node}/spark/alibaba2018-trace/out/batch_jobs_clean_00015S_1F",
        mode="overwrite",
        header=True,
        compression="gzip"
    )

    spark_session.stop()
