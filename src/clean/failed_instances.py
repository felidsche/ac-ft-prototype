""" python script to extract and clean only the failed instances from the Alibaba 2018 cluster trace """
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T, functions as F

batch_task_schema = T.StructType([
    # task name indicates the DAG information, see the explanation of batch workloads
    T.StructField("task_name", T.StringType(), nullable=False),  # task name. unique within a job
    # NOTE: tas_name doesn't change throughout different job executions
    T.StructField("instance_num", T.IntegerType(), nullable=True),  # number of instances for the task
    T.StructField("job_name", T.StringType(), nullable=True),
    T.StructField("task_type", T.IntegerType(), nullable=True),  # values 1-12, meaning unknown
    # Task states includes Ready | Waiting | Running | Terminated | Failed | Cancelled
    T.StructField("status", T.StringType(), nullable=True),  # task status
    T.StructField("start_time", T.LongType(), nullable=True),
    # start time of the task (in seconds since the start of the trace (0))
    T.StructField("end_time", T.LongType(), nullable=True),
    # end of time the task (in seconds since the start of the trace (0))
    T.StructField("plan_cpu", T.DoubleType(), nullable=True),  # number of cpu needed by the task, 100 is 1 core
    T.StructField("plan_mem", T.DoubleType(), nullable=False)  # normalized memory size, [0, 100]
])

batch_instance_schema = T.StructType([
    T.StructField("instance_name", T.StringType(), nullable=True),  # instance name of the instance
    T.StructField("task_name", T.StringType(), nullable=True),  # name of task to which the instance belong
    T.StructField("job_name", T.StringType(), nullable=True),  # name of the job to which the instance belongs
    # 1-12
    T.StructField("task_type_inst", T.StringType(), nullable=True),
    # instance states includes Ready | Waiting | Running | Terminated | Failed | Cancelled
    T.StructField("status_inst", T.StringType(), nullable=True),  # instance status
    T.StructField("start_time_inst", T.LongType(), nullable=True),  # start time of the instance
    T.StructField("end_time_inst", T.LongType(), nullable=True),  # end of time the instance
    T.StructField("machine_id_inst", T.StringType(), nullable=False),  # uid of host machine of the instance
    T.StructField("seq_no_inst", T.LongType(), nullable=True),  # sequence number of this instance
    T.StructField("total_seq_no_inst", T.LongType(), nullable=True),  # total sequence number of this instance
    T.StructField("cpu_avg_inst", T.DoubleType(), nullable=True),  # average cpu used by the instance, 100 is 1 core
    T.StructField("cpu_max_inst", T.DoubleType(), nullable=True),  # max cpu used by the instance, 100 is 1 core
    T.StructField("mem_avg_inst", T.DoubleType(), nullable=True),  # average memory used by the instance (normalized)
    T.StructField("mem_max_inst", T.DoubleType(), nullable=True),
    # max memory used by the instance (normalized, [0, 100])
])


def persist_df(batch_fail: DataFrame, write_path: str, no_part: int) -> None:
    batch_fail.repartition(no_part).write.csv(
        write_path,
        mode="overwrite",
        header=True,
        compression="gzip"
    )
    return None


def clean_dfs(batch_tasks: DataFrame, batch_instances: DataFrame):
    """
    :param batch_tasks:
    :param batch_instances:
    :return:
    """
    # filter for failed events
    batch_tasks_fail = batch_tasks.filter(F.col("status") == "Failed")
    batch_instances_fail = batch_instances.filter(F.col("status_inst") == "Failed")
    # join
    batch_fail = (
        batch_instances_fail
            .join(batch_tasks_fail,
                  on=["task_name", "job_name"],
                  how="inner")
    )
    # commented out bc not possible on the cluster
    """
    # UDF for converting column type from vector to double type
    unlist = udf(lambda x: round(float(list(x)[0]), 3), T.DoubleType())

    scale_imp_cols = ["plan_cpu", "cpu_avg_inst", "cpu_max_inst", "plan_mem", "mem_avg_inst", "mem_max_inst"]
    for i in scale_imp_cols:
        # "Imputer" Transformatuon
        imputer = Imputer(inputCol=i, outputCol=i + "_Imp")

        # VectorAssembler Transformation - Converting column to vector type
        assembler = VectorAssembler(inputCols=[i + "_Imp"], outputCol=i + "_Vect", handleInvalid="keep")

        # MinMaxScaler Transformation
        scaler = MinMaxScaler(inputCol=i + "_Vect", outputCol=i + "_Scaled")

        pipeline = Pipeline(stages=[imputer, assembler, scaler])

        # Fitting pipeline on dataframe
        batch_fail = pipeline.fit(batch_fail).transform(batch_fail).withColumn(i + "_Scaled",
                                                                               unlist(i + "_Scaled")).drop(
            i + "_Vect")
    """
    return batch_fail


class FailedInstances:
    def __init__(self, spark_session: SparkSession, name_node: str, batch_task_path: str, batch_instance_path: str,
                 write_path: str):
        self.spark_session = spark_session
        self.name_node = name_node
        self.batch_task_path = batch_task_path
        self.batch_instance_path = batch_instance_path
        self.write_path = write_path

    def get_write_path(self) -> str:
        return self.write_path

    def create_dfs(self):
        """
        
        :return: 
        """
        # read from disk

        batch_tasks = self.spark_session.read.csv(
            path=self.batch_task_path,
            header=False,
            schema=batch_task_schema
        )

        batch_instances = self.spark_session.read.csv(
            path=self.batch_instance_path,
            header=False,
            schema=batch_instance_schema
        )

        return batch_tasks, batch_instances

    def run(self, no_part: int) -> None:
        """
        wrapper function to run the script
        :return: None
        """
        batch_tasks, batch_instances = self.create_dfs()
        batch_fail = clean_dfs(batch_tasks, batch_instances)
        persist_df(batch_fail, no_part=no_part, write_path=self.write_path)
        return None


if __name__ == '__main__':
    NAMENODE = "ip:port"
    task_path = f"hdfs://{NAMENODE}/spark/alibaba2018-trace/batch_task.tar.gz"
    inst_path = f"hdfs://{NAMENODE}/spark/alibaba2018-trace/batch_instance.tar.gz"
    write = f"hdfs://{NAMENODE}/spark/alibaba2018-trace/out/batch_fail"
    app_name = "clean failed Alibaba 2018 cluster trace events"
    no_part = 1

    session = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

    fi = FailedInstances(
        session,
        NAMENODE,
        task_path,
        inst_path,
        write
    )
    fi.run(no_part=no_part)
    # takes 30 min