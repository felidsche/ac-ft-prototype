from pyspark.sql import types as T

# see https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt
batch_task_schema = T.StructType([
    # task name indicates the DAG information, see the explanation of batch workloads
    T.StructField("task_name", T.StringType(), nullable=False),  # task name. unique within a job
    # NOTE: tas_name doesn't change throughout different job executions
    T.StructField("instance_num", T.IntegerType(), nullable=True),  # number of instances for the task
    T.StructField("job_name", T.StringType(), nullable=True),
    T.StructField("task_type", T.IntegerType(), nullable=True),  # values 1-12, meaning unknown
    # Task states includes Ready | Waiting | Running | Terminated | Failed | Cancelled
    T.StructField("status", T.StringType(), nullable=True), # task status
    T.StructField("start_time", T.LongType(), nullable=True),  # start time of the task (in seconds since the start of the trace (0))
    T.StructField("end_time", T.LongType(), nullable=True),  # end of time the task (in seconds since the start of the trace (0))
    T.StructField("plan_cpu", T.DoubleType(), nullable=True),  # number of cpu needed by the task, 100 is 1 core
    T.StructField("plan_mem", T.DoubleType(), nullable=False) # normalized memory size, [0, 100]
])

batch_instance_schema = T.StructType([
    T.StructField("instance_name", T.StringType(), nullable=True), # instance name of the instance
    T.StructField("task_name", T.StringType(), nullable=True),  # name of task to which the instance belong
    T.StructField("job_name", T.StringType(), nullable=True),  # name of the job to which the instance belongs
    T.StructField("task_type", T.StringType(), nullable=True),  # values 1-12, meaning unknown
    # instance states includes Ready | Waiting | Running | Terminated | Failed | Cancelled
    T.StructField("status", T.StringType(), nullable=True), # instance status
    T.StructField("start_time", T.LongType(), nullable=True),  # start time of the instance (in seconds since the start of the trace (0))
    T.StructField("end_time", T.LongType(), nullable=True),  # end of time the instance (in seconds since the start of the trace (0))
    T.StructField("machine_id", T.StringType(), nullable=False),  # uid of host machine of the instance
    T.StructField("seq_no", T.LongType(), nullable=True), # sequence number of this instance
    T.StructField("total_seq_no", T.LongType(), nullable=True), # total sequence number of this instance
    T.StructField("cpu_avg", T.DoubleType(), nullable=True),  # average cpu used by the instance, 100 is 1 core
    T.StructField("cpu_max", T.DoubleType(), nullable=True),  # max cpu used by the instance, 100 is 1 core
    T.StructField("mem_avg", T.DoubleType(), nullable=True),  # average memory used by the instance (normalized)
    T.StructField("mem_max", T.DoubleType(), nullable=True),  # max memory used by the instance (normalized, [0, 100])
])