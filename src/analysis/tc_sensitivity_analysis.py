""" class to do the sensitivity analysis of the time for a checkpoint """

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F, types as T

sty = "seaborn"
mpl.style.use(sty)


def _add_tc(prediction_output: DataFrame, tw_steps: list, io_ratio: float):
    """
    adds the column `tc` (array<string>) with the time for a checkpoint (in min)
    :param prediction_output: the prediction_output
    :return: the prediction_output with the `tc` column
    """

    # the checkpoint time is reading and writing, so we divide the writing values by a fixed, measured IO ratio
    tc_vals = [F.lit('{0:.2g}'.format(tw + tr)) for tw, tr in zip(tw_steps, map(lambda x: (x / io_ratio), tw_steps))]
    prediction_output = (
        prediction_output
            .withColumn(
            "tc",
            F.when(
                # when at least one checkpoint model wants to checkpoint
                (F.col("reduce_checkpoint").isNotNull()) |
                (F.col("second_quant_checkpoint").isNotNull()) |
                (F.col("third_quant_checkpoint").isNotNull()) |
                (F.col("prediction") == 1),
                # array() only works with list of lit() strings
                F.array(tc_vals)
            ).otherwise(None)

        )
    )
    return prediction_output


def _add_td(prediction_output):
    """
    adds the `td` (float) column that contains $T_d = MTTS - TTR$ (time difference)
    Note: $T_d >= 0$ mean there is no benefit of using checkpoints
    :param prediction_output: the output of the model prediction
    :return: the output of the model prediction with the `td` column
    """
    prediction_output = (
        prediction_output
            .withColumn(
            "td",
            F.col("mtts_task") - F.col("ttr_task")
        )
    )

    return prediction_output


def _add_reduce_co(prediction_output):
    prediction_output = (
        prediction_output
            .withColumn(
            "reduce_co",
            F.when(
                F.col("reduce_checkpoint").isNotNull(),
                F.transform(F.col("tc"), lambda x: F.col("td") + x)
            ).otherwise(None)
        )
    )
    return prediction_output


def _add_lower_quart_co(prediction_output):
    prediction_output = (
        prediction_output
            .withColumn(
            "second_quant_co",
            F.when(
                F.col("second_quant_checkpoint").isNotNull(),
                F.transform(F.col("tc"), lambda x: F.col("td") + x)
            ).otherwise(None)
        )
    )
    return prediction_output


def _add_upper_quart_co(prediction_output):
    prediction_output = (
        prediction_output
            .withColumn(
            "third_quant_co",
            F.when(
                F.col("third_quant_checkpoint").isNotNull(),
                F.transform(F.col("tc"), lambda x: F.col("td") + x)
            ).otherwise(None)
        )
    )
    return prediction_output


def _add_adaptive_co(prediction_output: DataFrame, tp: int) -> DataFrame:
    """

    :param prediction_output: the validation DataFrame
    :param tp: the prediction time in minutes
    :return:
    """
    prediction_output = (
        prediction_output
            .withColumn(
            "adaptive_co",
            F.when(
                # the prediction does not have to be true bc either way the checkpoint overhead will be induced
                (F.col("prediction") == 1),  # &
                # (F.col("correct_predictions") == True),
                F.transform(F.col("tc"), lambda x: (F.col("td") + F.lit(tp).cast(T.FloatType())) + x)
            ).otherwise(None)
        )
    )
    return prediction_output


def _add_co(prediction_output, tp):
    """
    adds a checkpoint overhead `co_{checkpoint_model}` (float) column for each checkpoint model
    $co = T_d + T_c$
    :param prediction_output: the prediction_output
    :return: the prediction_output with the `co` column
    """
    prediction_output = _add_reduce_co(prediction_output)
    prediction_output = _add_lower_quart_co(prediction_output)
    prediction_output = _add_upper_quart_co(prediction_output)
    prediction_output = _add_adaptive_co(prediction_output, tp)
    return prediction_output


def _get_agg_co_per_tc_model(prediction_output: DataFrame, acc: int, metric="mean"):
    """
    creates an aggregated version of the cluster trace by tc and checkpoint model
    :param acc: the accuracy for the median calc to not run OOM
    :param metric: "mean" or "median" the metric to calculate for co
    :param prediction_output: the cluster trace
    :return: the aggregated cluster trace
    """

    if metric == "median":
        reduce_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_reduce_co", F.arrays_zip("tc", "reduce_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_reduce_co", F.explode("zip_tc_reduce_co"))
                .select("*", F.col("zip_tc_reduce_co.tc").alias("zip_tc"),
                        F.col("zip_tc_reduce_co.reduce_co").alias("zip_reduce_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.percentile_approx(F.col("zip_reduce_co"), [0.5], accuracy=acc)[0].alias("median_reduce_co") /
                        F.mean(F.col("task_duration")).alias("mean_task_duration_reduce_co")
                ).alias("median_reduce_co_perc")
            )
        )

        second_quant_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_second_quant", F.arrays_zip("tc", "second_quant_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_second_quant", F.explode("zip_tc_second_quant"))
                .select("*", F.col("zip_tc_second_quant.tc").alias("zip_tc"),
                        F.col("zip_tc_second_quant.second_quant_co").alias("zip_second_quant_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.percentile_approx(F.col("zip_second_quant_co"), [0.5], accuracy=acc)[0].alias(
                            "median_second_quant_co") /
                        F.mean(F.col("task_duration")).alias("mean_task_duration_second_quant_co")
                ).alias("median_second_quant_co_perc")
            )
        )

        third_quant_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_third_quant", F.arrays_zip("tc", "third_quant_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_third_quant", F.explode("zip_tc_third_quant"))
                .select("*", F.col("zip_tc_third_quant.tc").alias("zip_tc"),
                        F.col("zip_tc_third_quant.third_quant_co").alias("zip_third_quant_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.percentile_approx(F.col("zip_third_quant_co"), [0.5], accuracy=acc)[0].alias(
                            "median_third_quant_co") /
                        F.mean(F.col("task_duration")).alias("mean_task_duration_third_quant_co")
                ).alias("median_third_quant_co_perc")
            )
        )

        adaptive_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_adaptive", F.arrays_zip("tc", "adaptive_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_adaptive", F.explode("zip_tc_adaptive"))
                .select("*", F.col("zip_tc_adaptive.tc").alias("zip_tc"),
                        F.col("zip_tc_adaptive.adaptive_co").alias("zip_adaptive_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.percentile_approx(F.col("zip_adaptive_co"), [0.5], accuracy=acc)[0].alias(
                            "median_adaptive_co") /
                        F.mean(F.col("task_duration").alias("mean_task_duration_adaptive_co"))
                ).alias("median_adaptive_co_perc")
            )
        )

        median_co_per_tc_model = (
            reduce_co_tc_agg
                .join(second_quant_co_tc_agg, how="inner", on="zip_tc")
                .join(third_quant_co_tc_agg, how="inner", on="zip_tc")
                .join(adaptive_co_tc_agg, how="inner", on="zip_tc")
        )

        return median_co_per_tc_model
    else:
        reduce_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_reduce_co", F.arrays_zip("tc", "reduce_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_reduce_co", F.explode("zip_tc_reduce_co"))
                .select("*", F.col("zip_tc_reduce_co.tc").alias("zip_tc"),
                        F.col("zip_tc_reduce_co.reduce_co").alias("zip_reduce_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.mean(F.col("zip_reduce_co")).alias("avg_reduce_co") /
                        F.mean(F.col("task_duration")).alias("mean_task_duration_reduce_co")
                ).alias("avg_reduce_co_perc")
            )
        )

        second_quant_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_second_quant", F.arrays_zip("tc", "second_quant_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_second_quant", F.explode("zip_tc_second_quant"))
                .select("*", F.col("zip_tc_second_quant.tc").alias("zip_tc"),
                        F.col("zip_tc_second_quant.second_quant_co").alias("zip_second_quant_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.mean(F.col("zip_second_quant_co")).alias("avg_second_quant_co") /
                        F.mean(F.col("task_duration")).alias("mean_task_duration_second_quant_co")
                ).alias("avg_second_quant_co_perc")
            )
        )

        third_quant_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_third_quant", F.arrays_zip("tc", "third_quant_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_third_quant", F.explode("zip_tc_third_quant"))
                .select("*", F.col("zip_tc_third_quant.tc").alias("zip_tc"),
                        F.col("zip_tc_third_quant.third_quant_co").alias("zip_third_quant_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.mean(F.col("zip_third_quant_co")).alias("avg_third_quant_co") /
                        F.mean(F.col("task_duration")).alias("mean_task_duration_third_quant_co")
                ).alias("avg_third_quant_co_perc")
            )
        )

        adaptive_co_tc_agg = (
            prediction_output
                # we zip to be able to explode only once
                .withColumn("zip_tc_adaptive", F.arrays_zip("tc", "adaptive_co"))
                # explode to get one row per tc and co
                .withColumn("zip_tc_adaptive", F.explode("zip_tc_adaptive"))
                .select("*", F.col("zip_tc_adaptive.tc").alias("zip_tc"),
                        F.col("zip_tc_adaptive.adaptive_co").alias("zip_adaptive_co"))
                .groupby("zip_tc")
                .agg(
                (
                        F.mean(F.col("zip_adaptive_co")).alias("avg_adaptive_co") /
                        F.mean(F.col("task_duration").alias("mean_task_duration_adaptive_co"))
                ).alias("avg_adaptive_co_perc")
            )
        )

        avg_co_per_tc_model = (
            reduce_co_tc_agg
                .join(second_quant_co_tc_agg, how="inner", on="zip_tc")
                .join(third_quant_co_tc_agg, how="inner", on="zip_tc")
                .join(adaptive_co_tc_agg, how="inner", on="zip_tc")
        )

        return avg_co_per_tc_model


def _plot_agg_co_per_tc_model(agg_co_per_tc_model: pd.DataFrame, filename: str, metric="avg"):
    """
    plots the avg(co) (in % compared to vanilla job runtime) of all 4 models as a bar chart
    for each model, there are 4 bars (1 for each tc value)
    :param metric: "avg" or "median" the metric to plot for co
    """
    # rename and cast for better labels in the legend and axis
    cols_dict = {f'{metric}_adaptive_co_perc': 'adaptive', f'{metric}_reduce_co_perc': 'reduce',
                 f"{metric}_second_quant_co_perc": "lowerQuartile",
                 f"{metric}_third_quant_co_perc": "upperQuartile"}
    agg_co_per_tc_model.rename(columns=cols_dict, inplace=True)
    # plot grouped bar chart
    plt.figure(figsize=(8, 6))

    # cast to float -> and set Tc as sorted index to sort x-axis in Barplot
    agg_co_per_tc_model['zip_tc'] = agg_co_per_tc_model['zip_tc'].astype(float)
    agg_co_per_tc_model['zip_tc'] = agg_co_per_tc_model['zip_tc'].astype(int)
    agg_co_per_tc_model.set_index("zip_tc", inplace=True)
    agg_co_per_tc_model.sort_index("index", inplace=True)

    agg_co_per_tc_model.plot(
        use_index=True,
        kind='bar',
        stacked=False,
        #title=f'{metric} $co$ (in % of the task duration) per $T_c$ and checkpoint strategy',
        grid=True,
        sort_columns=True
    )
    plt.ylabel(f"{metric.capitalize()} $co$ (in %)")
    plt.xlabel("$T_c$")
    plt.legend()
    #plt.tight_layout()

    # this is necessary to leave enough space for the legend
    if metric == "median":
        plt.ylim(-1.75, 0.0)
    else:
        plt.ylim(-2.3, 0.0)
    # this overwrites
    plt.savefig(f"../../out/analysis/sa/plots/bar/{filename}_{metric}_co.pdf", format="pdf")
    plt.show()


def _write_eval_df(prediction_output: DataFrame):
    """
    write an exploded DataFrame to analyse the checkpoint models
    :param prediction_output: the not yet exploded DataFrame
    """
    prediction_output = (
        prediction_output
            .withColumn("zip_tc_adaptive", F.arrays_zip("tc", "adaptive_co"))
            # explode to get one row per tc and co
            .withColumn("zip_tc_adaptive", F.explode("zip_tc_adaptive"))
            .select(
            "*",
            F.col("zip_tc_adaptive.tc").alias("zip_tc"),
            F.col("zip_tc_adaptive.adaptive_co").alias("zip_adaptive_co"))
            .toPandas()
            .to_csv(f"../../out/analysis/sa/eval/{MODEL_NAME}", compression="gzip", index=False)
    )
    pass


def _plot_agg(prediction_output: DataFrame, metric: str, filename: str, acc: int):
    """
    wrapper function to call the aggregation function for {metric} and create the barplot
    :param prediction_output: the eval DataFrame
    :param metric: "mean" or "median"
    """
    agg_co_per_tc_model = _get_agg_co_per_tc_model(prediction_output, acc=acc, metric=metric)
    agg_co_per_tc_model_pd = agg_co_per_tc_model.toPandas()
    _plot_agg_co_per_tc_model(agg_co_per_tc_model_pd, filename=filename, metric=metric)
    pass


def _plot_check_distr_model(prediction_output: DataFrame, filename: str):
    """
    plots a bar chart showing which checkpoint model checkpointed how many tasks (in %)
    :param prediction_output: the DataFrame
    :param filename: the name of the model
    """
    checkpoint_count_model = (
        prediction_output
            .select("task_name", "reduce_checkpoint", "second_quant_checkpoint", "third_quant_checkpoint", "prediction")
            .agg(
            F.sum("reduce_checkpoint").alias("sum_reduce_checkpoint"),
            F.count(F.col("task_name")).alias("task_count"),
            F.sum("second_quant_checkpoint").alias("sum_second_quant_checkpoint"),
            F.sum("third_quant_checkpoint").alias("sum_third_quant_checkpoint"),
            F.sum("prediction").alias("sum_adaptive")
        )
            .select(
            "sum_reduce_checkpoint", "sum_second_quant_checkpoint", "sum_third_quant_checkpoint", "sum_adaptive",
            "task_count"
        )
    )

    # the alias is set as the legend label
    checkpoint_perc_model = (
        checkpoint_count_model
            .select(
            (F.col("sum_reduce_checkpoint") / F.col("task_count")).alias("reduce"),
            (F.col("sum_second_quant_checkpoint") / F.col("task_count")).alias("lowerQuartile"),
            (F.col("sum_third_quant_checkpoint") / F.col("task_count")).alias("upperQuartile"),
            (F.col("sum_adaptive") / F.col("task_count")).alias("adaptive"),
        )
    )

    plt.figure(figsize=(8, 6))

    checkpoint_perc_model.toPandas().plot(
        use_index=True,
        kind='bar',
        #title='Percentage of checkpointed tasks per checkpoint strategy',
        grid=True,
        sort_columns=True
    )
    plt.ylabel("% of all tasks")
    plt.xticks([])
    plt.ylim(-2.0, 0.0)
    plt.xlabel("checkpoint model")
    plt.tight_layout()
    # this overwrites
    plt.savefig(f"../../out/analysis/sa/plots/bar/{filename}_perc_check.pdf", format="pdf")
    plt.show()
    pass


def _plot_check_distr_model_over_time(prediction_output: DataFrame, filename: str, days: int, label: int):
    """
    plots a bar and line chart showing which checkpoint model checkpointed how many of all successful, failed tasks (in %)
    over a number of days
    :param days: the number of days to plot
    :param label: wheter to plot for label 1 ("failed") or label 0 ("success")
    :param prediction_output: the DataFrame
    :param filename: the name of the model
    """

    # add day column to group and filter the data by day
    sec_per_day = 86400

    prediction_output = (
        prediction_output
            .withColumn(
            "day",
            (F.col("end_time") / sec_per_day).cast(T.IntegerType()),
        )
            .filter(
            # not equal because there is a day 0
            F.col("day") < days
        )
            .filter(
            F.col("labels") == label
        )
            .groupby("day")
            .agg(
            F.sum("reduce_checkpoint").alias("sum_reduce_checkpoint"),
            F.count(F.col("task_name")).alias("task_count"),
            F.sum("second_quant_checkpoint").alias("sum_second_quant_checkpoint"),
            F.sum("third_quant_checkpoint").alias("sum_third_quant_checkpoint"),
            F.sum("prediction").alias("sum_adaptive")
        )
            .select(
            F.col("day"),
            (F.col("sum_reduce_checkpoint") / F.col("task_count")).alias("reduce"),
            (F.col("sum_second_quant_checkpoint") / F.col("task_count")).alias("lowerQuartile"),
            (F.col("sum_third_quant_checkpoint") / F.col("task_count")).alias("upperQuartile"),
            (F.col("sum_adaptive") / F.col("task_count")).alias("adaptive")
        )
    )
    title_label = "failed" if label == 1 else "successful"
    plt.figure(figsize=(8, 6))
    prediction_output = prediction_output.toPandas().set_index(["day"])
    prediction_output.sort_index().plot(
        use_index=True,
        kind='bar',
        #title=f'Percentage of {title_label} tasks that were checkpointed over {days} days',
        grid=True,
        sort_columns=True
    )
    plt.ylabel(f"Proportion of {title_label} tasks")
    plt.ylim(0.0, 1.2)
    plt.xlabel("Day")
    plt.legend()
    #plt.tight_layout()
    # this overwrites
    plt.savefig(f"../../out/analysis/sa/plots/bar/{filename}_perc_check_label_{label}_{days}_days.pdf", format="pdf")
    plt.show()


def tc_sensitivity_analysis(prediction_output: DataFrame):
    prediction_output = _add_tc(prediction_output, TW_STEPS, MEAN_IO_RATIO)
    prediction_output = _add_td(prediction_output)
    prediction_output = _add_co(prediction_output, TP)
    _write_eval_df(prediction_output)

    # agg and plot avg and median
    _plot_agg(prediction_output, "avg", MODEL_NAME, acc=100)
    _plot_agg(prediction_output, "median", MODEL_NAME, acc=100)
    _plot_check_distr_model(prediction_output, MODEL_NAME)
    _plot_check_distr_model_over_time(prediction_output, MODEL_NAME, DAYS, label=0)
    _plot_check_distr_model_over_time(prediction_output, MODEL_NAME, DAYS, label=1)
    pass


if __name__ == '__main__':
    # Spark wants no special chars in the filename

    MODEL_NAME = "GBT5CV3parall_priorFeat_tune_maxDepth_Iter_Bins_03inst777Seed"
    DATASET_NAME = "batch_jobs_clean_03inst_1task_00015S_1F"
    MODEL_PATH = "../../out/model/dump/" + MODEL_NAME
    DATA_PATH = f"../../out/clean/{DATASET_NAME}/*.csv.gz"
    MODEL_OUTPUT_PATH = f"../../out/model/eval/{MODEL_NAME}Val/*.csv.gz"
    # TW_STEPS = [10, 12, 15, 17, 20]
    TW_STEPS = [0.5, 1, 3, 5, 10, 30]  # larger variance in the results
    MEAN_TW = 17.594
    MEAN_TR = 1.04
    MEAN_IO_RATIO = MEAN_TW / MEAN_TR  # this was measured in the checkpoint experiment
    """
    we define TP (in minutes as the time it takes to read the test data, do the transformation and write the results badck to HDFS
    (the latest query in the SQL tab of the history server: http://localhost:18080/history/spark-71bcef34c2d347679e1a291417f1583b/SQL/execution/?id=254)
    """
    TP = 23 / 60

    DAYS = 8
    APP_NAME = "sensitivity analysis of the checkpoint overhead of GBTNestedCVValidation data"

    spark_session = SparkSession.builder \
        .master("local[3]") \
        .config("spark.driver.memory", "6g") \
        .appName(APP_NAME) \
        .getOrCreate()

    prediction_output = spark_session.read.csv(
        MODEL_OUTPUT_PATH,
        header=True,
        inferSchema=True
    )

    tc_sensitivity_analysis(prediction_output)

    spark_session.stop()
