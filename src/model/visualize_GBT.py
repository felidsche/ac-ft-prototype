import json
import logging
from itertools import chain
from typing import Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import SparkSession

logging.basicConfig(filename="../../logs/model/visualize_GBT.log",
                    level=logging.INFO, format="%(asctime)s %(message)s")

sty = "seaborn"
mpl.style.use(sty)


def show_values(axs, orient="v", space=.01):
    """
    from: https://www.statology.org/seaborn-barplot-show-values/
    :param axs:
    :param orient:
    :param space:
    :return:
    """

    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                value = "{:.3f}".format(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                value = "{:.3f}".format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _plot_feature_importances(spark: SparkSession, model_name: str, model_path: str, data_path: str, nested: bool,
                              test: bool, figsize: Tuple):
    if nested:
        # the outer pipeline model has only one stage (the pipeline)
        cv_model = CrossValidatorModel.load(model_path).bestModel.stages[-1]
        # the last stage of the pipeline is the GBT model
        gbt_classifier = cv_model.stages[-1]
    else:
        cv_model = CrossValidatorModel.load(model_path)
        gbt_classifier = cv_model.bestModel.stages[-1]
    debug_str = gbt_classifier.toDebugString
    logging.info("best GBT Model debugstring: \n" + debug_str)
    imps = gbt_classifier.featureImportances
    # add column with feature_names
    # extract metadata from the transformed dataset
    dataset = spark.read.csv(data_path, header=True, inferSchema=True)
    # add feature vector to the transformed dataset for attrs
    dataset_tranf = cv_model.bestModel.transform(dataset)
    attrs = sorted(
        (attr["idx"], attr["name"]) for attr in (chain(*dataset_tranf
                                                       .schema["features"]
                                                       .metadata["ml_attr"]["attrs"].values())))
    # combine it with the feature importances
    name_imps = [(name, imps[idx]) for idx, name in attrs if imps[idx]]
    name_imps_df = pd.DataFrame(name_imps, columns=["name", "imps"])
    # shorten the names
    short_names = []
    for name in name_imps_df["name"]:
        # check if it is a "L_j" feature (the first 3 characters are "log")
        if name[:3] == "log":
            # use the last three elements of the split parts L, j, XXXX and concat with "-" them to a shortname
            short_name = "-".join(name.split("_", -1)[-3:])
            short_names.append(short_name)
        else:
            # do the same as in "if" but for only the first two elements
            short_name = "-".join(name.split("_", 2)[:2])
            short_names.append(short_name)
    # add a new column with the shortened name
    name_imps_df["short_name"] = short_names
    plt.figure(figsize=figsize)
    plt.tight_layout()
    #plt.title("Estimate of the importance of each feature")
    p = sns.barplot(
        y=name_imps_df["short_name"],
        x=name_imps_df["imps"],
        data=name_imps_df,
        order=name_imps_df.sort_values("imps", ascending=False)["short_name"],
        orient="h",
        palette="coolwarm"
    )
    plt.xlabel("Feature importance in % (normalized)")
    plt.ylabel("Feature name")

    # show values on barplot
    show_values(p, "h", space=0)
    out_path = f"../../out/model/plots/{model_name}_feat_imp.pdf"
    if test:
        out_path = f"../../out/model/plots/{model_name}_feat_imp_test.pdf"

    plt.savefig(out_path, format="pdf")
    plt.show()


def _get_str_features(cvModel):
    """
    helper function to get the string labels of the feature columns
    :param cvModel:
    :return:
    """
    featVecAssembler = cvModel.bestModel.stages[5]
    features = featVecAssembler.getInputCols()
    # create a dict from the features list with the name and the index to create a pandas DF
    features_data = dict(zip(range(0, len(features)), features))
    features_df = pd.DataFrame.from_dict(features_data, orient="index", columns=["name"])
    # turn the index into a column so we can join
    features_df = features_df.reset_index()
    # shorten the name to make it fit on the plot
    features_df["name"] = ["-".join(x.split("_", 2)[:2]) for x in features_df["name"]]
    # rename index column to prevent mix up with the actual index
    features_df = features_df.rename(columns={"index": "idx"})
    return features_df


def ith_(v, i):
    """
    UDF helper function to transform vector columns to array columns
    :param v: the vector column
    :param i: the index
    """
    try:
        return float(v[i])
    except ValueError:
        return None


def _get_str_index(spark: SparkSession, cvModel: CrossValidatorModel, data_path: str) -> pd.DataFrame:
    """
    apply string indexing again to get the indices for categorical features to map feature importances
    :param spark: the spark session
    :param cvModel: the model to evaluate
    :param data_path: the path to the actual data
    :return:
    """
    nominal_categ_cols = ["map_reduce", "logical_job_name"]
    data = spark.read.csv(data_path, header=True, inferSchema=True).select(nominal_categ_cols)
    str_indexer = cvModel.bestModel.stages[0]
    str_indexer_out = str_indexer.transform(data).withColumnRenamed("logical_job_name_idx",
                                                                    "idx_imp").withColumnRenamed("logical_job_name",
                                                                                                 "name").toPandas()
    return str_indexer_out


def _parse(lines):
    """
    parser from https://github.com/felidsche/decision-tree-viz-spark
    :param lines:
    :return:
    """
    block = []
    while lines:

        if lines[0].startswith("If"):
            bl = " ".join(lines.pop(0).split()[1:]).replace("(", "").replace(")", "")
            block.append({"name": bl, "children": _parse(lines)})

            if lines[0].startswith("Else"):
                be = " ".join(lines.pop(0).split()[1:]).replace("(", "").replace(")", "")
                block.append({"name": be, "children": _parse(lines)})
        elif not lines[0].startswith(("If", "Else")):
            block2 = lines.pop(0)
            block.append({"name": block2})
        else:
            break
    return block


def _tree_json(tree: str, model_name: str, test: bool = False):
    """
    convert Tree to JSON from https://github.com/felidsche/decision-tree-viz-spark
    writes the tree structure as .json
    :param tree: 
    :param model_name:
    """
    data = []
    for line in tree.splitlines():
        if line.strip():
            line = line.strip()
            data.append(line)
        else:
            break
        if not line: break
    res = []
    res.append({"name": "Root", "children": _parse(data[1:])})
    if test:
        out_path = f"../../out/model/tree/{model_name}_structure_test.json"
    else:
        out_path = f"../../out/model/tree/{model_name}_structure.json"
    with open(out_path, "w") as outfile:
        json.dump(res[0], outfile)
    logging.info("Conversion Success !")


if __name__ == "__main__":
    appName = "GBT Visualization"
    spark = SparkSession \
        .builder \
        .appName(appName) \
        .master("local[2]") \
        .config("spark.driver.memory", "3g") \
        .getOrCreate()

    model_name = "GBT5CV3parall_priorFeat_tune_maxDepth_Iter_Bins_03inst777Seed"
    # all features
    #model_name = "GBTCV_03inst_066033split"
    model_path = f"../../out/model/dump/final/{model_name}"
    data_name = "batch_jobs_clean_03inst_1task_00015S_1F"
    # one part is enough to get all important LJNs
    data_path = f"../../out/clean/{data_name}/*.csv.gz"
    _plot_feature_importances(spark=spark, model_path=model_path, data_path=data_path, nested=False,
                              model_name=model_name, test=False, figsize=(8, 6))
    """
    from logs/model/visualize_GBT.log
    we only use tree 1/5 because it has weight 1.0 and the rest have 0.1
    """
    tree_to_json = """Tree 0 (weight 1.0):
    If (feature 7 in {0.0})
     If (feature 1 <= 2.0)
      If (feature 3 <= 1.5)
       If (feature 0 <= 20257.5)
        If (feature 0 <= 1110.5)
         Predict: -0.5539482053611995
        Else (feature 0 > 1110.5)
         Predict: -0.1663290113452188
       Else (feature 0 > 20257.5)
        If (feature 2 <= 0.595)
         Predict: -0.9398791652766967
        Else (feature 2 > 0.595)
         Predict: -0.4682926829268293
      Else (feature 3 > 1.5)
       If (feature 5 <= 196.5)
        Predict: 0.952286282306163
       Else (feature 5 > 196.5)
        If (feature 5 <= 52170.5)
         Predict: 0.9883792048929664
        Else (feature 5 > 52170.5)
         Predict: 0.9992081559932693
     Else (feature 1 > 2.0)
      If (feature 2116 <= 0.0628140703517588)
       If (feature 0 <= 174.5)
        Predict: -0.8236397748592871
       Else (feature 0 > 174.5)
        If (feature 5 <= 142575.5)
         Predict: -0.9621052631578947
        Else (feature 5 > 142575.5)
         Predict: -0.9784172661870504
      Else (feature 2116 > 0.0628140703517588)
       If (feature 63 in {1.0})
        Predict: -0.5699797160243407
       Else (feature 63 not in {1.0})
        If (feature 0 <= 3335.5)
         Predict: 0.6610760182966674
        Else (feature 0 > 3335.5)
         Predict: 0.9354973384824131
    Else (feature 7 not in {0.0})
     If (feature 5 <= 54781.0)
      If (feature 0 <= 53574.0)
       If (feature 5 <= 14467.5)
        If (feature 5 <= 13772.5)
         Predict: -0.8385650224215246
        Else (feature 5 > 13772.5)
         Predict: 0.9538239538239538
       Else (feature 5 > 14467.5)
        If (feature 0 <= 3335.5)
         Predict: -0.7805714285714286
        Else (feature 0 > 3335.5)
         Predict: -0.9621928166351607
      Else (feature 0 > 53574.0)
       If (feature 0 <= 53691.0)
        Predict: 1.0
       Else (feature 0 > 53691.0)
        Predict: -0.6871287128712872
     Else (feature 5 > 54781.0)
      If (feature 2 <= 0.20500000000000002)
       If (feature 4 <= 3.5)
        Predict: -0.9067164179104478
       Else (feature 4 > 3.5)
        If (feature 4 <= 10.5)
         Predict: -0.9835526315789473
        Else (feature 4 > 10.5)
         Predict: -0.9576470588235294
      Else (feature 2 > 0.20500000000000002)
       If (feature 2 <= 0.305)
        If (feature 5 <= 77485.0)
         Predict: 0.5
        Else (feature 5 > 77485.0)
         Predict: -0.3977066988533494
       Else (feature 2 > 0.305)
        If (feature 0 <= 2128.5)
         Predict: -0.7669902912621359
        Else (feature 0 > 2128.5)
         Predict: -0.9416342412451362
    """""
    _tree_json(tree=tree_to_json, model_name=model_name)
