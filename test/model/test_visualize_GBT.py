import logging
import unittest
from os.path import exists

import pandas as pd
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import SparkSession

from src.model.visualize_GBT import _get_str_index, _plot_feature_importances, _tree_json

logging.basicConfig(filename="../../logs/model/test_visualize_GBT.log",
                    level=logging.INFO, format="%(asctime)s %(message)s")



class TestVisualizeGBT(unittest.TestCase):
    MODEL_NAME = "GBTNested5CV3parall_priorFeat_tune_maxDepth_Iter_03inst"
    model_path = f"../../out/model/dump/{MODEL_NAME}"
    DATA_NAME = "batch_jobs_clean_03inst_1task_00015S_1F"
    data_part_path = f"../../out/clean/{DATA_NAME}/part-00000-5538db0e-8e7d-45e9-852d-11424b1f7f74-c000.csv.gz"
    data_path = f"../../out/clean/{DATA_NAME}/*.csv.gz"
    test_path = f"../../out/model/eval/GBTNested5CV3parall_priorFeat_tune_maxDepth_Bins_03instVal/*.csv.gz"
    APP_NAME = "Test GBT Visualization"

    def setUp(self):
        self.spark = SparkSession \
            .builder \
            .appName(self.APP_NAME) \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()

    def test__get_str_index(self):
        cv_model = CrossValidatorModel.load(self.model_path)
        str_indices = _get_str_index(self.spark, cv_model, self.data_part_path)
        self.assertTrue(type(str_indices) == pd.DataFrame)

    def test__plot_feature_importances(self):
        _plot_feature_importances(spark=self.spark, model_name=self.MODEL_NAME, nested=False,
                                  model_path=self.model_path,
                                  data_path=self.data_part_path, test=True, figsize=(8, 6))
        assert exists(f"../../out/model/plots/{self.MODEL_NAME}_feat_imp_test.pdf")

    def test__tree_json(self):
        tree_to_json = """Tree 0 (weight 1.0):
    If (feature 7 in {0.0})
     If (feature 1 <= 2.0)
      If (feature 3 <= 1.5)
       If (feature 0 <= 16898.5)
        If (feature 0 <= 462.5)
         Predict: -0.6722085773663329
        Else (feature 0 > 462.5)
         Predict: -0.19642827170647376
       Else (feature 0 > 16898.5)
        If (feature 2 <= 0.78)
         Predict: -0.9343528783224255
        Else (feature 2 > 0.78)
         Predict: 0.7083333333333334
      Else (feature 3 > 1.5)
       If (feature 0 <= 16898.5)
        If (feature 23 in {1.0})
         Predict: -1.0
        Else (feature 23 not in {1.0})
         Predict: 0.9946015058957238
       Else (feature 0 > 16898.5)
        If (feature 4 <= 2.5)
         Predict: 1.0
        Else (feature 4 > 2.5)
         Predict: -1.0
     Else (feature 1 > 2.0)
      If (feature 2174 <= 0.0628140703517588)
       If (feature 5 <= 37098.5)
        If (feature 5 <= 31721.5)
         Predict: -0.9506172839506173
        Else (feature 5 > 31721.5)
         Predict: 0.06382978723404255
       Else (feature 5 > 37098.5)
        If (feature 218 in {0.0})
         Predict: -0.9722627737226277
        Else (feature 218 not in {0.0})
         Predict: -0.6226415094339622
      Else (feature 2174 > 0.0628140703517588)
       If (feature 67 in {1.0})
        If (feature 5 <= 2932.5)
         Predict: -0.23015873015873015
        Else (feature 5 > 2932.5)
         Predict: -0.9071729957805907
       Else (feature 67 not in {1.0})
        If (feature 0 <= 3649.5)
         Predict: 0.6566405114792212
        Else (feature 0 > 3649.5)
         Predict: 0.9403742467491278
    Else (feature 7 not in {0.0})
     If (feature 5 <= 55438.0)
      If (feature 5 <= 52129.5)
       If (feature 5 <= 14297.5)
        If (feature 5 <= 12136.5)
         Predict: -0.8627002288329519
        Else (feature 5 > 12136.5)
         Predict: 0.851140456182473
       Else (feature 5 > 14297.5)
        If (feature 0 <= 91.5)
         Predict: -0.3333333333333333
        Else (feature 0 > 91.5)
         Predict: -0.9209914794732765
      Else (feature 5 > 52129.5)
       If (feature 0 <= 20008.0)
        If (feature 0 <= 101.5)
         Predict: 1.0
        Else (feature 0 > 101.5)
         Predict: -0.8333333333333334
       Else (feature 0 > 20008.0)
        Predict: 1.0
     Else (feature 5 > 55438.0)
      If (feature 5 <= 75406.5)
       If (feature 0 <= 63.5)
        If (feature 0 <= 51.5)
         Predict: -0.7142857142857143
        Else (feature 0 > 51.5)
         Predict: 1.0
       Else (feature 0 > 63.5)
        If (feature 0 <= 19844.5)
         Predict: -0.8347107438016529
        Else (feature 0 > 19844.5)
         Predict: -0.9871589085072231
      Else (feature 5 > 75406.5)
       If (feature 5 <= 75450.0)
        Predict: 1.0
       Else (feature 5 > 75450.0)
        If (feature 5 <= 90144.5)
         Predict: -0.905254091300603
        Else (feature 5 > 90144.5)
         Predict: -0.238300935925126
    """""
        _tree_json(model_name=self.MODEL_NAME, tree=tree_to_json, test=True)
        assert exists(f"../../out/model/tree/{self.MODEL_NAME}_structure_test.json")

    def tearDown(self):
        self.spark.stop()


if __name__ == '__main__':
    unittest.main()
