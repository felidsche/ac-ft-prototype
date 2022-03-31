
package de.felixjanschneider.classification
import org.apache.log4j.Logger
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FunSuite}

import java.io.File


class GBTCVTaskFailBinPredTest extends FunSuite with BeforeAndAfter {
  // Declarations
  var gbtPredictor: GBTPredictor = _
  var params: GBTCVArgs = _
  var spark: SparkSession = _
  var context: SparkContext = _
  var logger: Logger = _
  var seed: Long = 777
  val run: Boolean = true

  before {
    // create class object
    gbtPredictor = new GBTPredictor(namenode = "ip:port")
    params = new GBTCVArgs(Seq("--model-name", "GBTNested5CV3parall_priorFeat_tune_maxDepth_Iter_03inst777SeedTest",
      "--local-fs", "--nested", "--folds", "2", "--parallelism", "2",
      "src/model/scala/src/test/resources/fixtures/batch_jobs_clean_03inst_1task_00015S_1F/part-00000-5538db0e-8e7d-45e9-852d-11424b1f7f74-c000_first_1000_lines.csv",
      "src/model/scala/src/test/out/model/eval/GBTNested5CV3parall_priorFeat_tune_maxDepth_Iter_03instTestVal"))

    // create spark
    val appName = "Test GBT CV Binary Task Failure Prediction"

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName(appName)
    context = new SparkContext(conf)

    spark = SparkSession
      .builder
      .master("local[2]")
      .config("driver-java-options", "-Dlog4j.configuration=file:src/test/resources/log4j.properties")
      .config("spark.driver.memory", "4g")
      .appName(appName)
      .getOrCreate()

    // create logger
    logger = Logger.getLogger(getClass.getName)
  }

  test("Model Selection CV and Model Training CV") {
    // !run to prevent the test from running again to save time
    assume(!run)
    // Define a Baseline GBT Classifier
    val gbt = new GBTClassifier()
      .setLabelCol("labels")
      .setFeaturesCol("features")
      .setFeatureSubsetStrategy("auto") //maxFeatures in sklearn
      .setSubsamplingRate(0.8) // Typical values ~0.8 generally work fine but can be fine-tuned further.
      .setStepSize(0.1) // the contribution of each decision tree -> lower to get more robust models but MLLib guides says dont tune
      .setSeed(seed)
      .setCacheNodeIds(true) //  If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees
      .setCheckpointInterval(params.checkpointInterval()) // after how many iterations the cache should be checkpointed

    // this will train: folds * combinations models
    val treeParamGrid = new ParamGridBuilder()
      // dummy config to have "some" params to tune
      .addGrid(gbt.maxDepth, Array(1, 2))
      .addGrid(gbt.maxIter, Array(1, 2))
      .build()
    gbtPredictor.runModelSelectionCVandModelTrainingCV(params, gbt, treeParamGrid, logger, seed, context, spark)
    val validationExists: Boolean = new File(params.outPath()).exists
    assert(validationExists)
  }

  test("train test split")(pending)

  test("train test validation split") {
    // !run to prevent the test from running again to save time
    assume(!run)
    // stratified (50/25/25) split to have 50% of the data for training, 25 for validation (hyper parameter tuning) and 25% for testing (results)
    logger.info("test")
    val splits: (DataFrame, DataFrame, DataFrame) = gbtPredictor.trainTestValSplit(
      path = params.dataPath(),
      spark = spark,
      trainFraction = 0.50,
      validationFraction = 0.25,
      seed = seed,
      logger = logger,
      local = params.localFs()
    )

    val training = splits._1
    val trainingCount = training.count()
    val trainingFailCount = training.filter("labels == 1").count()
    val test = splits._2
    val testCount = test.count()
    val testFailCount = test.filter("labels == 1").count()
    val validation = splits._3
    val validationCount = validation.count()
    val validationFailCount = validation.filter("labels == 1").count()

    logger.info(
      "training count: " + trainingCount.toString + "training fail count: " + trainingFailCount.toString + "\n" +
        "test count: " + testCount.toString + "test fail count: " + testFailCount.toString + "\n" +
        "validation count: " + validationCount.toString + "validation fail count: " + validationFailCount.toString + "\n"
    )

    assert((trainingCount > testCount) && (trainingCount > validationCount))
    // this is almost equal
    assert(trainingCount != testCount + validationCount)

  }

  after{
    spark.stop()
  }


}