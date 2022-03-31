package de.felixjanschneider.classification

import org.apache.hadoop.fs.{FileSystem, LocatedFileStatus, Path, RemoteIterator}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.GBTClassificationModel
// Log messages will be written to the driver log
import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.exceptions.ScallopException
import org.rogach.scallop.{ScallopConf, ScallopOption}

import java.io.File
import java.net.URI
import scala.Array._
import scala.collection.mutable.ArrayBuffer

/**
 * class to run GBT Predictions
 *
 * @param namenode ip and port of the HDFS namenode
 */
class GBTPredictor(namenode: String) {

  /*
 * use only columns that are known prior to job execution
 * drop columns that are irrelevant for model training
 * e.g. because of: little intuitive impact, too many categories
 */
  val columns: Array[String] = Array[String](
    // from batch tasks
    "instance_num", // number of instances for the task
    "task_type", // values 1-12, meaning unknown (ordinal categ. feature)
    "plan_cpu", // number of cpus needed by the task, 100 is 1 core
    "plan_mem", // normalized memory size, [0, 100]
    // from batch instances
    "seq_no",
    "labels",
    // custom fields I added in trace.py
    "map_reduce", // whether this task is a map "m" or reduce "r" operation (nominal categ. feature) -> 1hot encode
    "sched_intv", // the schedule interval (in ?) if this value is set, its a recurring job (batch job)
    "job_exec", // the number of execution of this task
    "logical_job_name" // the overarching "app" that runs the jobs (nominal categ. feature) -> 1hot encode
  )

  // this is used in CV and Nested Cv for Model evaluation
  val emptyParamGrid: Array[ParamMap] = new ParamGridBuilder().build()

  /**
   * @param spark the SparkSession
   * @param path  the path to the cleaned dataset
   * @param seed  the random seed
   * @return
   */
  private def trainTestSplit(spark: SparkSession, path: String, trainFraction: Double, seed: Long, logger: Logger): (DataFrame, Dataset[Row]) = {
    logger.info("========== TRAIN TEST SPLIT ==========")

    // load cleaned training data from disk
    val data = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> ",", "header" -> "true")).csv(path = path)

    // stratified (80/20) split to make sure training has 80% of the failed tasks  for imbalanced classes
    val fractions = Map(0 -> trainFraction, 1 -> trainFraction)

    var training = data.stat.sampleBy(col = "labels", fractions = fractions, seed = seed)
    val test = data.except(training)

    training = training.select(
      columns.map(col): _*
    )

    (training, test)
  }


  /**
   * @param spark              the SparkSession object
   * @param path               the path to the data
   * @param trainFraction      the fraction of data to use for model training
   * @param validationFraction the fraction of data to use for model training
   * @param seed               the ranom seed
   * @param logger             the logger object
   * @param local              if run local
   * @return
   */
  def trainTestValSplit(spark: SparkSession, path: String, trainFraction: Double, validationFraction: Double, seed: Long, logger: Logger, local: Boolean): (DataFrame, DataFrame, DataFrame) = {
    logger.info("========== TRAIN TEST VALIDATION SPLIT ==========")

    // load cleaned training data from disk
    val data = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> ",", "header" -> "true")).csv(path = path)

    // stratified (50/25/25) split to have 50% of the data for training, 25 for validation (hyper parameter tuning) and 25% for testing (results)
    var fractions = Map(0 -> trainFraction, 1 -> trainFraction)
    // without replacement
    var training = data.stat.sampleBy(col = "labels", fractions = fractions, seed = seed)
    val rest = data.except(training)
    fractions = Map(0 -> validationFraction, 1 -> validationFraction)
    var validation = rest.stat.sampleBy(col = "labels", fractions = fractions, seed = seed)
    val test = rest.except(validation)

    if (!local) {
      val validationCount = validation.count()
      val validationFailCount = validation.filter("labels == 1").count()


      val testCount = test.count()
      val testFailCount = test.filter("labels == 1").count()

      val trainingCount = training.count()
      val trainingFailCount = training.filter("labels == 1").count()

      logger.info(
        "Data count: " + data.count().toString + "\n" +
          "training count: " + trainingCount.toString + " training fail count: " + trainingFailCount.toString + "\n" +
          "test count: " + testCount.toString + " test fail count: " + testFailCount.toString + "\n" +
          "validation count: " + validationCount.toString + " validation fail count: " + validationFailCount.toString + "\n"
      )
    } else logger.info("Skip count in local mode")
    training = training.select(
      columns.map(col): _*
    )

    validation = validation.select(
      columns.map(col): _*
    )
    (training, test, validation)
  }

  /**
   * @param training    the training dataset
   * @param gbt         the classifier object
   * @param paramGrid   the parameter grid to try using CV
   * @param folds       the numer of folds during CV
   * @param parallelism the number of parallelism during CV
   * @param logger      the logger object
   * @return
   */
  def getCVAndFeatures(training: DataFrame, gbt: GBTClassifier, paramGrid: Array[ParamMap], folds: Integer, parallelism: Integer, logger: Logger): (Array[String], CrossValidator) = {
    logger.info("========== GET CV AND FEATURES ==========")

    val nominalCategCols = Array[String]("map_reduce", "logical_job_name")
    // for comprehension to add suffix "_idx" to all indexed cols
    val nominalCategColsIndexed: Array[String] = for (j <- nominalCategCols) yield j + "_idx"

    // OneHotEncode nominal categ. features
    val strIndexer = new StringIndexer()
      .setInputCols(nominalCategCols)
      .setOutputCols(nominalCategColsIndexed)
      .setHandleInvalid("skip")

    val nominalCategColsIndexedOneHot: Array[String] = for (j <- nominalCategColsIndexed) yield j + "_one_hot"

    val oneHotEncoder = new OneHotEncoder()
      .setInputCols(strIndexer.getOutputCols)
      .setOutputCols(nominalCategColsIndexedOneHot)

    // for comprehension to filter for feature types
    val numericTypes = Array[String]("IntegerType", "FloatType", "DoubleType")
    val numericCols: Array[String] = for (i <- training.dtypes if numericTypes.contains(i._2) && i._1 != "labels") yield i._1

    // for comprehension to add suffix to all imputed cols
    val imputedCols: Array[String] = for (j <- numericCols) yield j + "_imp"

    val imputer = new Imputer()
      .setInputCols(numericCols)
      .setOutputCols(imputedCols)

    val cpuCols: Array[String] = Array("plan_cpu_imp")

    val cpuColsAssembler = new VectorAssembler()
      .setInputCols(cpuCols)
      .setOutputCol("plan_cpu_vec")

    // plan_cpu is not scaled yet as opposed to plan_mem
    val scaler = new MinMaxScaler()
      .setMin(0.0)
      .setMax(1.0)
      .setInputCol(cpuColsAssembler.getOutputCol)
      .setOutputCol(cpuColsAssembler.getOutputCol + "_scaled")

    // remove imputed cpu cols because they are in CpuColsVec already
    val imputedColsClean = imputedCols.filter(name => !name.contains("cpu"))
    // concat the outputs of the transformers
    val features: Array[String] = concat(imputedColsClean, oneHotEncoder.getOutputCols, Array(scaler.getOutputCol))
    logger.info("Features: \n" + features.mkString("Array(", ", ", ")"))

    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(strIndexer, oneHotEncoder, imputer, cpuColsAssembler, scaler, assembler, gbt))

    val cvEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("labels")
      .setMetricName("areaUnderROC")


    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(cvEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(folds)
      .setParallelism(parallelism)

    (features, cv)
  }

  /**
   * @param localFs   if run on local fs else HDFS
   * @param dataset   the dataset to use for training
   * @param cv        the CrossValidator object
   * @param modelName the name of the model
   * @param sc        the SparkContext
   * @param logger    the logger object
   * @return
   */
  def loadOrTrainAndSaveCV(localFs: Boolean, dataset: Dataset[Row], cv: CrossValidator, modelName: String, sc: SparkContext, logger: Logger): CrossValidatorModel = {
    /*
     * if the model path exists, the model is returned. Else the model is trained and saved
     */

    var modelExists: Boolean = false
    val hdfsPath = "hdfs://" + namenode + "/spark/alibaba2018-trace/out"
    val localPath = "/home/felix/TUB_Master_ISM/SoSe21/MA/acs-simulation/out/model/dump"
    // will be determined in the first if
    var path: String = ""

    if (localFs) {
      path = localPath + "/" + modelName
      modelExists = new File(path).exists


    } else {
      path = hdfsPath + "/" + modelName
      val hdfsFiles: Seq[String] = getAllFiles(hdfsPath, sc)
      modelExists = hdfsFiles.contains(modelName)
    }

    // todo: this doesnt work for HDFS
    val cvModel = {
      if (modelExists) CrossValidatorModel.load(path) else cv.fit(dataset)
    }

    if (modelExists) {
      logger.info("CV model was loaded from disk")
    } else if (localFs) {
      logger.info("Ran CV and chose the best set of parameters. saving locally...")
      cvModel.write.overwrite().save(path)
    } else {
      logger.info("Ran CV and chose the best set of parameters. saving to hdfs...")
      cvModel.write.overwrite().save(path)
    }

    cvModel
  }

  /**
   * getAllFiles - get all files recursively from the sub folders of HDFS modelPath.
   *
   * @param path String
   * @param sc   SparkContext
   * @return Seq[String]
   */
  def getAllFiles(path: String, sc: SparkContext): Seq[String] = {
    val conf = sc.hadoopConfiguration
    val fs = FileSystem.get(URI.create(path), conf)
    val files: RemoteIterator[LocatedFileStatus] = fs.listFiles(new Path(path), true) // true for recursive lookup
    val buf = new ArrayBuffer[String]
    while (files.hasNext) {
      val fileStatus = files.next()
      buf.append(fileStatus.getPath.toString)
    }
    buf
  }

  /**
   * @param cvModel       the trained model
   * @param test          the test set
   * @param features      the features
   * @param output        write output?
   * @param numPartitions number of partitions for the output
   */
  def modelEvaluation(cvModel: CrossValidatorModel, test: DataFrame, features: Array[String], output: Boolean, outPath: String, numPartitions: Integer, logger: Logger): Unit = {
    logger.info("========== MODEL EVALUATION ==========")
    // log the metrics for all parameter combinations
    val innerCvAvgMetrics: Array[Double] = cvModel.avgMetrics
    logger.info("Best params: " + getBestEstimatorParamMap(cvModel).toString() + "\n")
    logger.info("All metrics: " + innerCvAvgMetrics.mkString("Array(", ", ", ")") + "\n")
    // the features vector is not selected because it can't be exported as csv
    val predictionCols = Array[String]("prediction", "probability", "rawPrediction")
    // select the *all* initial columns, the features *in column format* and the prediction columns
    val validationCols: Array[String] = concat(test.columns, features, predictionCols)

    var testTransformed = cvModel.bestModel.transform(test)
    testTransformed = testTransformed.select(
      validationCols.map(col): _*
    ).select(
      col("*"),
      // get elements from vector columns because these are not supported in csv export
      vector_to_array(col("plan_cpu_vec_scaled")).getItem(0).alias("plan_cpu_imp_scaled"),
      vector_to_array(col("probability")).getItem(0).alias("proba0"),
      vector_to_array(col("rawPrediction")).getItem(0).alias("rawPred0"),
      vector_to_array(col("probability")).getItem(1).alias("proba1"),
      vector_to_array(col("rawPrediction")).getItem(1).alias("rawPred1")
    ).drop("map_reduce_idx_one_hot", "logical_job_name_idx_one_hot", // use the indexed for better readability
      "plan_cpu_vec_scaled", "probability", "rawPrediction")

    if (output) {
      logger.info("transformed test set is written to HDFS")
      // we choose no partitions because the validation set will be 30% of 240MB
      testTransformed.repartition(numPartitions).write.mode(SaveMode.Overwrite).
        options(Map("compression" -> "gzip", "delimiter" -> ",", "header" -> "true")).csv(outPath)
    } else logger.info("no output is written to HDFS")

  }

  /**
   * @param cvModel the trained CorssValidatorModel
   * @return the best parameter map
   */
  def getBestEstimatorParamMap(cvModel: CrossValidatorModel): ParamMap = {
    cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1
  }

  /**
   * Method to run Gradient Boosted Trees (GBT) Cross Validation (CV)
   *
   * @param params  the passed parameters
   * @param logger  the logger
   * @param seed    the random seed
   * @param context the spark context
   * @param spark   the spark session
   */
  def runModelTrainingCV(params: GBTCVArgs, logger: Logger, seed: Long, context: SparkContext, spark: SparkSession): Unit = {
    val dataPreprocessingResult: (DataFrame, DataFrame) = trainTestSplit(spark, params.dataPath(), trainFraction = params.trainFraction(), seed, logger)
    val training: DataFrame = dataPreprocessingResult._1
    val test: DataFrame = dataPreprocessingResult._2

    // Define a Baseline GBT Classifier using the best parameters that were identified
    val gbt = new GBTClassifier()
      .setLabelCol("labels")
      .setFeaturesCol("features")
      .setFeatureSubsetStrategy("auto") //maxFeatures in sklearn
      .setSubsamplingRate(params.trainFraction()) // Typical values ~0.8 generally work fine but can be fine-tuned further.
      .setStepSize(0.1) // (default) learning-rate in sk-learn -> lower to get more robust models
      .setSeed(seed)
      .setCacheNodeIds(true) //  If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees
      .setCheckpointInterval(params.checkpointInterval()) // after how many iterations the cache should be checkpointed
      .setMaxBins(50) // tuned on 1%
      .setMaxDepth(10) // tuned on 1%
      .setMinInstancesPerNode(400) // tuned on 1%

    // we use an empty parameter grid to use CrossValidator for Model evaluation
    val (features: Array[String], trainedGBTModel: CrossValidatorModel) = runModelTraining(params, logger, context, training, gbt)
    modelEvaluation(trainedGBTModel, test, features, output = true, params.outPath(), numPartitions = 1, logger = logger)
  }


  /**
   * @param params  the params passed to the program
   * @param logger  the logger object
   * @param seed    the random seed
   * @param context the spark context
   * @param spark   the spark session
   */
  def runModelSelectionCVandModelTrainingCV(params: GBTCVArgs, gbt: GBTClassifier, paramGrid: Array[ParamMap], logger: Logger, seed: Long, context: SparkContext, spark: SparkSession): Unit = {
    // the splits are (train 0.50, test 0.375,  validation 0.125)
    val splits: (DataFrame, DataFrame, DataFrame) = trainTestValSplit(
      path = params.dataPath(),
      spark = spark,
      trainFraction = params.trainFraction(),
      validationFraction = params.validationFraction(),
      seed = seed,
      logger = logger,
      local = params.localFs()
    )

    val training: DataFrame = splits._1
    val test: DataFrame = splits._2
    val validation: DataFrame = splits._3

    val tunedGBTModel: GBTClassifier = runModelSelection(params, gbt, paramGrid, logger, seed, context, validation)
    val (features: Array[String], trainedAndTunedGBTModel: CrossValidatorModel) = runModelTraining(params, logger, context, training, tunedGBTModel)

    modelEvaluation(trainedAndTunedGBTModel, test, features, output = true, params.outPath(), numPartitions = 1, logger = logger)

  }


  /**
   * @param params the passed parameters
   * @param logger the logger object
   * @param context the SparkContext
   * @param training the DataFrame for training the model
   * @param gbt the GBT estimator to train
   * @return the features and the trained CV model
   */
  private def runModelTraining(params: GBTCVArgs, logger: Logger, context: SparkContext, training: DataFrame, gbt: GBTClassifier) = {
    logger.info("========== MODEL TRAINING ==========")
    val modelTrainingCvAndFeatures: (Array[String], CrossValidator) = getCVAndFeatures(training, gbt, emptyParamGrid, params.folds(), params.parallelism(), logger)
    logger.info("Model params: " + gbt.extractParamMap().toString())
    val features = modelTrainingCvAndFeatures._1
    val modelTrainingCv = modelTrainingCvAndFeatures._2
    val trainedCVModel: CrossValidatorModel = loadOrTrainAndSaveCV(localFs = params.localFs(), training, modelTrainingCv, params.modelName(), context, logger)
    (features, trainedCVModel)
  }

  /**
   * wrapper for running the model selection
   *
   * @param params the passed parameters
   * @param gbt the GBT estimator to train
   * @param paramGrid the parameter grid with the options to select from
   * @param logger the logger object
   * @param seed the sample seed
   * @param context the SparkContext
   * @param validation the validation set
   * @return
   */
  private def runModelSelection(params: GBTCVArgs, gbt: GBTClassifier, paramGrid: Array[ParamMap], logger: Logger, seed: Long, context: SparkContext, validation: DataFrame) = {
    logger.info("========== MODEL SELECTION ==========")
    val modelSelectionCvAndFeatures: (Array[String], CrossValidator) = getCVAndFeatures(validation, gbt, paramGrid, params.folds(), params.parallelism(), logger)
    val modelSelectionCv = modelSelectionCvAndFeatures._2

    val tunedCV: CrossValidatorModel = loadOrTrainAndSaveCV(params.localFs(), validation, modelSelectionCv, params.modelName() + "_select", context, logger)

    val modelSelectionMetrics: Array[Double] = tunedCV.avgMetrics
    logger.info("Best params: " + getBestEstimatorParamMap(tunedCV).toString() + "\n")
    logger.info("All params: " + tunedCV.getEstimatorParamMaps.mkString("Array(", ", ", ")") + "\n")
    logger.info("All metrics: " + modelSelectionMetrics.mkString("Array(", ", ", ")") + "\n")
    val bestPipelineModel = tunedCV.bestModel.asInstanceOf[PipelineModel]
    val tunedGBTModel = bestPipelineModel.stages.last.asInstanceOf[GBTClassificationModel]

    // get the tuned hyper parameters
    val tunedMaxIter = tunedGBTModel.getMaxIter
    val tunedMaxDepth = tunedGBTModel.getMaxDepth
    val tunedMaxBins = tunedGBTModel.getMaxBins
    // set the tuned hyper parameters
    gbt.setMaxDepth(tunedMaxDepth)
    gbt.setMaxIter(tunedMaxIter)
    gbt.setMaxBins(tunedMaxBins)
    gbt
  }
}


/**
 * class that defines the parameters to pass
 * @param a a sequence of parameter string arguments
 */
class GBTCVArgs(a: Seq[String]) extends ScallopConf(a) {
  val dataPath: ScallopOption[String] = trailArg[String](required = true, name = "<dataPath>",
    descr = "The path to the dataset")

  val outPath: ScallopOption[String] = trailArg[String](required = true, name = "<outPath>",
    descr = "The path to the results")

  // boolean options are flag options
  val localFs: ScallopOption[Boolean] = opt[Boolean](noshort = true, default = Option(false),
    descr = "if true use localFS else use HDFS")

  val nested: ScallopOption[Boolean] = opt[Boolean](noshort = true, default = Option(false),
    descr = "if true CV should be nested")

  val trainFraction: ScallopOption[Double] = opt[Double](noshort = true, default = Option(0.50),
    descr = "the stratified fraction (by labels) of data to use for trainnig"
  )

  val validationFraction: ScallopOption[Double] = opt[Double](noshort = true, default = Option(0.25),
    descr = "the stratified fraction (by labels) of data to use for hyperparameter tuning"
  )

  val modelName: ScallopOption[String] = opt[String](noshort = true, default = Option("model"),
    descr = "the name of the model")


  val folds: ScallopOption[Int] = opt[Int](noshort = true, default = Option(5),
    descr = "Amount of CV folds")

  val parallelism: ScallopOption[Int] = opt[Int](noshort = true, default = Option(3),
    descr = "Amount of CV parallelism")

  val checkpointInterval: ScallopOption[Int] = opt[Int](noshort = true, default = Option(5),
    descr = "After how many iterations of the algo to checkpoint")


  override def onError(e: Throwable): Unit = e match {
    case ScallopException(message) =>
      println(message)
      println()
      printHelp()
      System.exit(1)
    case _ => super.onError(e)
  }

  verify()
}

/**
 * An ML Pipeline for binary task failure classification using GBTPredictor
 */
object GBTCVTaskFailBinPred {

  val namenode = "ip:port"


  var spark: SparkSession = _
  var context: SparkContext = _
  var conf: SparkConf = _

  def main(args: Array[String]): Unit = {

    val params = new GBTCVArgs(args)
    val logger = Logger.getLogger(getClass.getName)
    val appName = "GBT CV Binary Task Failure Prediction"
    val seed: Long = 777

    // spark boilerplate relative to exec mode
    if (params.localFs()) {
      val master = "local[2]" // use 2 cores locally
      conf = new SparkConf()
        .setAppName(appName)
        .setMaster(master)
        .set("spark.driver.memory", "2g") // use X GB RAM locally

      context = new SparkContext(conf)

      spark = SparkSession
        .builder
        .master(master)
        .config("spark.driver.memory", "2g")
        .appName(appName)
        .getOrCreate()

    } else {
      conf = new SparkConf()
        .setAppName(appName)
      context = new SparkContext(conf)
      context.setCheckpointDir("hdfs://" + namenode + "/checkpoints/felix-schneider-thesis")

      spark = SparkSession
        .builder
        .appName(appName)
        .getOrCreate()

    }


    val gbtPredictor: GBTPredictor = new GBTPredictor(namenode)

    // run nested or normal CV
    if (params.nested()) {
      // Define a Baseline GBT Classifier
      val gbt = new GBTClassifier()
        .setLabelCol("labels")
        .setFeaturesCol("features")
        .setFeatureSubsetStrategy("auto") //maxFeatures in sklearn
        .setSubsamplingRate(0.8) // Typical values ~0.8 generally work fine but can be fine-tuned further.
        .setSeed(seed)
        .setCacheNodeIds(true) //  If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees
        .setCheckpointInterval(params.checkpointInterval()) // after how many iterations the cache should be checkpointed

      // this will train: folds * combinations models
      val treeParamGrid = new ParamGridBuilder()
        // for reference see GBM_Parameters.xslx
        .addGrid(gbt.maxIter, Array(2, 5)) //maximum number of trees, it is ofen s often reasonable to use smaller (shallower) trees with GBTs
        .addGrid(gbt.maxDepth, Array(2, 5)) // Maximum depth of the tree, important -> tune first
        .addGrid(gbt.maxBins, Array(32, 2200)) // must be at least 2 and at lest max number of categories(features) -> (ljn ~2500)
        .build()
      gbtPredictor.runModelSelectionCVandModelTrainingCV(params, gbt, treeParamGrid, logger, seed, context, spark)
    } else gbtPredictor.runModelTrainingCV(params, logger, seed, context, spark)
    spark.stop()
  }

}

