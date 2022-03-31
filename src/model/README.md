# Spark MLlib Models

## Auto-Sklearn  Binary Task Failure Prediction
- `predict_task_failures.py`
## MLlib GBT Nested CV Binary Task Failure Prediction
`scala/src/main/scala/de/felixjanschneider/classification/GBTNestedCVTaskFailBinPred.scala`
### Build
``
mvn clean package
``
### Run
- locally on 3 cores with custom log4j
```shell
$SPARK_HOME/bin/spark-submit \
 --class de.felixjanschneider.classification.GBTNestedCVTaskFailBinPred \
 --master local[3] \
 --driver-java-options "-Dlog4j.configuration=file:src/main/resources/log4j.properties" \
 target/scala-1.0-SNAPSHOT-jar-with-dependencies.jar
```