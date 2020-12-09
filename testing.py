import time
from pyspark.sql import DataFrame
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import RandomForestClassificationModel

spark = SparkSession.builder.master("local").appName("wineClasssification").getOrCreate()

testDf = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("s3://cs643/TrainingDataset.csv")
feature = [c for c in testDf.columns if c != 'quality']
assembler_test = VectorAssembler(inputCols=feature, outputCol="features")
test_trans = assembler_test.transform(testDf)

model= RandomForestClassificationModel.load("s3://cs643/wine_train_model.model")

predictions = model.transform(test_trans)

eval = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="accuracy")
accuracy = eval.evaluate(predictions)
print("accuracy test Error = %g" % (1.0 - accuracy))

from pyspark.mllib.evaluation import MulticlassMetrics
transformed_data = model.transform(test_trans)
print(eval.getMetricName(), 'accuracy:', eval.evaluate(transformed_data))

eval1 = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
accuracy = eval1.evaluate(predictions)
print("f1 score test Error = %g" % (1.0 - accuracy))
transformed_data = model.transform(test_trans)
print(eval1.getMetricName(), 'accuracy :', eval1.evaluate(transformed_data))