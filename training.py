from pyspark.sql import DataFrame
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics


spark = SparkSession.builder.master("local").appName("WineQualityPrediction").config("spark.some.config.option","some-value").getOrCreate()

trainDf = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("s3://cs643/TrainingDataset.csv")                                      
valDf = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("s3://cs643/ValidationDataset.csv")

featureColumns = [c for c in trainDf.columns if c != 'quality']
assembler_t = VectorAssembler(inputCols=featureColumns, outputCol="features")
train_trans = assembler_t.transform(trainDf)
train_trans.cache()

feature = [c for c in valDf.columns if c != 'quality']
assembler_v = VectorAssembler(inputCols=feature, outputCol="features")
val_trans = assembler_v.transform(valDf)

from pyspark.ml.classification import RandomForestClassifier
random_forest = RandomForestClassifier(labelCol='""""quality"""""', featuresCol="features", numTrees=10)
model = random_forest.fit(train_trans)
model.save("s3://cs643/wine_train_model.model")

print("Model Trained Sucessfully")

