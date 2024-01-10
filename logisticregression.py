from pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext()
sqlContext = SQLContext(sc)
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
    'sf-crime/train.csv')
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution',
             'Address', 'X', 'Y']
data = data.select([column for column in data.columns
                    if column not in drop_list])
data.show(5)

data.printSchema()

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features",
                               vocabSize=10000, minDF=5)

from pyspark.ml.feature import StringIndexer

label_stringIdx = StringIndexer(inputCol="Category", outputCol="label")

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

# set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=42)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Descript", "Category", "probability", "label", "prediction") \
    .orderBy("probability", ascending=False) \
    .show(n=10, truncate=30)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(predictions))
