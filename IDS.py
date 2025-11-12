import kagglehub
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StandardScaler
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import time

spark = (
    SparkSession.builder
    .appName("IDS-Distributed")
    .master("spark://spark-master:7077")
    .config("spark.executor.memory", "2g")
    .config("spark.executor.cores", "2")
    .getOrCreate()
)

print("Spark master:", spark.sparkContext.master)
print("Executor count:", spark.sparkContext.defaultParallelism)

cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level']

train_df = spark.read.csv("/data/KDDTrain+.txt", header=False, sep=",")
test_df  = spark.read.csv("/data/KDDTest+.txt", header=False, sep=",")

for old, new in zip(train_df.columns, cols):
    train_df=train_df.withColumnRenamed(old, new)

for old, new in zip(test_df.columns, cols):
    test_df = test_df.withColumnRenamed(old, new)

def cast_numeric_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    # Crea una copia per sicurezza
    df_casted = df

    for col_name in df.columns:
        if col_name not in exclude_cols:
            df_casted = df_casted.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    return df_casted

categorical_cols = ["protocol_type", "service", "flag", "outcome"]
train_df=cast_numeric_columns(train_df,categorical_cols)
test_df=cast_numeric_columns(test_df,categorical_cols)

train_df = train_df.withColumn(
    "outcome",
    when(col("outcome") == "normal", 0).otherwise(1)
)

test_df = test_df.withColumn(
    "outcome",
    when(col("outcome") == "normal", 0).otherwise(1)
)

y_train=train_df["outcome"]

train_df=train_df.drop("protocol_type", "service", "flag")
test_df=test_df.drop("protocol_type", "service", "flag")

numeric_cols = [c for c in train_df.columns if c not in ["protocol_type","service","flag","outcome"]]
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vec")
train_df_vec = assembler.transform(train_df)
test_df_vec  = assembler.transform(test_df)

scaler = StandardScaler(
    inputCol  = "features_vec",
    outputCol = "features_scaled",
    withMean  = True,   
    withStd   = True    
)

start_time = time.time()
scaler_model = scaler.fit(train_df_vec)
train_df_scaled = scaler_model.transform(train_df_vec)
end_time = time.time()
print(f"Time taken to scale training data: {end_time - start_time} seconds")

start_time = time.time()
test_df_scaled = scaler_model.transform(test_df_vec)
end_time = time.time()
print(f"Time taken to scale test data: {end_time - start_time} seconds")

lr = LogisticRegression(featuresCol="features_scaled", labelCol="outcome", maxIter=20)
start_time = time.time()
lr_model = lr.fit(train_df_scaled)
end_time = time.time()
print(f"Time taken to train logistic regression model: {end_time - start_time} seconds")

lr_preds = lr_model.transform(test_df_scaled)
lr_preds.select("outcome", "prediction", "probability").show(5)

evaluator = BinaryClassificationEvaluator(labelCol="outcome")
auc = evaluator.evaluate(lr_preds)
print("AUC:", auc)

spark.stop()