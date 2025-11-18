# ============================================================
# VERSIONE SPARK CON TIMING DETTAGLIATO
# (Ottimizzata per cluster con Standard Scaler)
# ============================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
import time
import os
import pandas as pd
import numpy as np # Necessario per alcune logiche Python nel contesto Spark

# ============================================================
# FILE DI OUTPUT E LOGGING
# ============================================================
# Inizializza i file di log
preproc_log = open("spark_timing_preprocessing.txt", "w")
model_log = open("spark_timing_models.txt", "w")
total_log = open("spark_timing_total.txt", "w")

def log_step(name, t_start, log_file):
    """Scrive tempo singolo step sia su console che su file."""
    t = time.time() - t_start
    msg = f"{name}: {t:.2f} sec"
    print(f"   → {msg}")
    log_file.write(msg + "\n")
    return t

print("\n====================== INIZIO PIPELINE SPARK ======================\n")
T0 = time.time()


# ============================================================
# 1) INIZIALIZZAZIONE SPARK SESSION
# ============================================================
# Configurazione per Cluster (come nel tuo codice)
spark = (
    SparkSession.builder
    .appName("accidents")
    .master("spark://spark-master:7077")
    .config("spark.driver.memory", "6g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.executor.memory", "6g")
    .config("spark.executor.cores", "2")
    .config("spark.executor.instances", "2")
    .config("spark.sql.shuffle.partitions", "200") 
    .config("spark.default.parallelism", "200")
    .getOrCreate()
)

# ============================================================
# 2) CARICAMENTO DATI
# ============================================================
t = time.time()
print("🔵 Caricamento Dataset...")
csv_path = "datasets/US_Accidents_March23.csv"
accidents = spark.read.csv(csv_path, header=True, inferSchema=True)
log_step("Caricamento CSV", t, preproc_log)


# ============================================================
# 3) FILL NA NUMERICI
# ============================================================
t = time.time()
print("🔵 Riempimento NaN numerici...")

mean_temp = accidents.select(F.mean("Temperature(F)")).first()[0]
accidents = accidents.fillna({"Temperature(F)": mean_temp})

weather_numeric = ['Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)','Precipitation(in)']
    
for col_name in weather_numeric:
    mean_val = accidents.select(F.mean(col_name)).first()[0]
    accidents = accidents.fillna({col_name: mean_val})

log_step("Riempimento NA numerici", t, preproc_log)

# ============================================================
# 4) DROP COLONNE INUTILI
# ============================================================
t = time.time()
print("🔵 Rimozione colonne inutili...")

to_drop = [
    'ID', "Source",'Description', 'Distance', 'End_Time', 'Distance(mi)', 
    'End_Lng', 'End_Lat', 'Duration', "Weather_Timestamp", "Country", "Turning_Loop"
]
accidents = accidents.drop(*[c for c in to_drop if c in accidents.columns])

log_step("Drop colonne inutili", t, preproc_log)


# ============================================================
# 5) SIMPLIFY WIND DIRECTION & WEATHER CONDITION
# ============================================================
t = time.time()
print("🔵 Simplify Wind_Direction & Weather_Condition...")

# Wind Direction
wind_dir_mapping = {
    "North": 'N', 'NNE': 'NE', 'NE': 'NE', 'ENE': 'E', 'East': 'E', 'ESE': 'SE', 'SE': 'SE', 'SSE': 'S', 
    'South': 'S', 'SSW': 'SW', 'SW': 'SW', 'WSW': 'W', 'West': 'W', 'WNW': 'NW', 'NW': 'NW', 'NNW': 'NW',
    "Calm":"CALM", "Variable":"VAR"
}
accidents = accidents.replace(wind_dir_mapping, subset=['Wind_Direction'])

# Weather Condition
patterns = {
    "Clear": "Clear", "Cloud": "Cloud|Overcast", "Rain": "Rain|storm", "Heavy_Rain": "Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms",
    "Snow": "Snow|Sleet|Ice", "Heavy_Snow": "Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls", "Fog": "Fog"
}
expr = F.lit(None)
for category, pattern in patterns.items():
    expr = F.when(F.col("Weather_Condition").rlike(f"(?i){pattern}"), category).otherwise(expr)
accidents = accidents.withColumn("Weather_Condition", expr)

log_step("Simplify Wind & Weather", t, preproc_log)


# ============================================================
# 6) FEATURE DATE
# ============================================================
t = time.time()
print("🔵 Conversione timestamp e Feature Date...")

accidents = accidents.withColumn("Start_Time_parsed", F.to_timestamp(F.col("Start_Time"), "yyyy-MM-dd HH:mm:ss"))

accidents = accidents.withColumn("Hour",  F.hour("Start_Time_parsed")) # Spark: Hour 0-23
accidents = accidents.withColumn("Day",   F.dayofweek("Start_Time_parsed"))
accidents = accidents.withColumn("Month", F.month("Start_Time_parsed"))
accidents = accidents.withColumn("Year",  F.year("Start_Time_parsed"))

accidents = accidents.withColumn("Part_of_Day",
    F.when(F.col("Hour").between(1, 6),  "Night")
     .when(F.col("Hour").between(7, 12), "Morning")
     .when(F.col("Hour").between(13, 18),"Afternoon")
     .otherwise("Evening")
)

accidents = accidents.withColumn("Sunrise_Sunset", F.when(F.col("Hour").between(1, 16), "Day").otherwise("Night"))
accidents = accidents.drop("Start_Time_parsed", "Start_Time")

log_step("Feature Date", t, preproc_log)


# ============================================================
# 7) FILL NA CATEGORICHE & ENCODING
# ============================================================
t = time.time()
print("🔵 Riempimento NA categoriche e Encoding...")

# Riempimento NA categoriche
cat_fill_cols = ["Street", "City", "Zipcode", "Weather_Condition", "Wind_Direction", "County", "State", "Airport_Code"]
for c in cat_fill_cols:
    if c in accidents.columns:
        accidents = accidents.withColumn(c, F.coalesce(F.col(c), F.lit("Unknown")))

# Numerical Encoding (StringIndexer) - Part_of_Day
weather_indexer = StringIndexer(inputCol="Part_of_Day", outputCol="Part_of_Day_idx", handleInvalid="keep")
accidents = weather_indexer.fit(accidents).transform(accidents).drop("Part_of_Day")

# Numerical Encoding (StringIndexer) - Wind_Direction
wind_indexer = StringIndexer(inputCol="Wind_Direction", outputCol="Wind_Direction_idx", handleInvalid="keep")
accidents = wind_indexer.fit(accidents).transform(accidents).drop("Wind_Direction")

# Frequency Encoding
high_card_cols = ["Street", "City", "Zipcode", "County", "Airport_Code", "State"]
def freq_encode(df, col):
    freq = df.groupBy(col).count().withColumnRenamed("count", f"{col}_freq")
    df = df.join(freq, on=col, how="left")
    return df.drop(col)

for c in high_card_cols:
    accidents = freq_encode(accidents, c)
    
# Binary Encoding
binary_cols = [
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", 
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
]

for c in binary_cols:
    if c in accidents.columns:
        accidents = accidents.withColumn(f"{c}_bin", F.when(F.lower(F.col(c)).isin("true", "yes", "1", "day"), 1).otherwise(0))
        accidents = accidents.drop(c)

log_step("Encoding & Freq Encoding", t, preproc_log)


# ============================================================
# 8) FEATURE SELECTION (CORRELATION DROP)
# ============================================================
t = time.time()
print("🔵 Feature Selection (Drop low correlation)...")

# Drop features come nello script originale
to_drop_corr = [
    "Civil_Twilight_bin","Nautical_Twilight_bin","Astronomical_Twilight_bin",
    "Traffic_Calming_bin","Bump_bin","Give_Way_bin","Day","Visibility(mi)","Roundabout_bin",
    "Wind_Direction_idx"
]
accidents = accidents.drop(*[c for c in to_drop_corr if c in accidents.columns])

log_step("Drop features", t, preproc_log)


# ============================================================
# 9) TRAIN/TEST SPLIT & VECTOR ASSEMBLER SETUP
# ============================================================
t = time.time()
print("🔵 Train/Test Split...")

# Stratified Shuffle Hack (come nel tuo codice)
accidents = accidents.withColumn("rand", F.rand(seed=42))
accidents = accidents.orderBy("Severity", "rand")

train, test = accidents.randomSplit([0.7, 0.3], seed=42)

log_step("Split Data", t, preproc_log)

# Preparazione Features per Assembler
feature_cols = [
    "Start_Lat", "Start_Lng", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
    "Pressure(in)", "Wind_Speed(mph)", "Precipitation(in)",
    "Hour", "Month", "Year",
    "Part_of_Day_idx", "Street_freq", "City_freq", "Zipcode_freq",
    "County_freq", "Airport_Code_freq", "State_freq",
    "Sunrise_Sunset_bin", "Amenity_bin", "Crossing_bin", "Junction_bin",
    "No_Exit_bin", "Railway_bin", "Station_bin", "Stop_bin",
    "Traffic_Signal_bin"
]

assembler = VectorAssembler(
    inputCols=[c for c in feature_cols if c in train.columns],
    outputCol="features",
    handleInvalid="skip" # IMPORTANTE: salta le righe con NaN (come fatto da dropna in Pandas)
)

# Creazione dello StandardScaler
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)


# ============================================================
# 10) RANDOM FOREST
# ============================================================
t = time.time()
print("🔵 Training Random Forest...")

rf = RandomForestClassifier(
    featuresCol="scaled_features", # Usiamo i dati scalati
    labelCol="Severity",
    numTrees=10,
    maxDepth=12,
    maxBins=32,
    seed=42
)

# Pipeline RF: Assembler -> Scaler -> RF
pipeline_rf = Pipeline(stages=[
    assembler,
    scaler,
    rf
])

# Fit
rf_model = pipeline_rf.fit(train)
rf_time = time.time() - t
print(f"   → RF Training Time: {rf_time:.2f} sec")
model_log.write(f"RF Training Time: {rf_time:.2f} sec\n")

# Eval RF
preds_rf = rf_model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction")

f1_rf = evaluator.setMetricName("f1").evaluate(preds_rf)
acc_rf = evaluator.setMetricName("accuracy").evaluate(preds_rf)

print(f"   ✅ RF F1 Score: {f1_rf:.4f}")
print(f"   ✅ RF Accuracy: {acc_rf:.4f}")
model_log.write(f"RF F1: {f1_rf:.4f}, Acc: {acc_rf:.4f}\n")


# ============================================================
# 11) LOGISTIC REGRESSION
# ============================================================
t = time.time()
print("🔵 Training Logistic Regression...")

lr = LogisticRegression(
    featuresCol="scaled_features", # Usa i dati scalati
    labelCol="Severity",
    maxIter=50,
    regParam=0.0,
    elasticNetParam=0.0
)

# Pipeline LR: Assembler -> Scaler -> LR
pipeline_lr = Pipeline(stages=[
    assembler,
    scaler,
    lr
])

# Fit
lr_model = pipeline_lr.fit(train)
lr_time = time.time() - t
print(f"   → LR Training Time: {lr_time:.2f} sec")
model_log.write(f"LR Training Time: {lr_time:.2f} sec\n")

# Eval LR
preds_lr = lr_model.transform(test)

f1_lr = evaluator.setMetricName("f1").evaluate(preds_lr)
acc_lr = evaluator.setMetricName("accuracy").evaluate(preds_lr)

print(f"   ✅ LR F1 Score: {f1_lr:.4f}")
print(f"   ✅ LR Accuracy: {acc_lr:.4f}")
model_log.write(f"LR F1: {f1_lr:.4f}, Acc: {acc_lr:.4f}\n")

# ============================================================
# 12) TEMPO TOTALE E CHIUSURA
# ============================================================
total_time = time.time() - T0
print("\n====================== FINE PIPELINE ======================\n")
print(f"⏱️  TEMPO TOTALE: {total_time:.2f} sec")
total_log.write(f"TOTAL TIME: {total_time:.2f} sec\n")

preproc_log.close()
model_log.close()
total_log.close()
spark.stop()