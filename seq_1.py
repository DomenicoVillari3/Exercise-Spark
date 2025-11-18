# ============================================================
# VERSIONE SEQUENZIALE IN PANDAS — CON TIMING SU FILE TXT
# ============================================================

import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import kagglehub
import os

# ============================================================
# FILE DI OUTPUT
# ============================================================
preproc_log = open("timing_preprocessing.txt", "w")
model_log = open("timing_models.txt", "w")
total_log = open("timing_total.txt", "w")

def log_step(name, t_start):
    """Scrive tempo singolo step sia su console che su file."""
    t = time.time() - t_start
    msg = f"{name}: {t:.2f} sec\n"
    print("   →", msg.strip())
    preproc_log.write(msg)
    return t


print("\n====================== INIZIO PIPELINE ======================\n")
T0 = time.time()


# ============================================================
# 1) LETTURA DATASET
# ============================================================
# Download latest version of the dataset
print("📦 Downloading dataset from KaggleHub...")
dataset_path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
print("📁 Path to dataset files:", dataset_path)

# Trova automaticamente il file CSV corretto
csv_path = None
for f in os.listdir(dataset_path):
    if f.endswith(".csv"):
        csv_path = os.path.join(dataset_path, f)
        break

if csv_path is None:
    raise FileNotFoundError("❌ Nessun CSV trovato nella cartella scaricata da KaggleHub!")

print("📄 CSV trovato:", csv_path)

# Caricamento CSV con log del tempo
t = time.time()
print("🔵 Caricamento CSV...")
df = pd.read_csv(csv_path)
log_step("Caricamento CSV", t)

# ============================================================
# 2) FILL NA — NUMERICI
# ============================================================
t = time.time()
print("🔵 Riempimento NaN numerici con mediana...")

num_cols = [
    'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
]

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

log_step("Riempimento NA numerici", t)


# ============================================================
# 3) DROP COLONNE NON UTILI
# ============================================================
t = time.time()
print("🔵 Rimozione colonne inutili...")

to_drop = [
    'ID', 'Source', 'Description', 'Distance(mi)',
    'End_Time', 'End_Lng', 'End_Lat', 'Weather_Timestamp'
]

df.drop(columns=to_drop, inplace=True)
log_step("Drop colonne inutili", t)


# ============================================================
# 4) SIMPLIFY WIND DIRECTION
# ============================================================
t = time.time()
print("🔵 Simplify Wind_Direction...")

wind_map = {
    "North": 'N', 'NNE': 'NE', 'NE': 'NE', 'ENE': 'E',
    'East': 'E', 'ESE': 'SE', 'SE': 'SE', 'SSE': 'S',
    'South': 'S', 'SSW': 'SW', 'SW': 'SW', 'WSW': 'W',
    'West': 'W', 'WNW': 'NW', 'NW': 'NW', 'NNW': 'NW',
    "Calm": "CALM", "Variable": "VAR"
}
df['Wind_Direction'] = df['Wind_Direction'].replace(wind_map)
log_step("Simplify Wind_Direction", t)


# ============================================================
# 5) SIMPLIFY WEATHER CONDITION
# ============================================================
t = time.time()
print("🔵 Simplify Weather_Condition...")

patterns = {
    "Clear": "Clear",
    "Cloud": "Cloud|Overcast",
    "Rain": "Rain|storm",
    "Heavy_Rain": "Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms",
    "Snow": "Snow|Sleet|Ice",
    "Heavy_Snow": "Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls",
    "Fog": "Fog"
}

def simplify_weather(x):
    for name, pat in patterns.items():
        if pd.notna(x) and re.search(pat, x, flags=re.IGNORECASE):
            return name
    return "Unknown"

df['Weather_Condition'] = df['Weather_Condition'].apply(simplify_weather)

log_step("Simplify Weather_Condition", t)


# ============================================================
# 6) FEATURE DATE
# ============================================================
t = time.time()
print("🔵 Conversione timestamp e creazione date features...")

df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")

df["Hour"] = df["Start_Time"].dt.hour
df["Day"] = df["Start_Time"].dt.dayofweek + 1
df["Month"] = df["Start_Time"].dt.month
df["Year"] = df["Start_Time"].dt.year

df["Part_of_Day"] = pd.cut(
    df["Hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    include_lowest=True
)

df["Sunrise_Sunset"] = np.where(df["Hour"] < 16, "Day", "Night")

log_step("Feature engineering date", t)


# ============================================================
# 7) FILL NA CATEGORICHE
# ============================================================
t = time.time()
print("🔵 Riempimento NA categoriche...")

cat_cols = ["Street", "City", "Zipcode", "Weather_Condition", "Wind_Direction"]

for c in cat_cols:
    df[c] = df[c].fillna("Unknown")

log_step("Riempimento NA categoriche", t)


# ============================================================
# 8) FREQUENCY ENCODING
# ============================================================
t = time.time()
print("🔵 Frequency encoding...")

high_card = ["Street", "City", "Zipcode", "County", "Airport_Code", "State"]

for c in high_card:
    freq = df[c].value_counts()
    df[c + "_freq"] = df[c].map(freq)
    df.drop(columns=[c], inplace=True)

log_step("Frequency encoding", t)


# ============================================================
# 9) ONE HOT
# ============================================================
t = time.time()
print("🔵 One-hot encoding...")

oh_cols = ["Weather_Condition", "Timezone"]
df = pd.get_dummies(df, columns=oh_cols)

log_step("One-hot encoding", t)


# ============================================================
# 10) BINARY ENCODING — VERSIONE CORRETTA
# ============================================================
t = time.time()
print("🔵 Binary encoding...")

binary_cols = [
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
    "Astronomical_Twilight", "Amenity", "Bump", "Crossing", "Give_Way",
    "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal"
]

def to_bin(x):
    if pd.isna(x):
        return 0
    x = str(x).strip().lower()
    if x in ["true", "yes", "1", "day"]:
        return 1
    return 0

for c in binary_cols:
    df[c + "_bin"] = df[c].apply(to_bin).astype(int)
    df.drop(columns=[c], inplace=True)

log_step("Binary encoding", t)


# ============================================================
# 11) CORRELAZIONI → DROP FEATURE IRRILEVANTI
# ============================================================
t = time.time()
print("🔵 Analisi correlazioni...")

corr = df.corr(numeric_only=True)["Severity"].abs()
low_corr = corr[corr < 0.05].index.tolist()

df.drop(columns=low_corr, inplace=True)

preproc_log.write(f"Drop feature per bassa correlazione: {len(low_corr)} colonne\n")
log_step("Drop feature basse correlazioni", t)


# ============================================================
# 12) TRAIN/TEST
# ============================================================
t = time.time()
print("🔵 Train/Test split...")

X = df.drop(columns=["Severity"])
y = df["Severity"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

log_step("Train/Test split", t)


# ============================================================
# 13) RANDOM FOREST
# ============================================================
t = time.time()
print("🔵 Training RandomForest...")

rf = RandomForestClassifier(n_estimators=60, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
rf_time = time.time() - t

model_log.write(f"RandomForest training: {rf_time:.2f} sec\n")

pred = rf.predict(X_test)
model_log.write(f"RF F1: {f1_score(y_test, pred, average='macro'):.4f}\n")
model_log.write(f"RF Accuracy: {accuracy_score(y_test, pred):.4f}\n")

print(f"   → RF TRAIN TIME = {rf_time:.2f} sec")
print(f"   F1 = {f1_score(y_test, pred, average='macro'):.4f}")
print(f"   ACC = {accuracy_score(y_test, pred):.4f}\n")


# ============================================================
# 14) LOGISTIC REGRESSION
# ============================================================
t = time.time()
print("🔵 Training Logistic Regression...")

lr = LogisticRegression(max_iter=80, n_jobs=-1)
lr.fit(X_train, y_train)
lr_time = time.time() - t

model_log.write(f"Logistic Regression training: {lr_time:.2f} sec\n")

pred = lr.predict(X_test)
model_log.write(f"LR F1: {f1_score(y_test, pred, average='macro'):.4f}\n")
model_log.write(f"LR Accuracy: {accuracy_score(y_test, pred):.4f}\n")

print(f"   → LR TRAIN TIME = {lr_time:.2f} sec")
print(f"   F1 = {f1_score(y_test, pred, average='macro'):.4f}")
print(f"   ACC = {accuracy_score(y_test, pred):.4f}\n")


# ============================================================
# 15) TEMPO TOTALE
# ============================================================
total_time = time.time() - T0
total_log.write(f"Tempo totale pipeline: {total_time:.2f} sec\n")
print("\n====================== FINE PIPELINE ======================\n")
print(f"⏱️  TEMPO TOTALE = {total_time:.2f} sec\n")

# CHIUDI FILE
preproc_log.close()
model_log.close()
total_log.close()

