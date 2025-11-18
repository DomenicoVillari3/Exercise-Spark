# ============================================================
# VERSIONE SEQUENZIALE IN PANDAS — DROP NA E SCALING PER LR
# (Equivalente al comportamento ottimale della pipeline Spark)
# ============================================================

import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
print("📦 Downloading dataset from KaggleHub...")
try:
    dataset_path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
    print("📁 Path to dataset files:", dataset_path)

    csv_path = None
    for f in os.listdir(dataset_path):
        if f.endswith(".csv"):
            csv_path = os.path.join(dataset_path, f)
            break
    
    if csv_path is None:
        raise FileNotFoundError("Nessun CSV trovato.")
except:
    csv_path = "US_Accidents_March23.csv"
    print("⚠️ KaggleHub non riuscito, cerco file locale:", csv_path)

print("📄 CSV target:", csv_path)

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

existing_num = [c for c in num_cols if c in df.columns]
for c in existing_num:
    df[c] = df[c].fillna(df[c].median())

log_step("Riempimento NA numerici", t)

# ============================================================
# 3) DROP COLONNE NON UTILI
# ============================================================
t = time.time()
print("🔵 Rimozione colonne inutili...")

to_drop = [
    'ID', "Source",'Description', 'Distance', 'End_Time', 'Distance(mi)', 'End_Lng', 'End_Lat', 'Duration',"Weather_Timestamp"
]

df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)
log_step("Drop colonne inutili", t)

# ============================================================
# 4) DROP COLONNE CATEGORICHE CON UNA SOLA CATEGORIA
# ============================================================

cat_names = ['Street', 'City', 'County', 'State', "Zipcode",
             'Timezone', 'Airport_Code', 'Wind_Direction',
              'Weather_Condition','Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
            "Sunrise_Sunset","Country","Turning_Loop","Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", 
            "Roundabout", "Station","Stop","Traffic_Calming",
             "Traffic_Signal"]

df.drop(["Country","Turning_Loop"], axis=1, inplace=True)  # Colonne con una sola categoria
cat_names.remove("Country")
cat_names.remove("Turning_Loop")

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
if 'Wind_Direction' in df.columns:
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
    if pd.isna(x): return "Unknown"
    for name, pat in patterns.items():
        if re.search(pat, x, flags=re.IGNORECASE):
            return name
    return "Unknown"

if 'Weather_Condition' in df.columns:
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

# Il codice Spark usa 1-16 (fino alle 16:00) come "Day"
df["Sunrise_Sunset_calc"] = np.where(df["Hour"] <= 16, "Day", "Night")

log_step("Feature engineering date", t)


df.drop(columns=["Start_Time_parsed","Start_Time", "Start_Time_parsed_str","Weather_Timestamp"], inplace=True)
    

# ============================================================
# 7) FILL NA CATEGORICHE
# ============================================================
t = time.time()
print("🔵 Riempimento NA categoriche...")

cat_cols = ["Street", "City", "Zipcode", "Weather_Condition", "Wind_Direction"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")

df["Part_of_Day"] = df["Part_of_Day"].astype(str).replace("nan", "Unknown")

log_step("Riempimento NA categoriche", t)

# ============================================================
# 8) FREQUENCY ENCODING (Identico a Spark)
# ============================================================
t = time.time()
print("🔵 Frequency encoding...")

high_card = ["Street", "City", "Zipcode", "County", "Airport_Code", "State"]

for c in high_card:
    if c in df.columns:
        freq = df[c].value_counts()
        df[c + "_freq"] = df[c].map(freq)
        df.drop(columns=[c], inplace=True)

log_step("Frequency encoding", t)

# ============================================================
# 9) ONE HOT / LABEL ENCODING
# ============================================================
t = time.time()
print("🔵 Encoding finali (Label/OHE)...")

# Label Encoding (StringIndexer Equivalent)
le = LabelEncoder()

if "Part_of_Day" in df.columns:
    df["Part_of_Day_idx"] = le.fit_transform(df["Part_of_Day"].astype(str))
    df.drop(columns=["Part_of_Day"], inplace=True)

if "Wind_Direction" in df.columns:
    df["Wind_Direction_idx"] = le.fit_transform(df["Wind_Direction"].astype(str))
    # Non lo droppiamo ancora, lo facciamo nello step 11 come da logica Spark

# One-Hot Encoding (Solo Timezone e Weather_Condition, le altre sono già state gestite)
oh_cols = ["Weather_Condition", "Timezone"]
cols_to_oh = [c for c in oh_cols if c in df.columns]
df = pd.get_dummies(df, columns=cols_to_oh, drop_first=False) # Lasciamo tutte le categorie

log_step("One-hot e Label Encoding", t)


# ============================================================
# 10) BINARY ENCODING
# ============================================================
t = time.time()
print("🔵 Binary encoding...")


binary_cols = [
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal"
]

def to_bin(x):
    if pd.isna(x): return 0
    x = str(x).strip().lower()
    if x in ["true", "yes", "1", "day"]: return 1
    return 0

for c in binary_cols:
    if c in df.columns:
        df[c + "_bin"] = df[c].apply(to_bin).astype(int)
        df.drop(columns=[c], inplace=True)
    # Usiamo il campo calcolato per Sunrise_Sunset
    if c == "Sunrise_Sunset" and "Sunrise_Sunset_calc" in df.columns:
        df["Sunrise_Sunset_bin"] = (df["Sunrise_Sunset_calc"] == "Day").astype(int)
        df.drop(columns=["Sunrise_Sunset_calc"], inplace=True)


log_step("Binary encoding", t)

# ============================================================
# 11) CORRELAZIONI E DROP (Identico a Spark)
# ============================================================
t = time.time()
print("🔵 Analisi correlazioni e Drop...")

date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
if date_cols:
    df.drop(columns=date_cols, inplace=True)

# Drop features come nello script Spark (quelle con corr ~0)
final_drop = [
    "Civil_Twilight_bin", "Nautical_Twilight_bin", "Astronomical_Twilight_bin",
    "Traffic_Calming_bin", "Bump_bin", "Give_Way_bin",
    "Day", "Visibility(mi)", "Roundabout_bin", "Wind_Direction_idx"
]
df.drop(columns=[c for c in final_drop if c in df.columns], inplace=True)

log_step("Drop features basse correlazioni", t)

# ============================================================
# 🔥 CRITICAL SAFETY CHECK: DROP NA VERSION 🔥
# ============================================================
print("🔵 Esecuzione controllo di sicurezza finale (NaN/Inf)...")

# 1. Rimuovi colonne non numeriche residue
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric) > 0:
    print(f"   → Rimozione colonne non numeriche residue: {non_numeric}")
    df.drop(columns=non_numeric, inplace=True)

# 2. Sostituisci Infinti con NaN per poterli droppare
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. DROPPA LE RIGHE CON NAN (come in Spark's handleInvalid="skip")
rows_before = len(df)
df.dropna(inplace=True)
rows_after = len(df)

print(f"   → Righe eliminate con NaN: {rows_before - rows_after}")
print(f"   → Righe residue: {rows_after}")

# ============================================================
# 12) TRAIN/TEST E SCALING
# ============================================================
t = time.time()
print("🔵 Train/Test split e Scaling...")

if len(df) == 0:
    raise ValueError("❌ ERRORE: Il dataset è vuoto dopo il dropna!")

X = df.drop(columns=["Severity"])
y = df["Severity"]

# Split Stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- SCALING ---
# Lo scaling viene eseguito qui ed è applicato DOPO l'assembler (che in Pandas è implicito)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# =================

print(f"   → Train shape: {X_train.shape}, Test shape: {X_test.shape}")
log_step("Train/Test split e Scaling", t)


# ============================================================
# 13) RANDOM FOREST
# ============================================================
t = time.time()
print("🔵 Training RandomForest...")

# RandomForest usa i dati NON SCALATI (X_train)
# Parametri: numTrees=10, maxDepth=12 (come da ultimo codice Spark fornito)
rf = RandomForestClassifier(n_estimators=10, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_time = time.time() - t

model_log.write(f"RandomForest training: {rf_time:.2f} sec\n")

pred = rf.predict(X_test)
f1_rf = f1_score(y_test, pred, average='macro')
acc_rf = accuracy_score(y_test, pred)
model_log.write(f"RF F1: {f1_rf:.4f}\n")
model_log.write(f"RF Accuracy: {acc_rf:.4f}\n")

print(f"   → RF TRAIN TIME = {rf_time:.2f} sec")
print(f"   F1 = {f1_rf:.4f}")
print(f"   ACC = {acc_rf:.4f}\n")


# ============================================================
# 14) LOGISTIC REGRESSION
# ============================================================
t = time.time()
print("🔵 Training Logistic Regression...")

# Logistic Regression usa i dati SCALATI (X_train_scaled)
# maxIter=50 (come da ultimo codice Spark fornito)
lr = LogisticRegression(max_iter=50, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
lr_time = time.time() - t

model_log.write(f"Logistic Regression training: {lr_time:.2f} sec\n")

pred_lr = lr.predict(X_test_scaled)
f1_lr = f1_score(y_test, pred_lr, average='macro')
acc_lr = accuracy_score(y_test, pred_lr)
model_log.write(f"LR F1: {f1_lr:.4f}\n")
model_log.write(f"LR Accuracy: {acc_lr:.4f}\n")

print(f"   → LR TRAIN TIME = {lr_time:.2f} sec")
print(f"   F1 = {f1_lr:.4f}")
print(f"   ACC = {acc_lr:.4f}\n")

# ============================================================
# 15) TEMPO TOTALE
# ============================================================
total_time = time.time() - T0
total_log.write(f"Tempo totale pipeline: {total_time:.2f} sec\n")
print("\n====================== FINE PIPELINE ======================\n")
print(f"⏱️  TEMPO TOTALE = {total_time:.2f} sec\n")

preproc_log.close()
model_log.close()
total_log.close()