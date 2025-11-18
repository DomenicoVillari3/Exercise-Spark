import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

# Lettura CSV con header implicito
accidents = pd.read_csv("datasets/US_Accidents_March23.csv", header=0)

# Sostituzione NA nelle colonne numeriche con media
num_cols = ["Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
for col in num_cols:
    mean_val = accidents[col].mean()
    accidents[col].fillna(mean_val, inplace=True)

# Drop colonne inutili
accidents.drop(columns=['ID', "Source", 'Description', 'End_Time', 'Distance(mi)', 'End_Lng', 'End_Lat', "Weather_Timestamp", "Country", "Turning_Loop"], inplace=True)

# Simplifica Wind_Direction
wind_dir_mapping = {
    "North": 'N', 'NNE': 'NE', 'NE': 'NE', 'ENE': 'E', 'East': 'E', 'ESE': 'SE', 'SE': 'SE',
    'SSE': 'S', 'South': 'S', 'SSW': 'SW', 'SW': 'SW', 'WSW': 'W', 'West': 'W', 'WNW': 'NW', 'NW': 'NW',
    'NNW': 'NW', "Calm":"CALM", "Variable":"VAR"
}
accidents['Wind_Direction'] = accidents['Wind_Direction'].replace(wind_dir_mapping)

# Semplifica Weather_Condition con regex mapping
patterns = {
    "Clear": r"Clear",
    "Cloud": r"Cloud|Overcast",
    "Rain": r"Rain|storm",
    "Heavy_Rain": r"Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms",
    "Snow": r"Snow|Sleet|Ice",
    "Heavy_Snow": r"Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls",
    "Fog": r"Fog"
}

def map_weather_condition(cond):
    if pd.isna(cond):
        return "Unknown"
    for cat, pattern in patterns.items():
        if pd.Series(cond).str.contains(pattern, case=False, regex=True).any():
            return cat
    return "Unknown"

accidents['Weather_Condition'] = accidents['Weather_Condition'].apply(map_weather_condition)

# Parsing Start_Time e estrazione campi temporali
accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')
accidents['Hour'] = accidents['Start_Time'].dt.hour + 1
accidents['Day'] = accidents['Start_Time'].dt.dayofweek + 1  # Monday=0 in pandas, so offset +1 per allineamento spark
accidents['Month'] = accidents['Start_Time'].dt.month
accidents['Year'] = accidents['Start_Time'].dt.year

# Part_of_Day
def part_of_day(hour):
    if 1 <= hour <= 6:
        return "Night"
    elif 7 <= hour <= 12:
        return "Morning"
    elif 13 <= hour <= 18:
        return "Afternoon"
    else:
        return "Evening"

accidents['Part_of_Day'] = accidents['Hour'].apply(part_of_day)

# Sunrise_Sunset
accidents['Sunrise_Sunset'] = np.where(accidents['Hour'].between(1, 16), "Day", "Night")

# Fillna for categorical columns with "Unknown"
for cat_col in ['Street', 'Zipcode', 'Weather_Condition', 'Wind_Direction', 'City']:
    accidents[cat_col].fillna("Unknown", inplace=True)

# Frequency encoding for high cardinality categorical columns
high_card_cols = ["Street", "City", "Zipcode", "County", "Airport_Code", "State"]
for c in high_card_cols:
    freq = accidents[c].value_counts()
    accidents[c + '_freq'] = accidents[c].map(freq)
accidents.drop(columns=high_card_cols, inplace=True)

# LabelEncoder + OneHotEncoder for small categorical columns
small_cat = ['Weather_Condition', 'Timezone']
le = LabelEncoder()
for c in small_cat:
    accidents[c + '_idx'] = le.fit_transform(accidents[c].fillna("Unknown"))
accidents.drop(columns=small_cat, inplace=True)

# Binary columns encoding
binary_cols = [
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
]
for c in binary_cols:
    accidents[c + '_bin'] = accidents[c].astype(str).str.lower().isin(['true', 'yes', '1', 'day']).astype(int)
accidents.drop(columns=binary_cols, inplace=True)

# Target variable
target = 'Severity'

# Drop less relevant columns (as per your correlation analysis)
to_drop = ["Civil_Twilight_bin", "Nautical_Twilight_bin", "Astronomical_Twilight_bin",
            "Traffic_Calming_bin", "Bump_bin", "Give_Way_bin",
            "Day", "Visibility(mi)", "Roundabout_bin", "Wind_Direction_idx"]
accidents.drop(columns=[col for col in to_drop if col in accidents.columns], inplace=True)

# Stratified train test split
X = accidents.drop(columns=[target])
y = accidents[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Features to be used
feature_cols = X_train.columns.tolist()

# Standardize features if needed (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=12, max_features='auto', random_state=42)
start_time = time.time()
rf.fit(X_train_scaled, y_train)
end_time = time.time()
rf_time = end_time - start_time
print(f"Random Forest training time: {end_time - start_time:.2f} seconds")

y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Logistic Regression
lr = LogisticRegression(max_iter=50, random_state=42)
start_time = time.time()
lr.fit(X_train_scaled, y_train)
end_time = time.time()
logistic_time = end_time - start_time
print(f"Logistic Regression training time: {end_time - start_time:.2f} seconds")

y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression F1 Score:", f1_score(y_test, y_pred_lr, average='weighted'))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

with open("sequential_results.txt", "w") as f:
    f.write(f"Random Forest F1 Score: {f1_score(y_test, y_pred_rf, average='weighted')}\n")
    f.write(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}\n")
    f.write(f"Logistic Regression F1 Score: {f1_score(y_test, y_pred_lr, average='weighted')}\n")
    f.write(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}\n")
    f.write(f"Random Forest training time: {rf_time:.2f} seconds\n")
    f.write(f"Logistic Regression training time: {logistic_time:.2f} seconds\n")
