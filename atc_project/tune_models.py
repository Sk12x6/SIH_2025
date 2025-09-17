# tune_models.py
import joblib, pandas as pd, numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# load data (same preprocessing as pipeline)
df = pd.read_csv("data/cow_data.csv")
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

# add engineered features (same as pipeline)
df["bmi"] = df["body_weight"] / (df["height_at_withers"].replace(0, np.nan) ** 2)
df["chest_ratio"] = df["chest_width"] / df["body_length"].replace(0, np.nan)
df["milk_per_age"] = df["historical_milk_yield"] / df["age"].replace(0, np.nan)
df["efficiency_index"] = df["historical_milk_yield"] / df["parity"].replace(0, np.nan)
df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

features = ["age","body_weight","height_at_withers","body_length","chest_width","parity","historical_milk_yield","bmi","chest_ratio","milk_per_age","efficiency_index"]
X = df[features]

# RandomForest reg param grid (sampled)
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_dist = {
    "n_estimators": [100,200,300,400],
    "max_depth": [None,8,12,16],
    "min_samples_split": [2,4,6],
    "min_samples_leaf": [1,2,3],
    "max_features": ["auto","sqrt","log2"]
}

kf = KFold(n_splits=min(5, max(2, len(df))), shuffle=True, random_state=42)
rs = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=kf, scoring="r2", random_state=42, n_jobs=-1, verbose=2)
rs.fit(X, df["milk_productivity"])   # example for milk productivity
print("Best params (milk):", rs.best_params_, "best r2:", rs.best_score_)
joblib.dump(rs.best_estimator_, "models/milk_productivity_tuned.pkl")

# Repeat for longevity and reproductive; classifier tuning:
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_dist_clf = {
    "n_estimators": [100,200,300],
    "max_depth": [None,8,12],
    "min_samples_split": [2,4,6],
    "min_samples_leaf": [1,2,3],
    "class_weight": ["balanced", None]
}
skf = StratifiedKFold(n_splits=min(5, max(2, len(df))), shuffle=True, random_state=42)
rs_clf = RandomizedSearchCV(clf, param_dist_clf, n_iter=20, cv=skf, scoring="accuracy", random_state=42, n_jobs=-1, verbose=2)
rs_clf.fit(X, df["elite_dam"])
print("Best classifier params:", rs_clf.best_params_, "best acc:", rs_clf.best_score_)
joblib.dump(rs_clf.best_estimator_, "models/elite_dam_tuned.pkl")
