
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("cases.csv")


# -------------------------
# 2. NLP: Stronger TF-IDF
# -------------------------
tfidf = TfidfVectorizer(
    max_features=50,
    ngram_range=(1, 2),
    stop_words="english"
)

text_features = tfidf.fit_transform(df["case_text"])

text_df = pd.DataFrame(
    text_features.toarray(),
    columns=tfidf.get_feature_names_out()
)


# -------------------------
# 3. Encode Categorical Data
# -------------------------
df_no_text = df.drop(columns=["case_text"])

df_encoded = pd.get_dummies(
    df_no_text,
    columns=["crime_type", "complexity"],
    drop_first=True
)


# -------------------------
# 4. Combine All Features
# -------------------------
df_final = pd.concat(
    [
        df_encoded.drop("processing_days", axis=1).reset_index(drop=True),
        text_df.reset_index(drop=True),
        df_encoded["processing_days"].reset_index(drop=True)
    ],
    axis=1
)


# -------------------------
# 5. Split Input / Output
# -------------------------
X = df_final.drop("processing_days", axis=1)
y = df_final["processing_days"]

feature_names = X.columns


# -------------------------
# 6. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -------------------------
# 7. Decision Tree (Nonlinear, Rule-Based)
# -------------------------
dt_model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)


# -------------------------
# 8. SVM Pipeline (Scaled, Smooth)
# -------------------------
svm_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svm", SVR(
            kernel="rbf",
            C=50,
            epsilon=5,
            gamma="scale"
        ))
    ]
)

svm_pipeline.fit(X_train, y_train)
svm_pred = svm_pipeline.predict(X_test)

svm_mae = mean_absolute_error(y_test, svm_pred)


# -------------------------
# 9. Results
# -------------------------
print("Decision Tree MAE:", round(dt_mae, 2))
print("SVM MAE:", round(svm_mae, 2))


# -------------------------
# 10. Predict New Case (Correct Way)
# -------------------------
new_case = pd.DataFrame(0, columns=feature_names, index=[0])

new_case["num_defendants"] = 10
new_case["num_documents"] = 14
new_case["text_length"] = 5000
new_case["crime_type_fraud"] = 1
new_case["complexity_medium"] = 1

dt_prediction = dt_model.predict(new_case)
svm_prediction = svm_pipeline.predict(new_case)

print("\nPredicted processing days (Decision Tree):", int(dt_prediction[0]))
print("Predicted processing days (SVM):", int(svm_prediction[0]))


# -------------------------
# 11. Top NLP Keywords
# -------------------------
print("\nTop NLP keywords used:")
print(tfidf.get_feature_names_out()[:15])
