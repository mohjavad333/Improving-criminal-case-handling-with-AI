# =========================
# AI Court Project
# Predicting Case Processing Time
# =========================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("cases.csv")


# -------------------------
# 2. NLP: TF-IDF on case text
# -------------------------
tfidf = TfidfVectorizer(max_features=5)
text_features = tfidf.fit_transform(df["case_text"])

text_df = pd.DataFrame(
    text_features.toarray(),
    columns=[f"text_feat_{i}" for i in range(text_features.shape[1])]
)


# -------------------------
# 3. Encode Categorical Data
# -------------------------
df_no_text = df.drop(columns=["case_text"])
df_encoded = pd.get_dummies(df_no_text, columns=["crime_type", "complexity"])

df_final = pd.concat(
    [
        df_encoded.drop("processing_days", axis=1),
        text_df,
        df_encoded["processing_days"]
    ],
    axis=1
)


# -------------------------
# 4. Split Input and Output
# -------------------------
X = df_final.drop("processing_days", axis=1)
y = df_final["processing_days"]

feature_names = X.columns


# -------------------------
# 5. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -------------------------
# 6. Decision Tree Model
# -------------------------
dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_pred)


# -------------------------
# 7. SVM Model
# -------------------------
svm_model = SVR(kernel="rbf")
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_mae = mean_absolute_error(y_test, svm_pred)


# -------------------------
# 8. Results
# -------------------------
print("Decision Tree MAE:", dt_mae)
print("SVM MAE:", svm_mae)


# -------------------------
# 9. Predict New Case
# -------------------------
new_case = pd.DataFrame(columns=feature_names)
new_case.loc[0] = 0

new_case.at[0, "num_defendants"] = 2
new_case.at[0, "num_documents"] = 14
new_case.at[0, "text_length"] = 700
new_case.at[0, "crime_type_fraud"] = 1
new_case.at[0, "complexity_medium"] = 1

dt_prediction = dt_model.predict(new_case)
svm_prediction = svm_model.predict(new_case)

print("\nPredicted processing days (Decision Tree):", int(dt_prediction[0]))
print("Predicted processing days (SVM):", int(svm_prediction[0]))

print("\nTop NLP keywords:")
print(tfidf.get_feature_names_out())
