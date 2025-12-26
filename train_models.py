import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


# ======================================
# 1️⃣ خواندن دیتاست Augmented
# ======================================
df = pd.read_csv("data/cases_augmented.csv")

# فقط ستون‌های مورد نیاز
df = df[["case_text", "crime_type"]]

# حذف ردیف‌هایی که NaN واقعی دارند
df = df.dropna(subset=["case_text", "crime_type"])

# تبدیل متن‌ها به string
df["case_text"] = df["case_text"].astype(str)

# حذف متن‌های خالی یا خیلی کوتاه
df = df[df["case_text"].str.strip().str.len() > 3]

# ======================================
# 2️⃣ تعریف ورودی (X) و خروجی (y)
# ⚠️ بعد از preprocessing
# ======================================
X = df["case_text"]
y = df["crime_type"]

# ======================================
# 3️⃣ تقسیم داده به train / test
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("NaN in X_train:", X_train.isna().sum())
print("NaN in X_test:", X_test.isna().sum())

# ======================================
# 4️⃣ مدل Decision Tree + NLP (TF-IDF)
# ======================================
dt_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        stop_words="english"
    )),
    ("clf", DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ))
])

dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# ======================================
# 5️⃣ مدل SVM + NLP (TF-IDF)
# ======================================
svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        stop_words="english"
    )),
    ("clf", SVC(
        kernel="linear",
        C=1.0
    ))
])

svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)


# ======================================
# 6️⃣ نتایج و گزارش
# ======================================

print("\n📊 Decision Tree Results")
print("Accuracy:", accuracy_score(y_test, dt_preds))
print(classification_report(y_test, dt_preds))


print("\n📊 SVM Results")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

print(confusion_matrix(y_test, dt_preds))
print(confusion_matrix(y_test, svm_preds))

