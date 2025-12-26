import pandas as pd
import random

# خواندن دیتاست
df = pd.read_csv("data/cases.csv")

augmented_rows = []

# دیکشنری پارافریز ساده
text_variations = {
    "fraud": [
        "financial deception involving falsified records",
        "economic fraud with manipulated documents"
    ],
    "theft": [
        "unauthorized taking of property",
        "illegal stealing with minimal evidence"
    ],
    "assault": [
        "physical attack involving multiple individuals",
        "violent assault with reported injuries"
    ]
}

for _, row in df.iterrows():
    augmented_rows.append(row)

    crime = row["crime_type"]
    if crime in text_variations:
        new_row = row.copy()

        # تغییر متن
        new_row["case_text"] = random.choice(text_variations[crime])

        # نویز عددی کوچک
        new_row["processing_days"] += random.randint(-10, 10)
        new_row["text_length"] = len(new_row["case_text"])

        augmented_rows.append(new_row)

# ساخت دیتاست جدید
augmented_df = pd.DataFrame(augmented_rows)

# ذخیره
augmented_df.to_csv("data/cases_augmented.csv", index=False)

print("✅ Dataset augmented successfully!")
print("Original size:", len(df))
print("New size:", len(augmented_df))
