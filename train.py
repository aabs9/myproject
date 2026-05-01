# ===================================================
# Intrusion Detection System (IDS)
# Multi-Class Classification with SMOTE
# Dataset: CICIDS2017
# Models: Random Forest + SVM
# ===================================================
# تثبيت المكتبات (شغّل مرة واحدة في Terminal):
# pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib
# ===================================================

import pandas as pd
import numpy as np
import glob
import joblib
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # مهم لـ PyCharm حتى تظهر الرسوم
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from imblearn.over_sampling import SMOTE

# ===============================
# ⚙️ المسار — غيّره حسب مكان ملفاتك
# ===============================
DATA_FOLDER = r"E:\dataset"   # ← غيّر هذا المسار
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__)) # يحفظ الملفات بجانب السكريبت

# ===============================
# 1️⃣ تحميل ودمج ملفات CSV
# ===============================
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if not files:
    raise FileNotFoundError(
        f"لم يتم العثور على ملفات CSV في: {DATA_FOLDER}\n"
        "تأكد من المسار أعلاه"
    )

print(f"تم العثور على {len(files)} ملف CSV...")
df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
print("Full Dataset shape:", df.shape)

# ===============================
# 2️⃣ أخذ عينة لتجنب البطء
# ===============================
sample_size = 50000
df = df.sample(n=min(sample_size, len(df)), random_state=42)
print("Sampled Dataset shape:", df.shape)

# ===============================
# 3️⃣ تنظيف أسماء الأعمدة
# ===============================
df.columns = df.columns.str.strip()

# ===============================
# 4️⃣ حصر الفئات المطلوبة فقط
# ===============================
allowed_classes = [
    'BENIGN', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
    'DoS slowloris', 'FTP-Patator', 'PortScan', 'SSH-Patator'
]

df = df[df['Label'].isin(allowed_classes)]
print("Filtered Dataset shape:", df.shape)
print("Remaining classes:", df['Label'].unique())

# ===============================
# 5️⃣ Features & Labels
# ===============================
drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
cols_to_drop = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=cols_to_drop + ['Label'])
y = df['Label']

# ===============================
# 6️⃣ تنظيف القيم غير الصالحة
# ===============================
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y[X.index]
X = X.clip(-1e6, 1e6)

# ===============================
# 7️⃣ Normalization
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 8️⃣ Encoding Labels
# ===============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===============================
# 9️⃣ Train / Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

# ===============================
# 🔟 إزالة الفئات الصغيرة جداً
# ===============================
unique, counts = np.unique(y_train, return_counts=True)
small_classes = unique[counts < 2]

mask = ~np.isin(y_train, small_classes)
X_train_f = X_train[mask]
y_train_f = y_train[mask]

if len(small_classes) > 0:
    print("Removed small classes:", le.inverse_transform(small_classes))

# ===============================
# 1️⃣1️⃣ SMOTE
# ===============================
unique2, counts2 = np.unique(y_train_f, return_counts=True)
min_count    = counts2.min()
k_neighbors  = max(1, min_count - 1)

smote = SMOTE(random_state=42, sampling_strategy='not majority', k_neighbors=k_neighbors)
X_train_res, y_train_res = smote.fit_resample(X_train_f, y_train_f)

print("Before SMOTE:", np.bincount(y_train_f))
print("After  SMOTE:", np.bincount(y_train_res))

# ===============================
# 1️⃣2️⃣ تدريب النماذج
# ===============================
print("\n⏳ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
print("✅ Random Forest done")

print("⏳ Training SVM (قد يأخذ وقتاً أطول)...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_res, y_train_res)
y_pred_svm = svm.predict(X_test)
print("✅ SVM done")

# ===============================
# 1️⃣3️⃣ قائمة الفئات الموجودة في الاختبار
# ===============================
existing_classes = np.unique(y_test)
target_names     = le.inverse_transform(existing_classes)

# ===============================
# 1️⃣4️⃣ دالة طباعة النتائج
# ===============================
def print_results(y_true, y_pred, model_name):
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
    print(classification_report(y_true, y_pred,
                                labels=existing_classes,
                                target_names=target_names))

    cm = confusion_matrix(y_true, y_pred, labels=existing_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_{model_name.replace(' ', '_')}.png"), dpi=150)
    plt.show()

print_results(y_test, y_pred_rf,  "Random Forest")
print_results(y_test, y_pred_svm, "SVM")

# ===============================
# 1️⃣5️⃣ جدول الأداء التفصيلي
# ===============================
metrics_df = pd.DataFrame({
    "Class":        target_names,
    "Precision_RF": precision_score(y_test, y_pred_rf,  labels=existing_classes, average=None),
    "Recall_RF":    recall_score   (y_test, y_pred_rf,  labels=existing_classes, average=None),
    "F1_RF":        f1_score       (y_test, y_pred_rf,  labels=existing_classes, average=None),
    "Precision_SVM":precision_score(y_test, y_pred_svm, labels=existing_classes, average=None),
    "Recall_SVM":   recall_score   (y_test, y_pred_svm, labels=existing_classes, average=None),
    "F1_SVM":       f1_score       (y_test, y_pred_svm, labels=existing_classes, average=None),
})

print("\n===== Performance Table =====")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "performance_table.csv"), index=False)

# ===============================
# 1️⃣6️⃣ رسم مقارنة F1-Score
# ===============================
x     = np.arange(len(target_names))
width = 0.35

plt.figure(figsize=(14, 6))
plt.bar(x - width/2, metrics_df["F1_RF"],  width, label="Random Forest", alpha=0.8, color='steelblue')
plt.bar(x + width/2, metrics_df["F1_SVM"], width, label="SVM",           alpha=0.8, color='coral')
plt.xticks(x, target_names, rotation=40, ha='right')
plt.ylabel("F1-score")
plt.title("F1-score per Class — RF vs SVM")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_comparison.png"), dpi=150)
plt.show()

# ===============================
# 1️⃣7️⃣ حفظ النماذج والأدوات
# ===============================
joblib.dump(rf,     os.path.join(OUTPUT_DIR, "rf_ids_model.pkl"))
joblib.dump(svm,    os.path.join(OUTPUT_DIR, "svm_ids_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(le,     os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

print("\n✅ Models saved successfully in:", OUTPUT_DIR)
print("   rf_ids_model.pkl")
print("   svm_ids_model.pkl")
print("   scaler.pkl")
print("   label_encoder.pkl")
