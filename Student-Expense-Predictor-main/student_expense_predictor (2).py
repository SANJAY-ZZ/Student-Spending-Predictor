# ============================================================
#   Student Expense Predictor — Logistic Regression
#   Step 1: Generate & save dataset as CSV
#   Step 2: Load CSV and run full ML pipeline
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ── STEP 1: GENERATE & SAVE DATASET AS CSV ────────────────────────────────────
np.random.seed(42)
n = 300

pocket_money  = np.random.randint(2000, 8000, n)
food          = np.random.randint(500,  3000, n)
travel        = np.random.randint(200,  1500, n)
entertainment = np.random.randint(100,  2000, n)
other         = np.random.randint(100,  1500, n)

total_expense = food + travel + entertainment + other
overspent     = (total_expense > 0.80 * pocket_money).astype(int)

df_raw = pd.DataFrame({
    "Pocket_Money":  pocket_money,
    "Food":          food,
    "Travel":        travel,
    "Entertainment": entertainment,
    "Other":         other,
    "Overspent":     overspent
})

csv_path = "student_expenses.csv"
df_raw.to_csv(csv_path, index=False)
print("=" * 55)
print("  Dataset saved as 'student_expenses.csv'")
print(f"     Rows: {len(df_raw)}  |  Columns: {list(df_raw.columns)}")
print("=" * 55)

# ── STEP 2: LOAD FROM CSV ─────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
print(f"\nLoaded CSV: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"   Overspent     : {df['Overspent'].sum()} ({df['Overspent'].mean()*100:.1f}%)")
print(f"   Not Overspent : {(df['Overspent']==0).sum()} ({(df['Overspent']==0).mean()*100:.1f}%)")
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# ── STEP 3: FEATURE ENGINEERING ───────────────────────────────────────────────
df["Total_Expense"] = df[["Food","Travel","Entertainment","Other"]].sum(axis=1)
df["Expense_Ratio"] = df["Total_Expense"] / df["Pocket_Money"]
df["Savings"]       = df["Pocket_Money"]  - df["Total_Expense"]

features = ["Pocket_Money","Food","Travel","Entertainment",
            "Other","Total_Expense","Expense_Ratio","Savings"]
X = df[features]
y = df["Overspent"]

# ── STEP 4: TRAIN/TEST SPLIT + SCALING ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── STEP 5: LOGISTIC REGRESSION ───────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

y_pred   = model.predict(X_test_sc)
y_prob   = model.predict_proba(X_test_sc)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print(f"  MODEL ACCURACY : {accuracy * 100:.2f}%")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not Overspent", "Overspent"]))

# ── STEP 6: VISUALISATIONS ────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
fig = plt.figure(figsize=(18, 13), facecolor="#0f1117")
fig.suptitle("Student Expense Predictor — Logistic Regression  |  Data: student_expenses.csv",
             fontsize=16, fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
LABEL_COLOR = "white"
TICK_COLOR  = "#aaaaaa"

def style_ax(ax, title):
    ax.set_facecolor("#1a1d27")
    ax.set_title(title, color=LABEL_COLOR, fontsize=11, pad=10)
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.xaxis.label.set_color(TICK_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)

# Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Overspent","Overspent"],
            yticklabels=["Not Overspent","Overspent"],
            ax=ax1, linewidths=1, linecolor="#333")
ax1.set_facecolor("#1a1d27")
ax1.set_title("Confusion Matrix", color=LABEL_COLOR, fontsize=11, pad=10)
ax1.tick_params(colors=TICK_COLOR, labelsize=8)
ax1.set_xlabel("Predicted", color=TICK_COLOR)
ax1.set_ylabel("Actual",    color=TICK_COLOR)

# ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color="#4fc3f7", lw=2, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0,1],[0,1], color="#555566", lw=1, linestyle="--")
ax2.fill_between(fpr, tpr, alpha=0.15, color="#4fc3f7")
ax2.legend(facecolor="#1a1d27", labelcolor=LABEL_COLOR, fontsize=9)
style_ax(ax2, "ROC Curve")

# Feature Coefficients
ax3   = fig.add_subplot(gs[0, 2])
coefs = pd.Series(model.coef_[0], index=features).sort_values()
colors = ["#ef5350" if c > 0 else "#42a5f5" for c in coefs]
coefs.plot(kind="barh", ax=ax3, color=colors, edgecolor="none")
ax3.axvline(0, color="#aaaaaa", linewidth=0.8)
style_ax(ax3, "Feature Coefficients\n(Red = increases risk)")

# Expense Ratio Distribution
ax4 = fig.add_subplot(gs[1, 0])
for label, grp_color in [(0,"#42a5f5"),(1,"#ef5350")]:
    subset = df[df["Overspent"]==label]["Expense_Ratio"]
    ax4.hist(subset, bins=25, alpha=0.7, color=grp_color,
             label="Not Overspent" if label==0 else "Overspent",
             edgecolor="none")
ax4.axvline(0.80, color="yellow", lw=1.5, linestyle="--", label="80% threshold")
ax4.legend(facecolor="#1a1d27", labelcolor=LABEL_COLOR, fontsize=8)
style_ax(ax4, "Expense Ratio Distribution")

# Predicted Probability Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(y_prob[y_test==0], bins=25, alpha=0.7,
         color="#42a5f5", label="Not Overspent", edgecolor="none")
ax5.hist(y_prob[y_test==1], bins=25, alpha=0.7,
         color="#ef5350", label="Overspent", edgecolor="none")
ax5.axvline(0.5, color="yellow", lw=1.5, linestyle="--", label="Decision boundary")
ax5.legend(facecolor="#1a1d27", labelcolor=LABEL_COLOR, fontsize=8)
style_ax(ax5, "Predicted Probability Distribution")

# Summary Card
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#1a1d27"); ax6.set_xlim(0,1); ax6.set_ylim(0,1); ax6.axis("off")
ax6.text(0.5, 0.88, "Model Summary", ha="center", va="top",
         fontsize=13, color=LABEL_COLOR, fontweight="bold")
metrics = [
    ("Accuracy",      f"{accuracy*100:.2f}%"),
    ("AUC-ROC",       f"{roc_auc:.4f}"),
    ("Dataset",       "student_expenses.csv"),
    ("Train Samples", str(len(X_train))),
    ("Test Samples",  str(len(X_test))),
    ("Algorithm",     "Logistic Regression"),
]
for i, (k, v) in enumerate(metrics):
    yp = 0.72 - i * 0.12
    ax6.text(0.08, yp, k+":", ha="left", va="center", fontsize=9, color="#aaaaaa")
    ax6.text(0.92, yp, v,    ha="right", va="center", fontsize=9,
             color="#4fc3f7", fontweight="bold")
    ax6.axhline(yp-0.04, color="#333344", linewidth=0.5, xmin=0.05, xmax=0.95)
ax6.set_title("Performance Metrics", color=LABEL_COLOR, fontsize=11, pad=10)
for spine in ax6.spines.values():
    spine.set_edgecolor("#333344")

plt.savefig("student_expense_chart.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("\nGraph saved -> student_expense_chart.png")

# ── STEP 7: PREDICT ON NEW STUDENTS ──────────────────────────────────────────
print("\n" + "=" * 55)
print("  PREDICTION ON NEW STUDENT DATA")
print("=" * 55)
new_students = pd.DataFrame({
    "Pocket_Money":  [5000, 3000, 7000],
    "Food":          [1500,  800, 1200],
    "Travel":        [ 500,  400,  300],
    "Entertainment": [ 800, 1000,  500],
    "Other":         [ 400,  600,  200],
})
new_students["Total_Expense"] = new_students[["Food","Travel","Entertainment","Other"]].sum(axis=1)
new_students["Expense_Ratio"] = new_students["Total_Expense"] / new_students["Pocket_Money"]
new_students["Savings"]       = new_students["Pocket_Money"]  - new_students["Total_Expense"]

new_sc = scaler.transform(new_students[features])
preds  = model.predict(new_sc)
probs  = model.predict_proba(new_sc)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "OVERSPENT" if pred == 1 else "Within Budget"
    print(f"  Student {i+1}: {label}  (risk = {prob*100:.1f}%)")

print("\nDone!")
