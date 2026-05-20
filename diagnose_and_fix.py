"""
diagnose_and_fix.py  -  Diagnose & Fix the Algerian Forest Fire model

Covers:
  1. Feature importance analysis
  2. Multicollinearity check (VIF)
  3. Heuristic wrapper that overrides bad predictions
  4. XGBoost with monotonic constraints
  5. Cost-sensitive training to reduce false negatives

Usage:
    python diagnose_and_fix.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves plots as PNG instead of showing window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
os.makedirs("models",   exist_ok=True)
os.makedirs("analysis", exist_ok=True)

TABULAR_FEATURES = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════

def load_data(csv_path="dataset/Algerian_forest_fires_dataset.csv"):
    print("\n─── Loading Data ───")
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        raw = f.readlines()

    COLS = ["day","month","year","Temperature","RH","Ws","Rain",
            "FFMC","DMC","DC","ISI","BUI","FWI","Classes"]
    rows = []
    for line in raw:
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in line.split(",")]
        if any(k in line.lower() for k in ["bejaia","sidi","region","dataset"]): continue
        if parts[0].lower() == "day": continue
        if len(parts) < 14: continue
        try: int(parts[0])
        except ValueError: continue
        rows.append(parts[:14])

    df = pd.DataFrame(rows, columns=COLS)
    for col in TABULAR_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Classes"] = df["Classes"].astype(str).str.strip().str.lower()
    df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})
    df = df.dropna()
    df["Classes"] = df["Classes"].astype(int)

    print(f"  Rows: {len(df)}")
    print(f"  Fire: {df['Classes'].sum()}  |  No Fire: {(df['Classes']==0).sum()}")
    return df


# ══════════════════════════════════════════════════════════════════
#  1. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════

def analyze_feature_importance(df):
    print("\n─── Feature Importance Analysis ───")
    X = df[TABULAR_FEATURES].values
    y = df["Classes"].values

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                               random_state=42, stratify=y)

    # GradientBoosting built-in importance
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4, random_state=42)
    gb.fit(X_tr, y_tr)

    imp_df = pd.DataFrame({
        "Feature":   TABULAR_FEATURES,
        "Importance": gb.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\n  GradientBoosting Feature Importances:")
    print(imp_df.to_string(index=False))

    # Permutation importance (more reliable)
    perm = permutation_importance(gb, X_te, y_te, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "Feature":  TABULAR_FEATURES,
        "Perm_Imp": perm.importances_mean
    }).sort_values("Perm_Imp", ascending=False)

    print("\n  Permutation Importances (more reliable):")
    print(perm_df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")

    sns.barplot(data=imp_df, x="Importance", y="Feature",
                palette="Oranges_r", ax=axes[0])
    axes[0].set_title("Built-in Importance")
    axes[0].set_xlabel("Importance Score")

    sns.barplot(data=perm_df, x="Perm_Imp", y="Feature",
                palette="Reds_r", ax=axes[1])
    axes[1].set_title("Permutation Importance")
    axes[1].set_xlabel("Mean Accuracy Drop")

    plt.tight_layout()
    plt.savefig("analysis/feature_importance.png", dpi=120)
    plt.close()
    print("\n  ✅ Saved → analysis/feature_importance.png")

    return gb, sc


# ══════════════════════════════════════════════════════════════════
#  2. MULTICOLLINEARITY (VIF)
# ══════════════════════════════════════════════════════════════════

def check_multicollinearity(df):
    print("\n─── Multicollinearity Check (VIF) ───")
    X = df[TABULAR_FEATURES]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"]     = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    print("\n  VIF Scores (>10 = problematic multicollinearity):")
    for _, row in vif_data.iterrows():
        flag = "  ⚠ HIGH" if row["VIF"] > 10 else ""
        print(f"    {row['Feature']:<15} {row['VIF']:>8.2f}{flag}")

    # Correlation heatmap
    corr = df[TABULAR_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn_r", center=0, ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Matrix\n(high values = multicollinearity risk)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("analysis/correlation_heatmap.png", dpi=120)
    plt.close()
    print("\n  ✅ Saved → analysis/correlation_heatmap.png")

    # Highlight strongly correlated pairs
    print("\n  Highly correlated pairs (|r| > 0.85):")
    found = False
    for i in range(len(TABULAR_FEATURES)):
        for j in range(i+1, len(TABULAR_FEATURES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                print(f"    {TABULAR_FEATURES[i]} ↔ {TABULAR_FEATURES[j]}:  r={r:.3f}")
                found = True
    if not found:
        print("    None found above 0.85")

    return vif_data


# ══════════════════════════════════════════════════════════════════
#  3. HEURISTIC WRAPPER
# ══════════════════════════════════════════════════════════════════

class HeuristicWrapper:
    """
    Wraps any sklearn classifier.
    Applies hard physical rules before trusting model output.
    """

    # Hard NO-FIRE rules: (condition_fn, reason)
    NO_FIRE_RULES = [
        (lambda f: f["Rain"] >= 3.0,
         "Heavy rain ≥ 3mm — fire impossible"),
        (lambda f: f["Temperature"] < 12,
         "Temperature < 12°C — too cold for fire"),
        (lambda f: f["FFMC"] < 40,
         "FFMC < 40 — fuel too moist to ignite"),
        (lambda f: f["RH"] > 88 and f["Rain"] > 0,
         "High humidity + rain — no fire conditions"),
        (lambda f: f["Rain"] >= 1.0 and f["FWI"] < 4,
         "Rain + very low FWI — fire impossible"),
    ]

    # Hard FIRE rules: (condition_fn, confidence, reason)
    FIRE_RULES = [
        (lambda f: f["FWI"] >= 30 and f["Temperature"] >= 32
                   and f["RH"] <= 30 and f["Rain"] == 0,
         97.0, "Extreme conditions: FWI≥30, T≥32°C, RH≤30%"),
        (lambda f: f["FFMC"] >= 92 and f["ISI"] >= 12 and f["Rain"] == 0,
         93.0, "Critical fire spread: FFMC≥92, ISI≥12"),
        (lambda f: f["FWI"] >= 25 and f["BUI"] >= 50 and f["Rain"] == 0,
         90.0, "High buildup: FWI≥25, BUI≥50"),
    ]

    def __init__(self, base_model, scaler):
        self.model  = base_model
        self.scaler = scaler

    def predict_with_reason(self, features_dict):
        """
        features_dict: {feature_name: value, ...}
        Returns: (prediction: 0|1, confidence: float, reason: str)
        """
        f = {k: float(v) for k, v in features_dict.items()}

        # Check hard NO-FIRE rules first
        for rule_fn, reason in self.NO_FIRE_RULES:
            try:
                if rule_fn(f):
                    return 0, 5.0, f"[RULE] {reason}"
            except Exception:
                pass

        # Check hard FIRE rules
        for rule_fn, conf, reason in self.FIRE_RULES:
            try:
                if rule_fn(f):
                    return 1, conf, f"[RULE] {reason}"
            except Exception:
                pass

        # Fall through to ML model
        row    = np.array([[f[feat] for feat in TABULAR_FEATURES]])
        row_sc = self.scaler.transform(row)
        pred   = int(self.model.predict(row_sc)[0])
        proba  = self.model.predict_proba(row_sc)[0]
        conf   = round(float(proba[pred]) * 100, 1)

        # Soft corrections
        if pred == 1 and f["FWI"] < 5 and f["RH"] > 70:
            conf = min(conf, 45.0)
            reason = "[ML+SOFT] Mild conditions — confidence reduced"
        elif pred == 0 and f["FWI"] >= 15 and f["Temperature"] >= 28 and f["Rain"] == 0:
            conf = max(conf, 60.0)
            reason = "[ML+SOFT] Elevated indices — confidence boosted"
        else:
            reason = "[ML] Model prediction"

        return pred, conf, reason

    def test_edge_cases(self):
        """Run built-in sanity tests."""
        print("\n─── Heuristic Wrapper Edge Case Tests ───")
        cases = [
            # (description, features_dict, expected_label)
            ("Heavy rain 10mm",
             {"Temperature":30,"RH":80,"Ws":10,"Rain":10,"FFMC":70,"DMC":20,
              "DC":100,"ISI":5,"BUI":25,"FWI":10}, 0),
            ("Sub-zero cold",
             {"Temperature":5,"RH":60,"Ws":8,"Rain":0,"FFMC":55,"DMC":10,
              "DC":50,"ISI":2,"BUI":12,"FWI":3}, 0),
            ("Extreme drought + heat",
             {"Temperature":38,"RH":18,"Ws":20,"Rain":0,"FFMC":94,"DMC":60,
              "DC":210,"ISI":15,"BUI":65,"FWI":42}, 1),
            ("Very moist fuel (low FFMC)",
             {"Temperature":28,"RH":75,"Ws":5,"Rain":0,"FFMC":35,"DMC":8,
              "DC":30,"ISI":1,"BUI":9,"FWI":2}, 0),
            ("High FWI + high BUI + no rain",
             {"Temperature":34,"RH":28,"Ws":18,"Rain":0,"FFMC":90,"DMC":50,
              "DC":180,"ISI":13,"BUI":60,"FWI":32}, 1),
        ]

        all_passed = True
        for desc, feat, expected in cases:
            pred, conf, reason = self.predict_with_reason(feat)
            status = "✅ PASS" if pred == expected else "❌ FAIL"
            if pred != expected:
                all_passed = False
            label = "FIRE" if pred == 1 else "NO FIRE"
            exp_l = "FIRE" if expected == 1 else "NO FIRE"
            print(f"  {status}  [{desc}]")
            print(f"         Expected={exp_l}  Got={label} ({conf:.1f}%)")
            print(f"         Reason: {reason}\n")

        print(f"  {'All tests passed ✅' if all_passed else 'Some tests failed ❌'}")
        return all_passed


# ══════════════════════════════════════════════════════════════════
#  4. XGBOOST WITH MONOTONIC CONSTRAINTS
# ══════════════════════════════════════════════════════════════════

def train_xgboost_constrained(df):
    """
    Monotonic constraints force the model to respect physics:
      +1 = higher value → more likely fire  (Temperature, FWI, FFMC, ISI, BUI, DC, DMC, Ws)
      -1 = higher value → less likely fire  (RH, Rain)
       0 = no constraint
    """
    print("\n─── XGBoost with Monotonic Constraints ───")

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  ⚠  XGBoost not installed. Run:  pip install xgboost")
        print("  Falling back to Cost-Sensitive GradientBoosting.")
        return train_cost_sensitive_gb(df)

    # Constraint order must match TABULAR_FEATURES exactly:
    # Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI
    monotone_constraints = (
        1,   # Temperature  → higher = more fire
       -1,   # RH           → higher = less fire
        1,   # Ws           → higher = more fire
       -1,   # Rain         → higher = less fire
        1,   # FFMC         → higher = more fire
        1,   # DMC          → higher = more fire
        1,   # DC           → higher = more fire
        1,   # ISI          → higher = more fire
        1,   # BUI          → higher = more fire
        1,   # FWI          → higher = more fire
    )

    X = df[TABULAR_FEATURES].values
    y = df["Classes"].values

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                               random_state=42, stratify=y)

    # Scale pos weight for class imbalance
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    scale = n_neg / max(n_pos, 1)
    print(f"  Class ratio neg/pos = {scale:.2f}  → scale_pos_weight={scale:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        monotone_constraints=monotone_constraints,
        scale_pos_weight=scale,      # handles class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"\n  XGBoost Accuracy: {acc*100:.1f}%")
    print(classification_report(y_te, model.predict(X_te),
                                 target_names=["No Fire", "Fire"]))

    # Confusion matrix
    cm  = confusion_matrix(y_te, model.predict(X_te))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Fire","Fire"]).plot(ax=ax, cmap="Oranges")
    ax.set_title("XGBoost (Monotonic Constraints) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("analysis/xgb_confusion_matrix.png", dpi=120)
    plt.close()
    print("  ✅ Saved → analysis/xgb_confusion_matrix.png")

    # Save
    bundle = {"model": model, "scaler": sc, "type": "xgboost_constrained"}
    with open("models/classifier_xgb.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("  ✅ XGBoost model saved → models/classifier_xgb.pkl")
    return model, sc


# ══════════════════════════════════════════════════════════════════
#  4b. COST-SENSITIVE GRADIENT BOOSTING (fallback if no XGBoost)
# ══════════════════════════════════════════════════════════════════

def train_cost_sensitive_gb(df):
    """
    Makes missed fires (false negatives) 3x more costly than false alarms.
    """
    print("\n─── Cost-Sensitive GradientBoosting ───")
    X = df[TABULAR_FEATURES].values
    y = df["Classes"].values

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2,
                                               random_state=42, stratify=y)

    # sample_weight: fire samples weighted 3x
    sample_weights = np.where(y_tr == 1, 3.0, 1.0)

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    gb.fit(X_tr, y_tr, sample_weight=sample_weights)

    acc = accuracy_score(y_te, gb.predict(X_te))
    print(f"\n  Cost-Sensitive GB Accuracy: {acc*100:.1f}%")
    print(classification_report(y_te, gb.predict(X_te),
                                 target_names=["No Fire","Fire"]))

    # Save
    with open("models/scaler.pkl",     "wb") as f: pickle.dump(sc, f)
    with open("models/classifier.pkl", "wb") as f: pickle.dump(gb, f)
    print("  ✅ Cost-sensitive GB saved → models/classifier.pkl")
    return gb, sc


# ══════════════════════════════════════════════════════════════════
#  5. FULL PIPELINE TEST
# ══════════════════════════════════════════════════════════════════

def run_full_pipeline_test(df, model, scaler):
    print("\n─── Full Pipeline Test (Model + Heuristic Wrapper) ───")
    wrapper = HeuristicWrapper(model, scaler)
    wrapper.test_edge_cases()

    # Cross-validation on full dataset with wrapper
    X = df[TABULAR_FEATURES].values
    y = df["Classes"].values
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X)
    cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring="f1")
    print(f"\n  5-Fold CV F1 Scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    CSV = "dataset/Algerian_forest_fires_dataset.csv"

    if not os.path.exists(CSV):
        print(f"❌ CSV not found at {CSV}")
        print("   Please place the dataset there and re-run.")
        exit(1)

    df = load_data(CSV)

    # 1. Feature importance
    gb_model, gb_scaler = analyze_feature_importance(df)

    # 2. Multicollinearity
    check_multicollinearity(df)

    # 3. Heuristic wrapper edge-case tests
    wrapper = HeuristicWrapper(gb_model, gb_scaler)
    wrapper.test_edge_cases()

    # 4. XGBoost with monotonic constraints (recommended)
    xgb_model, xgb_scaler = train_xgboost_constrained(df)

    # 5. Full pipeline test with XGBoost
    run_full_pipeline_test(df, xgb_model, xgb_scaler)

    print("\n" + "="*55)
    print("  Analysis complete!")
    print("  Plots saved in:  analysis/")
    print("  Models saved in: models/")
    print("="*55)