"""
Anti-Money Laundering (AML) Detection System
for SME Banks & Mid-Size Financial Institutions

Author  : Saiful (saifulcu105-lang)
Target  : Community banks, credit unions, regional banks
Purpose : Detect money laundering patterns in real time
NIW     : Supports US Bank Secrecy Act & FinCEN AML compliance

References:
- FinCEN AML National Priorities (2022)
- Bank Secrecy Act (BSA) 31 U.S.C. 5311
- USA PATRIOT Act Section 326
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. GENERATE AML TRANSACTION DATA
# ─────────────────────────────────────────────
def generate_aml_data(n=8000, suspicious_rate=0.03):
    """
    Synthetic AML transaction dataset for SME bank training.
    Mimics real-world money laundering patterns:
    - Structuring (smurfing)
    - Round-trip transactions
    - Layering through multiple accounts
    - Shell company transactions
    """
    np.random.seed(42)
    n_clean = int(n * (1 - suspicious_rate))
    n_sus   = int(n * suspicious_rate)

    # Normal transactions
    clean = pd.DataFrame({
        'amount':              np.random.lognormal(5, 1.5, n_clean),
        'num_transactions':    np.random.poisson(4, n_clean),
        'unique_recipients':   np.random.randint(1, 6, n_clean),
        'cross_border':        np.random.binomial(1, 0.05, n_clean),
        'cash_intensive':      np.random.binomial(1, 0.1, n_clean),
        'round_amount':        np.random.binomial(1, 0.15, n_clean),
        'velocity_24h':        np.random.poisson(2, n_clean),
        'avg_txn_gap_hours':   np.random.exponential(12, n_clean),
        'shell_company_flag':  np.random.binomial(1, 0.02, n_clean),
        'high_risk_country':   np.random.binomial(1, 0.03, n_clean),
        'structuring_flag':    np.random.binomial(1, 0.05, n_clean),
        'is_suspicious':       0
    })

    # Suspicious (money laundering) transactions
    sus = pd.DataFrame({
        'amount':              np.random.choice(
                                   [9000, 9500, 9800, 9900, 8500], n_sus),  # structuring
        'num_transactions':    np.random.randint(10, 50, n_sus),
        'unique_recipients':   np.random.randint(8, 25, n_sus),
        'cross_border':        np.random.binomial(1, 0.7, n_sus),
        'cash_intensive':      np.random.binomial(1, 0.8, n_sus),
        'round_amount':        np.random.binomial(1, 0.9, n_sus),
        'velocity_24h':        np.random.randint(8, 30, n_sus),
        'avg_txn_gap_hours':   np.random.exponential(1, n_sus),
        'shell_company_flag':  np.random.binomial(1, 0.6, n_sus),
        'high_risk_country':   np.random.binomial(1, 0.5, n_sus),
        'structuring_flag':    np.random.binomial(1, 0.9, n_sus),
        'is_suspicious':       1
    })

    data = pd.concat([clean, sus], ignore_index=True).sample(frac=1, random_state=42)
    return data


# ─────────────────────────────────────────────
# 2. AML DETECTION MODEL
# ─────────────────────────────────────────────
class AMLDetector:
    """
    AI-powered Anti-Money Laundering detection for SME banks.

    Detects:
    - Structuring (breaking large amounts into smaller ones)
    - Layering (moving money through multiple accounts)
    - Shell company transactions
    - High-risk country transfers
    - Unusual transaction velocity
    """

    FEATURES = [
        'amount', 'num_transactions', 'unique_recipients',
        'cross_border', 'cash_intensive', 'round_amount',
        'velocity_24h', 'avg_txn_gap_hours',
        'shell_company_flag', 'high_risk_country', 'structuring_flag'
    ]

    AML_PATTERNS = {
        'structuring':     'Breaking transactions below $10,000 to avoid reporting',
        'layering':        'Moving funds through multiple accounts rapidly',
        'shell_company':   'Transactions involving shell/front companies',
        'high_risk':       'Transfers to/from high-risk jurisdictions',
        'velocity':        'Unusually high number of transactions in 24 hours',
        'round_tripping':  'Funds leaving and returning through different paths',
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight='balanced',
            random_state=42
        )
        self.trained = False

    def train(self, data: pd.DataFrame):
        print("\n🔄 Training AML Detection Model...")
        X = self.scaler.fit_transform(data[self.FEATURES])
        y = data['is_suspicious']

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(X_tr, y_tr)
        self.trained = True

        y_pred = self.model.predict(X_te)
        y_prob = self.model.predict_proba(X_te)[:, 1]

        print("\n" + "="*58)
        print("  AML DETECTION MODEL — PERFORMANCE RESULTS")
        print("="*58)
        print(f"  Precision : {precision_score(y_te, y_pred):.1%}")
        print(f"  Recall    : {recall_score(y_te, y_pred):.1%}")
        print(f"  F1 Score  : {f1_score(y_te, y_pred):.1%}")
        print(f"  AUC-ROC   : {roc_auc_score(y_te, y_prob):.1%}")
        print("="*58)
        print(classification_report(y_te, y_pred,
              target_names=['Clean', 'Suspicious']))
        return self

    def screen_transaction(self, txn: dict) -> dict:
        """Screen a single transaction for AML risk."""
        if not self.trained:
            raise ValueError("Train model first.")

        df   = pd.DataFrame([txn])[self.FEATURES]
        X    = self.scaler.transform(df)
        prob = self.model.predict_proba(X)[0][1]
        risk = int(prob * 100)

        # Identify triggered AML patterns
        patterns = []
        if txn.get('structuring_flag'):    patterns.append('Structuring')
        if txn.get('shell_company_flag'):  patterns.append('Shell Company')
        if txn.get('high_risk_country'):   patterns.append('High-Risk Country')
        if txn.get('velocity_24h', 0) > 7: patterns.append('High Velocity')
        if txn.get('cross_border'):        patterns.append('Cross-Border')

        # SAR = Suspicious Activity Report (required by BSA)
        sar_required = risk >= 60

        if risk < 30:
            level, action, emoji = "LOW",    "✅ CLEAR",             "🟢"
        elif risk < 60:
            level, action, emoji = "MEDIUM", "⚠️  ENHANCED DUE DILIGENCE", "🟡"
        else:
            level, action, emoji = "HIGH",   "🚨 FILE SAR REPORT",   "🔴"

        return {
            'risk_score':    risk,
            'risk_level':    level,
            'action':        action,
            'emoji':         emoji,
            'sar_required':  sar_required,
            'patterns':      patterns if patterns else ['None detected'],
            'aml_prob':      f"{prob:.1%}"
        }


# ─────────────────────────────────────────────
# 3. SUSPICIOUS ACTIVITY REPORT GENERATOR
# ─────────────────────────────────────────────
class SARGenerator:
    """
    Automatically generates Suspicious Activity Report (SAR)
    summaries for FinCEN filing — required by Bank Secrecy Act.
    Saves SME bank compliance officers hours of manual work.
    """

    def generate_sar_summary(self, account_id: str,
                              txn: dict, screening: dict) -> str:
        patterns = ', '.join(screening['patterns'])
        return f"""
╔══════════════════════════════════════════════════════════╗
║         SUSPICIOUS ACTIVITY REPORT (SAR) SUMMARY        ║
║              Bank Secrecy Act Compliance                 ║
╠══════════════════════════════════════════════════════════╣
  Account ID     : {account_id}
  Risk Score     : {screening['risk_score']}/100
  Risk Level     : {screening['risk_level']}
  AML Probability: {screening['aml_prob']}

  DETECTED PATTERNS:
  {patterns}

  TRANSACTION DETAILS:
  Amount         : ${txn.get('amount', 0):,.2f}
  Transactions/day: {txn.get('num_transactions', 0)}
  Cross-Border   : {'Yes' if txn.get('cross_border') else 'No'}
  High-Risk Ctry : {'Yes' if txn.get('high_risk_country') else 'No'}
  Shell Company  : {'Yes' if txn.get('shell_company_flag') else 'No'}

  RECOMMENDED ACTION:
  {screening['action']}

  SAR FILING REQUIRED: {'YES — File within 30 days' if screening['sar_required'] else 'NO'}

  Regulatory Reference:
  - Bank Secrecy Act 31 U.S.C. § 5318(g)
  - FinCEN SAR Filing Instructions
  - USA PATRIOT Act Section 351
╚══════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────
# 4. PORTFOLIO AML SCANNER
# ─────────────────────────────────────────────
class PortfolioAMLScanner:
    """Scan entire customer portfolio for AML risk."""

    def scan(self, detector: AMLDetector, data: pd.DataFrame):
        results = []
        for _, row in data.head(500).iterrows():
            r = detector.screen_transaction(row.to_dict())
            results.append(r['risk_level'])

        total = len(results)
        low    = results.count('LOW')
        medium = results.count('MEDIUM')
        high   = results.count('HIGH')

        print("\n" + "="*58)
        print("  PORTFOLIO AML SCAN RESULTS")
        print("="*58)
        print(f"  Accounts Scanned : {total:,}")
        print(f"  🟢 Low Risk      : {low:,}  ({low/total:.1%})")
        print(f"  🟡 Medium Risk   : {medium:,} ({medium/total:.1%})")
        print(f"  🔴 High Risk     : {high:,}  ({high/total:.1%})")
        print(f"\n  📋 SARs to File  : {high:,}")
        print(f"  ⏱️  Time Saved    : ~{high * 2} hours of manual review")
        print("="*58)


# ─────────────────────────────────────────────
# 5. MAIN DEMO
# ─────────────────────────────────────────────
def main():
    print("="*58)
    print("  ANTI-MONEY LAUNDERING DETECTION SYSTEM")
    print("  Author  : Saiful (saifulcu105-lang)")
    print("  Targets : SME Banks & Credit Unions")
    print("  Law     : Bank Secrecy Act / FinCEN AML")
    print("="*58)

    # Generate & train
    print("\n📊 Loading transaction data...")
    data = generate_aml_data(8000, suspicious_rate=0.03)
    print(f"   Total transactions  : {len(data):,}")
    print(f"   Suspicious flagged  : {data['is_suspicious'].sum():,}")

    detector = AMLDetector()
    detector.train(data)

    sar_gen = SARGenerator()

    # ── Test 1: Clean transaction ──
    print("\n" + "─"*58)
    print("TEST 1 — Normal Business Transaction")
    print("─"*58)
    clean_txn = {
        'amount': 2500, 'num_transactions': 3,
        'unique_recipients': 2, 'cross_border': 0,
        'cash_intensive': 0, 'round_amount': 0,
        'velocity_24h': 2, 'avg_txn_gap_hours': 24,
        'shell_company_flag': 0, 'high_risk_country': 0,
        'structuring_flag': 0
    }
    r1 = detector.screen_transaction(clean_txn)
    print(f"  Amount       : ${clean_txn['amount']:,}")
    print(f"  Risk Score   : {r1['emoji']} {r1['risk_score']}/100")
    print(f"  Risk Level   : {r1['risk_level']}")
    print(f"  Action       : {r1['action']}")
    print(f"  SAR Required : {'Yes' if r1['sar_required'] else 'No'}")

    # ── Test 2: Suspicious (structuring) ──
    print("\n" + "─"*58)
    print("TEST 2 — Suspicious Transaction (Structuring)")
    print("─"*58)
    sus_txn = {
        'amount': 9500, 'num_transactions': 22,
        'unique_recipients': 15, 'cross_border': 1,
        'cash_intensive': 1, 'round_amount': 1,
        'velocity_24h': 18, 'avg_txn_gap_hours': 0.5,
        'shell_company_flag': 1, 'high_risk_country': 1,
        'structuring_flag': 1
    }
    r2 = detector.screen_transaction(sus_txn)
    print(f"  Amount       : ${sus_txn['amount']:,}")
    print(f"  Risk Score   : {r2['emoji']} {r2['risk_score']}/100")
    print(f"  Risk Level   : {r2['risk_level']}")
    print(f"  Action       : {r2['action']}")
    print(f"  Patterns     : {', '.join(r2['patterns'])}")
    print(f"  SAR Required : {'✅ YES — File with FinCEN' if r2['sar_required'] else 'No'}")

    # Generate SAR report
    if r2['sar_required']:
        sar = sar_gen.generate_sar_summary("ACC-2024-8821", sus_txn, r2)
        print(sar)

    # Portfolio scan
    scanner = PortfolioAMLScanner()
    scanner.scan(detector, data)

    print("\n✅ AML System ready for SME bank deployment")
    print("🇺🇸 Supporting FinCEN & Bank Secrecy Act compliance\n")


if __name__ == "__main__":
    main()
