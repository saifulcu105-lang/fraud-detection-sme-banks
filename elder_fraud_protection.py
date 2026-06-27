"""
Elder Financial Fraud Protection System
for SME Banks & Community Financial Institutions

Author  : Saiful (saifulcu105-lang)
Target  : Community banks serving senior Americans
Purpose : Detect and prevent financial exploitation of elderly customers
NIW     : Protects vulnerable US population — strong national interest

References:
- Elder Justice Act (42 U.S.C. § 1397j)
- CFPB Advisory on Elder Financial Exploitation
- FinCEN Advisory FIN-2022-A002
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def generate_elder_data(n=5000):
    """Generate synthetic senior customer transaction data."""
    np.random.seed(99)
    n_safe = int(n * 0.95)
    n_risk = int(n * 0.05)

    safe = pd.DataFrame({
        'customer_age':          np.random.randint(65, 90, n_safe),
        'withdrawal_amount':     np.random.lognormal(5, 1, n_safe),
        'num_withdrawals_week':  np.random.poisson(2, n_safe),
        'new_payee_added':       np.random.binomial(1, 0.05, n_safe),
        'wire_transfer':         np.random.binomial(1, 0.03, n_safe),
        'atm_frequency':         np.random.poisson(1, n_safe),
        'unusual_hour':          np.random.binomial(1, 0.05, n_safe),
        'gift_card_purchase':    np.random.binomial(1, 0.02, n_safe),
        'phone_initiated':       np.random.binomial(1, 0.1, n_safe),
        'account_change':        np.random.binomial(1, 0.02, n_safe),
        'is_exploitation':       0
    })

    risk = pd.DataFrame({
        'customer_age':          np.random.randint(75, 95, n_risk),
        'withdrawal_amount':     np.random.lognormal(8, 1.5, n_risk),
        'num_withdrawals_week':  np.random.randint(8, 20, n_risk),
        'new_payee_added':       np.random.binomial(1, 0.8, n_risk),
        'wire_transfer':         np.random.binomial(1, 0.7, n_risk),
        'atm_frequency':         np.random.randint(5, 15, n_risk),
        'unusual_hour':          np.random.binomial(1, 0.6, n_risk),
        'gift_card_purchase':    np.random.binomial(1, 0.7, n_risk),
        'phone_initiated':       np.random.binomial(1, 0.8, n_risk),
        'account_change':        np.random.binomial(1, 0.5, n_risk),
        'is_exploitation':       1
    })

    return pd.concat([safe, risk], ignore_index=True).sample(frac=1, random_state=42)


class ElderFraudProtector:
    """
    Detects financial exploitation of elderly bank customers.
    Covers: romance scams, grandparent scams, tech support fraud,
    lottery scams, and caregiver exploitation.
    """

    FEATURES = [
        'customer_age', 'withdrawal_amount', 'num_withdrawals_week',
        'new_payee_added', 'wire_transfer', 'atm_frequency',
        'unusual_hour', 'gift_card_purchase', 'phone_initiated', 'account_change'
    ]

    SCAM_TYPES = {
        'gift_card_purchase': 'Gift Card Scam — common in IRS/tech support fraud',
        'wire_transfer':      'Wire Transfer Scam — funds rarely recovered',
        'new_payee_added':    'New Payee — possible undue influence',
        'phone_initiated':    'Phone-Initiated — possible phone scam',
        'unusual_hour':       'Unusual Hours — possible coercion',
        'account_change':     'Account Change — possible takeover',
    }

    def __init__(self):
        self.scaler  = StandardScaler()
        self.model   = GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        )
        self.trained = False

    def train(self, data):
        print("\n🔄 Training Elder Fraud Protection Model...")
        X = self.scaler.fit_transform(data[self.FEATURES])
        y = data['is_exploitation']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(X_tr, y_tr)
        self.trained = True

        y_pred = self.model.predict(X_te)
        y_prob = self.model.predict_proba(X_te)[:, 1]

        print("\n" + "="*58)
        print("  ELDER FRAUD PROTECTION — PERFORMANCE")
        print("="*58)
        print(f"  AUC-ROC : {roc_auc_score(y_te, y_prob):.1%}")
        print(classification_report(y_te, y_pred,
              target_names=['Safe', 'Exploitation Risk']))
        return self

    def assess(self, txn: dict) -> dict:
        df   = pd.DataFrame([txn])[self.FEATURES]
        X    = self.scaler.transform(df)
        prob = self.model.predict_proba(X)[0][1]
        risk = int(prob * 100)

        warnings = [
            self.SCAM_TYPES[k]
            for k in self.SCAM_TYPES
            if txn.get(k, 0) == 1
        ]

        if risk < 25:
            level, action, emoji = "LOW",    "✅ NO ACTION",           "🟢"
        elif risk < 55:
            level, action, emoji = "MEDIUM", "⚠️  CONTACT CUSTOMER",   "🟡"
        else:
            level, action, emoji = "HIGH",   "🚨 HOLD & CONTACT FAMILY","🔴"

        return {
            'risk_score':   risk,
            'risk_level':   level,
            'action':       action,
            'emoji':        emoji,
            'exploit_prob': f"{prob:.1%}",
            'warnings':     warnings if warnings else ['No specific warnings'],
        }


def main():
    print("="*58)
    print("  ELDER FINANCIAL FRAUD PROTECTION SYSTEM")
    print("  Author  : Saiful (saifulcu105-lang)")
    print("  Law     : Elder Justice Act / CFPB Guidelines")
    print("="*58)

    data      = generate_elder_data(5000)
    protector = ElderFraudProtector()
    protector.train(data)

    # Safe senior
    print("\n── TEST 1: Normal Senior Transaction ──")
    safe = {
        'customer_age': 72, 'withdrawal_amount': 300,
        'num_withdrawals_week': 2, 'new_payee_added': 0,
        'wire_transfer': 0, 'atm_frequency': 1,
        'unusual_hour': 0, 'gift_card_purchase': 0,
        'phone_initiated': 0, 'account_change': 0
    }
    r1 = protector.assess(safe)
    print(f"  Risk Score : {r1['emoji']} {r1['risk_score']}/100")
    print(f"  Action     : {r1['action']}")

    # At-risk senior
    print("\n── TEST 2: Exploitation Risk Detected ──")
    risk_txn = {
        'customer_age': 84, 'withdrawal_amount': 15000,
        'num_withdrawals_week': 12, 'new_payee_added': 1,
        'wire_transfer': 1, 'atm_frequency': 8,
        'unusual_hour': 1, 'gift_card_purchase': 1,
        'phone_initiated': 1, 'account_change': 1
    }
    r2 = protector.assess(risk_txn)
    print(f"  Risk Score : {r2['emoji']} {r2['risk_score']}/100")
    print(f"  Action     : {r2['action']}")
    print(f"  Warnings   :")
    for w in r2['warnings']:
        print(f"    ⚠️  {w}")

    print("\n✅ Elder Protection System ready for deployment")
    print("🇺🇸 Protecting America's senior citizens from financial fraud\n")


if __name__ == "__main__":
    main()
