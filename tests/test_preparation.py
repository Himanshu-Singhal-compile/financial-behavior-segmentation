"""Check aggregation, missing standard deviations, and fraud-label mapping."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).resolve().parents[1] / 'scripts/prepare_customer_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

class PreparationTests(unittest.TestCase):
    def test_synthetic_customer_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({
                'id': [1, 2, 3], 'client_id': [10, 10, 20],
                'date': ['2024-01-01 23:00', '2024-01-06 12:00', '2024-01-02 08:00'],
                'amount': ['$10', '$20', '$30'],
                'use_chip': ['Online Transaction', 'Swipe Transaction', 'Chip Transaction'],
                'mcc': [100, 200, 100], 'errors': [None, 'error', None],
            }).to_csv(path / 'transactions_data.csv', index=False)
            pd.DataFrame({'id': [10, 20], 'yearly_income': ['$1,000', '$0'],
                'credit_score': [700, 600], 'total_debt': ['$100', '$50']
            }).to_csv(path / 'users_data.csv', index=False)
            (path / 'train_fraud_labels.json').write_text(json.dumps({'target': {'1': 'Yes', '2': 'No'}}))
            features = prepare.prepare(path).set_index('client_id')
            self.assertEqual(features.loc[10, 'total_spend'], 30)
            self.assertEqual(features.loc[10, 'merchant_diversity'], 2)
            self.assertEqual(features.loc[10, 'night_txn_ratio'], 0.5)
            self.assertEqual(features.loc[20, 'std_transaction_amount'], 0)
            self.assertTrue(np.isfinite(features.to_numpy()).all())
            self.assertNotIn('is_fraud', features.columns)
            labels = pd.read_csv(path / 'fraud_check.csv')
            self.assertEqual(labels.iloc[0]['fraud_txn_count'], 1)
            self.assertEqual(len(labels), 1)  # Unlabeled customer isn't treated as negative.

if __name__ == '__main__':
    unittest.main()
