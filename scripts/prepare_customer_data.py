"""Aggregate the source transaction dataset into customer-level features.

Usage: python scripts/prepare_customer_data.py --data-dir data
Only the documented fields are loaded; card numbers and customer names are ignored.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def prepare(data_dir):
    data_dir = Path(data_dir)
    txn = pd.read_csv(data_dir / 'transactions_data.csv', usecols=[
        'id', 'client_id', 'date', 'amount', 'use_chip', 'mcc', 'errors'])
    users = pd.read_csv(data_dir / 'users_data.csv', usecols=[
        'id', 'yearly_income', 'credit_score', 'total_debt']).rename(columns={'id': 'client_id'})
    for col in ['yearly_income', 'total_debt']:
        users[col] = pd.to_numeric(users[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='raise')
    txn['amount'] = pd.to_numeric(txn['amount'].astype(str).str.replace(r'[$,]', '', regex=True), errors='raise')
    txn['date'] = pd.to_datetime(txn['date'], errors='raise')
    txn = txn.merge(users, on='client_id', how='left', validate='many_to_one')
    txn['is_online'] = (txn['use_chip'] == 'Online Transaction').astype(int)
    txn['is_swipe'] = (txn['use_chip'] == 'Swipe Transaction').astype(int)
    txn['is_night'] = txn['date'].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    txn['is_weekend'] = txn['date'].dt.dayofweek.isin([5, 6]).astype(int)
    txn['has_error'] = txn['errors'].notna().astype(int)
    features = txn.groupby('client_id').agg(
        avg_transaction_amount=('amount', 'mean'), total_spend=('amount', 'sum'),
        transaction_count=('amount', 'count'), std_transaction_amount=('amount', 'std'),
        max_transaction_amount=('amount', 'max'), online_txn_ratio=('is_online', 'mean'),
        swipe_txn_ratio=('is_swipe', 'mean'), night_txn_ratio=('is_night', 'mean'),
        weekend_txn_ratio=('is_weekend', 'mean'), merchant_diversity=('mcc', 'nunique'),
        error_rate=('has_error', 'mean'), yearly_income=('yearly_income', 'first'),
        credit_score=('credit_score', 'first'), total_debt=('total_debt', 'first')).reset_index()
    features['std_transaction_amount'] = features['std_transaction_amount'].fillna(0)
    income = features['yearly_income'].replace(0, np.nan)
    features['spend_to_income_ratio'] = features['total_spend'] / income
    features = features.replace([np.inf, -np.inf], np.nan)
    for col in features.columns.drop('client_id'):
        features[col] = features[col].fillna(features[col].median() if features[col].notna().any() else 0)
    features.to_csv(data_dir / 'customer_features.csv', index=False)
    txn[['id', 'client_id']].to_csv(data_dir / 'id_client_id.csv', index=False)
    label_path = data_dir / 'train_fraud_labels.json'
    if label_path.exists():
        target = json.loads(label_path.read_text(encoding='utf-8'))['target']
        labels = pd.DataFrame(list(target.items()), columns=['id', 'is_fraud'])
        labels['id'] = labels['id'].astype(int)
        labels['is_fraud'] = labels['is_fraud'].map({'Yes': 1, 'No': 0})
        if labels['is_fraud'].isna().any():
            raise ValueError('Unknown fraud label; expected Yes or No')
        # Unknown labels remain missing, rather than silently becoming non-fraud.
        check = txn[['id', 'client_id']].merge(labels, on='id', how='inner', validate='one_to_one')
        check.groupby('client_id')['is_fraud'].sum().rename('fraud_txn_count').reset_index().to_csv(data_dir / 'fraud_check.csv', index=False)
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    print(f'Prepared {len(prepare(args.data_dir))} customer records.')
