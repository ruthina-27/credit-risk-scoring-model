import pandas as pd
import numpy as np
import pytest
from src.data_processing import AggregateFeatures, CategoricalEncoder

def test_aggregate_features():
    data = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 2],
        'Amount': [100, 200, 50, 60, 70],
        'TransactionDate': [
            '2024-01-01', '2024-01-10',
            '2024-01-05', '2024-01-15', '2024-01-20'
        ]
    })
    agg = AggregateFeatures('CustomerId', 'Amount', 'TransactionDate')
    result = agg.fit_transform(data)
    assert 'Recency' in result.columns
    assert 'Frequency' in result.columns
    assert 'Monetary' in result.columns
    assert result.loc[result['CustomerId'] == 1, 'Frequency'].values[0] == 2
    assert result.loc[result['CustomerId'] == 2, 'Monetary'].values[0] == 180

def test_categorical_encoder():
    data = pd.DataFrame({
        'CustomerId': [1, 2, 3],
        'Category': ['A', 'B', 'A']
    })
    encoder = CategoricalEncoder(['Category'])
    encoder.fit(data)
    transformed = encoder.transform(data)
    assert 'Category' in transformed.columns
    assert set(transformed['Category']) == {0, 1}
