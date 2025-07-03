import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from datetime import datetime

# Custom transformer for aggregate features
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', date_col='TransactionDate'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure date is datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        snapshot_date = df[self.date_col].max() + pd.Timedelta(days=1)
        agg = df.groupby(self.customer_id_col).agg(
            Recency=(self.date_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(self.date_col, 'count'),
            Monetary=(self.amount_col, 'sum'),
            AvgTransactionAmount=(self.amount_col, 'mean'),
            StdTransactionAmount=(self.amount_col, 'std'),
        ).reset_index()
        agg['AvgTransactionAmount'] = agg['AvgTransactionAmount'].fillna(0)
        agg['StdTransactionAmount'] = agg['StdTransactionAmount'].fillna(0)
        return agg

# Custom transformer for extracting time features
class TimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='TransactionDate'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df['TransactionHour'] = df[self.date_col].dt.hour
        df['TransactionDay'] = df[self.date_col].dt.day
        df['TransactionMonth'] = df[self.date_col].dt.month
        df['TransactionYear'] = df[self.date_col].dt.year
        return df[[c for c in df.columns if c not in [self.date_col]]]

# Custom transformer for categorical encoding
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.categorical_cols:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        return df

# Main processing function

def process_data(
    raw_data_path='data.csv',
    customer_id_col='CustomerId',
    amount_col='Amount',
    date_col='TransactionDate',
    categorical_cols=None,
    n_clusters=3,
    random_state=42
):
    # Load data
    df = pd.read_csv(raw_data_path)
    if categorical_cols is None:
        # Guess categorical columns (object type, except id/date)
        categorical_cols = [c for c in df.select_dtypes(include='object').columns if c not in [customer_id_col, date_col]]

    # Aggregate features
    agg_pipe = Pipeline([
        ('agg', AggregateFeatures(customer_id_col, amount_col, date_col)),
    ])
    agg_features = agg_pipe.fit_transform(df)

    # Extract time features
    time_pipe = Pipeline([
        ('time', TimeFeatures(date_col)),
    ])
    time_features = time_pipe.fit_transform(df[[customer_id_col, date_col]].drop_duplicates())
    time_features = time_features.groupby(customer_id_col).agg(
        MostCommonHour=('TransactionHour', lambda x: x.mode()[0] if not x.mode().empty else 0),
        MostCommonDay=('TransactionDay', lambda x: x.mode()[0] if not x.mode().empty else 0),
        MostCommonMonth=('TransactionMonth', lambda x: x.mode()[0] if not x.mode().empty else 0),
        MostCommonYear=('TransactionYear', lambda x: x.mode()[0] if not x.mode().empty else 0),
    ).reset_index()

    # Merge features
    features = agg_features.merge(time_features, on=customer_id_col, how='left')

    # Encode categorical variables
    if categorical_cols:
        cat_pipe = Pipeline([
            ('cat', CategoricalEncoder(categorical_cols)),
        ])
        cat_features = df[[customer_id_col] + categorical_cols].drop_duplicates(subset=customer_id_col)
        cat_features = cat_pipe.fit_transform(cat_features)
        features = features.merge(cat_features, on=customer_id_col, how='left')

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[num_cols] = imputer.fit_transform(features[num_cols])

    # Normalize/Standardize
    scaler = StandardScaler()
    features[num_cols] = scaler.fit_transform(features[num_cols])

    # RFM for clustering
    rfm = features[[customer_id_col, 'Recency', 'Frequency', 'Monetary']].copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_cluster = kmeans.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])
    rfm['RiskCluster'] = rfm_cluster
    # Assume cluster with lowest Frequency/Monetary is high risk
    cluster_stats = rfm.groupby('RiskCluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats[['Frequency', 'Monetary']].sum(axis=1).idxmin()
    rfm['is_high_risk'] = (rfm['RiskCluster'] == high_risk_cluster).astype(int)

    # Merge back
    features = features.merge(rfm[[customer_id_col, 'is_high_risk']], on=customer_id_col, how='left')

    return features

if __name__ == '__main__':
    df = process_data()
    df.to_csv('data/processed/processed_data.csv', index=False)
