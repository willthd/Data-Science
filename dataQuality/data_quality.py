import pandas as pd


def check_null(
    x: pd.DataFrame,
    cols=[
        'col1',
        'col2',
        'col3',
        'col4',
    ],
) -> pd.DataFrame:
    """ 
    Check null counts of columns.
    """
    result_df = pd.DataFrame()
    intersection_cols = list(x.columns.intersection(cols))
    nans = pd.DataFrame(x[intersection_cols].isnull().sum().sort_values(ascending=False), columns=['null_counts'])
    idx = nans['null_counts'] > 0
    
    if len(nans[idx]) == 0:
        print('Null counts warnings: There are no columns.')
        
    return nans[idx]


def check_cos_similarity(x: pd.DataFrame) -> pd.DataFrame:
    """
    Check cos similarity between corresponding columns in forecast and observation data not including null.
    The corresponding columns are fixed.
    """
    import warnings
    from sklearn.metrics.pairwise import cosine_similarity
    
    column_pairs = [
        ('col1', 'corresponding_col1'),
        ('col2', 'corresponding_col2'),
        ('col3', 'corresponding_col3'),
        ('col4', 'corresponding_col4'),
    ]
    
    df = pd.DataFrame(columns=['cos_similarity'])
    for fore_col, obs_col in column_pairs:
        if (fore_col in x.columns) and (obs_col in x.columns):
            temp = x[[fore_col, obs_col]]
            temp = temp.dropna()
            cos_similarity = cosine_similarity(temp[fore_col].values.reshape((1, -1)), temp[obs_col].values.reshape((1, -1))).item(0, 0)
            df.loc[fore_col, 'cos_similarity'] = cos_similarity
    
    if len(df) == 0:
        print('Cos similarity warnings: There are no foreacst-observation column pairs.')
        
    return df.sort_values(ascending=False, by=['cos_similarity'])


def check_outlier_iforest(
    x: pd.DataFrame,
    cols=[
        'col1',
        'col2',
        'col3',
        'col4',
    ]
) -> pd.DataFrame:
    """
    Check outliers with isolation forest not including null.
    """
    from sklearn.ensemble import IsolationForest
    
    intersection_cols = list(x.columns.intersection(cols))
    df = pd.DataFrame(columns=['outlier_percent'])
    for col in intersection_cols:
        temp = x[[col]]
        temp = temp.dropna()
        clf = IsolationForest()
        preds = clf.fit_predict(temp[col].values.reshape(-1, 1))
        percent = len(preds[preds < 0])/len(preds)
        df.loc[col, 'outlier_percent'] = percent
        
    if len(df) == 0:
        print('Outlier warnings: There are no columns.')

    return df.sort_values(ascending=False, by=['outlier_percent'])


def check_outlier_IQR(
    x: pd.DataFrame,
    cols=['col1',
          'col2',
          'col3',
          'col4',
         ],
) -> pd.DataFrame:
    """
    Check outliers with inter quntaile range not including null.
    """
    import numpy as np
    
    intersection_cols = list(x.columns.intersection(cols))
    df = pd.DataFrame(columns=['outlier_percent'])
    for col in intersection_cols:
        temp = x[[col]]
        temp = temp.dropna()
        Q1 = np.percentile(temp[col], 25)
        Q3 = np.percentile(temp[col], 75)
        IQR = Q3 - Q1
        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR
        percent = len(temp[(temp[col] < low_lim) | (temp[col] > up_lim)]) / len(temp[col])
        df.loc[col, 'outlier_percent'] = percent
        
    if len(df) == 0:
        print('Outlier warnings: There are no columns.')
    
    return df.sort_values(ascending=False, by=['outlier_percent'])


def check_data(x: pd.DataFrame):
    """
    Check null counts of columns, outliers and cosine similarity.
    """
    null_df = check_null(x)
    cos_df = check_cos_similarity(x)
    outlier_df = check_outlier_IQR(x)
        
    if len(null_df) + len(cos_df) + len(outlier_df) != 0:
        merged_df = null_df.merge(outlier_df, how='outer', left_index=True, right_index=True)
        merged_df = merged_df.merge(cos_df, how='outer', left_index=True, right_index=True)

        return merged_df.sort_values(ascending=False, by=['cos_similarity'])