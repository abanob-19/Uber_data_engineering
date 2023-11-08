import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import os
import requests
import time
def rename_columns(dataframe):
    dataframe.columns = dataframe.columns.str.lower()
    dataframe.columns = [col.replace(' ', '_') for col in dataframe.columns]
def remove_duplicates(dataframe):
    print("Number of duplicate rows = ", dataframe.duplicated().sum())
    dataframe.drop_duplicates(inplace=True)
    print("Number of duplicate rows after removing duplicates = ", dataframe.duplicated().sum())
def column_to_data_type(dataframe, column, data_type):
    print(f"original {column} data type: ", dataframe[column].dtype)
    dataframe[column] = dataframe[column].astype(data_type)
    print(f"updated {column} data type: ", dataframe[column].dtype)
def replace_with_mean(df,column):
   
    negative_count = (df[column] < 0).sum()
    print(f"Number of negative {column} = {negative_count}")
        
    if(negative_count>0):
     # Replace negative fares with the median fare
        index_of_first_negative = df[df[column] < 0].index[0]
        print("1st row having the issue ",df[df[column] < 0].head(1)[column] )
        mean_fare = df[df[column] >= 0][column].mean()
        print(column," mean ",mean_fare)
        df.loc[df[column] < 0, column] = mean_fare
        print("1st row having the issue after updating ", index_of_first_negative, df.loc[index_of_first_negative, column] )
        negative_count = (df[column] < 0).sum()
        print(f"updated Number of negative {column} = {negative_count}")

def column_negatives(df):
    numerical_columns = ['passenger_count', 'fare_amount', 'trip_distance', 'tip_amount', 'tolls_amount', 'total_amount']

    for column in numerical_columns:
        replace_with_mean(df,column)
    return df    
def replace_values_with(df, column, condition, new):
    before = (condition(df[column]))
    print(f"Sum of values that do not meet the condition before replacement: {before.sum()}")
    print(f"{df[before][column]}")
    df.loc[condition(df[column]), column] = new
    after = (condition(df[column]))
    print(f"Sum of values that do not meet the condition after replacement: {after.sum()}")
def remove_data_with_different_month_and_year(df, target_month, target_year):
    date_column = 'lpep_pickup_datetime'
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])

    filtered_df = df[
        (df[date_column].dt.year == target_year) &
        (df[date_column].dt.month == target_month)
    ]

    # Print the rows to be dropped
    rows_to_remove = df[~df.index.isin(filtered_df.index)]
    print("Rows to be dropped:")
    print(rows_to_remove)
    return filtered_df
def handle_missing(df, column, replacement):
    
        
    print(f'Column {column} replacement value ', replacement)

    df[column].replace('Unknown', np.nan, inplace=True)
    df[column].replace('Uknown', np.nan, inplace=True)
    
    if column == 'ehail_fee':
        df.loc[df['trip_type'] == 'street-hail', column] = replacement
        df.loc[df['trip_type'] != 'street-hail', column] = 0
    
    # Fill missing values in the column with the specified replacement
    if (column == 'vendor'):
        df.dropna(axis='index', subset=[column], inplace=True)
    elif column == 'trip_distance':
        df[column].replace(0, replacement, inplace=True)
    else:    
        df[column].fillna(value=replacement, inplace=True)
def handle_missing_columns(df):
    handle_missing(df,'store_and_fwd_flag','N')
    handle_missing(df,'passenger_count',df['passenger_count'].mean())
    handle_missing(df,'trip_distance',df['trip_distance'].mean())
    handle_missing(df,'extra',0)
    handle_missing(df,'trip_type','Street-hail')
    handle_missing(df,'ehail_fee',0.5)
    handle_missing(df,'vendor','drop')
    handle_missing(df,'rate_type','Standard rate')
    handle_missing(df,'payment_type',df['payment_type'].mode().iloc[0])
    df.drop('congestion_surcharge', axis=1, inplace=True)        
def clip_outliers_with_q1_q3(df,columns):
    for col in columns:
#         print(df[col].isna().sum())  # Check how many missing values are in the column
#         print(col)
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr=q3-q1
        print(q1,q3)
        df[col] = np.where(df[col] <q1-1.5*iqr, q1-1.5*iqr,df[col])
        df[col] = np.where(df[col] >q3+1.5*iqr, q3+1.5*iqr,df[col])
def clip_outliers_with_log(df,columns):
    for col in columns:
        df[col] = np.log1p(df[col])
def clip_with_z(df,columns):
    for col in columns:
        median = df[col].median()
        cutoff_pos = df[col].mean() + df[col].std() * 3
        cutoff_neg = df[col].mean() - df[col].std() * 3
        condition_series = (( df[col]> cutoff_pos) | (df[col] < cutoff_neg))
        df[col] = np.where(condition_series, median,df[col])                        
def passenger_count_outliers(df):
    df.drop(df[df['passenger_count'] > 10].index, inplace=True)
def handle_multi_variant(df,col1,col2, zscore_threshold=3):
    z_scores_col1 = np.abs((df[col1] - df[col1].mean()) / df[col1].std())
    z_scores_col2 = np.abs((df[col2] - df[col2].mean()) / df[col2].std())
    outliers = (z_scores_col1 > zscore_threshold) | (z_scores_col2 > zscore_threshold)
    df.drop(df[outliers].index, inplace=True)

def create_discretize(df):
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['week_number'] = df['lpep_pickup_datetime'].dt.strftime('%U')
    df['date_range'] = df['lpep_pickup_datetime'].dt.to_period('W').dt.to_timestamp()
    df['date_range'] = pd.to_datetime(df['date_range'])
    print(df[['lpep_pickup_datetime', 'week_number', 'date_range']].head())
    print(df[['lpep_pickup_datetime', 'week_number', 'date_range']].tail())        
    
def label_encoding(df, categorical_columns):
    lookup_table = {}
    for column in categorical_columns:
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])
        lookup_table[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    return lookup_table
def one_hot_encoding(df, categorical_columns):
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
   # df_encoded = df_encoded.drop(columns=categorical_columns)
    return df_encoded
def calculate_top_categories(df, variable, how_many):
    return [
        x for x in df[variable].value_counts().sort_values(
            ascending=False).head(how_many).index
    ]
# manually encode the most frequent values in a catgeorical feature.
def one_hot_encode(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable + '_' + label] = np.where(
            df[variable] == label, 1, 0)
	
def box_cox_normalization(dataframe, columns):
    for col in columns:
        # Get the index of all positive values (Box-Cox only takes positive values)
        index = dataframe[col] > 0
        # Select only from the column we're currently processing
        positive_vals = dataframe.loc[index, col]
        # Apply Box-Cox transformation
        normalized_vals, _ = stats.boxcox(positive_vals)
        # Assign the transformed data back to the original dataframe
        dataframe.loc[index, col] = normalized_vals
    return dataframe	
def min_max_scaling(dataframe, columns):
    scaler = MinMaxScaler()
    for col in columns:
        # Ensure the column is in a 2D shape for the scaler
        col_data = dataframe[col].values.reshape(-1, 1)
        scaled = scaler.fit_transform(col_data)
        dataframe[col] = scaled
    return dataframe
def add_2_columns(df):
    df['is_weekend'] = df['lpep_pickup_datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    return df
def get_coordinates(location):
    time.sleep(0.25)
    url = f"https://geocode.maps.co/search?q={location}"
    response = requests.get(url)
    data = response.json()
    if data:
        return data[0]['lat'], data[0]['lon']
    else:
        return np.nan, np.nan
def populate_coordinates(df):
    if os.path.exists('unique_location_coordinates.csv'):
        coordinates_df = pd.read_csv('unique_location_coordinates.csv',index_col='location')
        print (coordinates_df)
        df['pu_lat'] = df['pu_location'].map(coordinates_df['latitude'])
        df['pu_long'] = df['pu_location'].map(coordinates_df['longitude'])
        df['do_lat'] = df['do_location'].map(coordinates_df['latitude'])
        df['do_long'] = df['do_location'].map(coordinates_df['longitude'])
    else:
        unique_locations = pd.concat([df['pu_location'], df['do_location']]).unique()
        coordinates_df = pd.DataFrame(unique_locations, columns=['location'])
        coordinates_df[['latitude', 'longitude']] = coordinates_df['location'].apply(get_coordinates).apply(pd.Series)
        coordinates_df.set_index('location', inplace=True)
        coordinates_df.to_csv('unique_location_coordinates.csv')
        df['pu_lat'] = df['pu_location'].map(coordinates_df['latitude'])
        df['pu_long'] = df['pu_location'].map(coordinates_df['longitude'])
        df['do_lat'] = df['do_location'].map(coordinates_df['latitude'])
        df['do_long'] = df['do_location'].map(coordinates_df['longitude'])
def save_lookup_table(lookup):
    lookup_df = pd.DataFrame([{"column_name": column_name, "original_value": original_value, "imputed_value": imputed_value}
    for column_name, mapping in lookup.items()
    for original_value, imputed_value in mapping.items()])
    lookup_df.to_csv('lookup_table_green_taxis.csv', index=False)
    return lookup_df
def save_to_csv_parquet(df):
    df.to_parquet('green_trip_data_{2018}-{09}clean.parquet')
    df.to_csv('green_trip_data_{2018}-{09}clean.csv')        