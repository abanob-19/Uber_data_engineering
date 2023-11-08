import pandas as pd
import os
import functions as functions
from sqlalchemy import create_engine

df = pd.read_csv('data/green_tripdata_2018-09.csv')
print('read csv')
cleaned_dataset_path = 'green_trip_data_{2018}-{09}clean.csv'


if os.path.exists(cleaned_dataset_path):
    print("Cleaned dataset found. No need to call cleaning functions.")
else:
    print("Cleaned dataset not found. Calling cleaning functions...")
    functions.rename_columns(df)
    functions.remove_duplicates(df)
    df=functions.column_negatives(df)
    print(df['improvement_surcharge'])
    functions.replace_values_with(df, 'improvement_surcharge', lambda x: (x != 0.3) & (x != 0.0),0)
    functions.replace_values_with(df, 'extra', lambda x: (x != 0.5) & (x != 0.0) & (x != 1.0),0)
    functions.replace_values_with(df, 'mta_tax', lambda x: (x != 0.5) & (x != 0.0)  ,0)
    df = functions.remove_data_with_different_month_and_year(df, 9 , 2018)
    functions.handle_missing_columns(df)
    functions.column_to_data_type(df, 'passenger_count', int)
    functions.passenger_count_outliers(df)
    functions.handle_multi_variant(df,'trip_distance','fare_amount')
    functions.create_discretize(df)
    functions.column_to_data_type(df, 'week_number', int)
    categorical_columns = list(df.select_dtypes(include=['object']).drop(columns=['pu_location','do_location']).columns) 
    df=functions.one_hot_encoding(df,categorical_columns)
    columns_to_normalize = ['tip_amount']
    df = functions.box_cox_normalization(df, columns_to_normalize)
    columns_to_scale = ['passenger_count', 'trip_distance', 'fare_amount', 'tolls_amount', 'total_amount','tip_amount']
    df = functions.min_max_scaling(df, columns_to_scale)
    df=functions.add_2_columns(df)
    functions.populate_coordinates(df)
    lookup=functions.label_encoding(df,df[['pu_location','do_location']])
    lookup_df=functions.save_lookup_table(lookup)
    functions.save_to_csv_parquet(df)
engine = create_engine('postgresql://root:root@pgdatabase:5432/green_taxi_9_2018')

if(engine.connect()):
	print('connected succesfully')
else:
	print('failed to connect')


df.to_sql(name = 'green_taxi_9_2018',con = engine,if_exists='append')
lookup_df.to_sql(name = 'lookup_green_taxi_9_2018',con = engine,if_exists='append')
print('created tables')
