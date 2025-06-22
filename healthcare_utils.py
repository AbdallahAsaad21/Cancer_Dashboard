import pandas as pd
import os

def load_csv_files(folder_path, keyword=None):
    data = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and (keyword is None or keyword.lower() in file.lower()):
            df = pd.read_csv(os.path.join(folder_path, file))
            data[file.replace('.csv', '')] = df
    return data

# Utility functions
def clean_cancer_data(df):
    df = df.drop(columns=['Cancer id', 'Population id', 'Type', 'Cumulative risk'], errors='ignore')
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Country label': 'Country', 'Total': 'Total incidence'})
    df['Sex'] = df['Sex'].replace({1: 'Male', 2: 'Female'})
    return df

def reshape_geo_data(df, value_name='Value'):
    df.columns = df.columns.str.strip()
    id_vars = ['Country']
    value_vars = [col for col in df.columns if col not in id_vars and col.strip().isdigit()]
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars,
                      var_name='Year', value_name=value_name)
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long

def merge_cancer_with_geo(cancer_df, geo_df, on=['Country', 'Year']):
    return pd.merge(cancer_df, geo_df, how='left', on=on)