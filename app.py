import pandas as pd
import os
from healthcare_utils import (
    load_csv_files, clean_cancer_data, reshape_geo_data, merge_cancer_with_geo
)

# Set data directory
data_dir = '.'
output_dir = 'merged_output'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load & clean all cancer country files
cancer_files = load_csv_files(data_dir, keyword='cancer')
cancer_dfs = {country: clean_cancer_data(df) for country, df in cancer_files.items()}

# Step 2: Load & reshape all geospatial data
geo_files = load_csv_files(data_dir)  # Load all CSVs
print("🔍 Geo files loaded:", list(geo_files.keys()))

geo_reshaped = {}

for name, df in geo_files.items():
    print(f"📂 Checking geospatial file: {name}")
    name_lower = name.lower()
    if 'co2' in name_lower:
        geo_reshaped['CO2_Emissions'] = reshape_geo_data(df, value_name='CO2_Emissions')
    elif 'pm2.5' in name_lower:
        geo_reshaped['PM2.5'] = reshape_geo_data(df, value_name='PM2.5')
    elif 'temp' in name_lower:
        geo_reshaped['Temperature'] = reshape_geo_data(df, value_name='Temperature')
    elif 'humid' in name_lower:
        geo_reshaped['Humidity'] = reshape_geo_data(df, value_name='Humidity')
    elif 'precip' in name_lower:
        geo_reshaped['Precipitation'] = reshape_geo_data(df, value_name='Precipitation')

# Step 3: Merge for each country and save separately
for country_key, cancer_df in cancer_dfs.items():
    country_name = cancer_df['Country'].iloc[0]
    df_merged = cancer_df.copy()

    for geo_name, geo_df in geo_reshaped.items():
        print(f"🔗 Merging {geo_name} into {country_name}")
        geo_country_df = geo_df[geo_df['Country'] == country_name]
        df_merged = pd.merge(df_merged, geo_country_df, on=['Country', 'Year'], how='left')

    # Drop duplicate precipitation columns if they exist
    if 'Precipitation_x' in df_merged.columns and 'Precipitation_y' in df_merged.columns:
        df_merged['Precipitation'] = df_merged['Precipitation_x'].combine_first(df_merged['Precipitation_y'])
        df_merged.drop(columns=['Precipitation_x', 'Precipitation_y'], inplace=True)

    output_path = os.path.join(output_dir, f"{country_name.lower().replace(' ', '_')}_merged.csv")
    df_merged.to_csv(output_path, index=False)

    print(f"✅ Saved merged file for {country_name} → {output_path}")