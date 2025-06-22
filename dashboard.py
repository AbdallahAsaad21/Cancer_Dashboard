import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

# Load merged files
data_path = 'merged_output'
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
file_display_names = [f.replace('_merged.csv', '').capitalize() for f in files]
file_mapping = dict(zip(file_display_names, files))

# Sidebar filters
selected_country_display = st.sidebar.selectbox("🌍 Select Country", file_display_names)
selected_country_file = file_mapping[selected_country_display]
df = pd.read_csv(os.path.join(data_path, selected_country_file))
country_name = selected_country_display

gender = st.sidebar.radio("🚻 Select Gender", sorted(df['Sex'].dropna().unique()))
cancer_types = sorted(df['Cancer label'].dropna().unique())
cancer = st.sidebar.selectbox("🧬 Select Cancer Type", cancer_types)

years = sorted(df['Year'].dropna().unique())
year_range = st.sidebar.slider("📅 Select Year Range", min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))))

env_cols = ['CO2_Emissions', 'PM2.5', 'Temperature', 'Humidity', 'Precipitation']
available_env = [col for col in env_cols if col in df.columns]
selected_env = st.sidebar.selectbox("🌡️ Select Geospatial Factor", available_env)

metric_options = {
    'Total incidence': 'Total incidence',
    'Age-Standardized Rate (ASR)': 'ASR (World)',
    'Crude rate': 'Crude rate'
}
selected_metric = st.sidebar.selectbox("📈 Select Cancer Metric to Visualize", options=list(metric_options.keys()))
metric_column = metric_options[selected_metric]

# Filter data
filtered_df = df[
    (df['Sex'] == gender) &
    (df['Cancer label'] == cancer) &
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1])
].copy()

st.title("Understanding the Relationship Between Geospatial Data and Cancer in Selected American Countries")
st.markdown("<i>Note: ASR and Crude Rate reflect mortality rates per 100,000 individuals.</i>", unsafe_allow_html=True)

# Summary boxes in one slim row
avg_value = filtered_df[metric_column].mean()

box_style = "background-color: #D0E6F7; padding: 10px; border-radius: 10px; width: 100%; text-align: center; margin-bottom: 10px;"
value_style = "font-size: 26px; font-weight: 500;"

colA, colB, colC, colD, colE = st.columns(5)
with colA:
    st.markdown(f"<div style='{box_style}'><strong>Country</strong><br><span style='{value_style}'>{country_name}</span></div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<div style='{box_style}'><strong>Cancer Type</strong><br><span style='{value_style}'>{cancer}</span></div>", unsafe_allow_html=True)
with colC:
    st.markdown(f"<div style='{box_style}'><strong>{selected_metric} (Average)</strong><br><span style='{value_style}'>{avg_value:.2f}</span></div>", unsafe_allow_html=True)
with colD:
    st.markdown(f"<div style='{box_style}'><strong>Year Range</strong><br><span style='{value_style}'>{year_range[0]} to {year_range[1]}</span></div>", unsafe_allow_html=True)
with colE:
    st.markdown(f"<div style='{box_style}'><strong>Gender</strong><br><span style='{value_style}'>{gender}</span></div>", unsafe_allow_html=True)

# Prepare data for charts
filtered_df.sort_values('Year', inplace=True)
filtered_df['YoY Change'] = filtered_df[metric_column].diff()

cor_df = df[
    (df['Sex'] == gender) &
    (df['Country'] == country_name) &
    (df['Cancer label'] == cancer) &
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1])
]

# Second row with 3 plots
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<h4 style='text-align: center'>{selected_metric} Trends in {country_name}</h4>", unsafe_allow_html=True)
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.bar(filtered_df['Year'], filtered_df[metric_column], color='#4A90E2')
    ax1.set_ylabel(selected_metric, color='#4A90E2')
    ax2 = ax1.twinx()
    ax2.plot(filtered_df['Year'], filtered_df['YoY Change'], color='#003366', marker='o')
    ax2.set_ylabel('YoY Change', color='#003366')
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown(f"<h4 style='text-align: center'>{selected_env} vs {selected_metric} Over Time</h4>", unsafe_allow_html=True)
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.plot(cor_df['Year'], cor_df[selected_env], color='#7BAFD4', marker='o')
    ax1.set_ylabel(selected_env, color='#7BAFD4')
    ax2 = ax1.twinx()
    ax2.plot(cor_df['Year'], cor_df[metric_column], color='black', marker='o')
    ax2.set_ylabel(metric_column, color='black')
    fig.tight_layout()
    st.pyplot(fig)

with col3:
    st.markdown(f"<h4 style='text-align: center'>Scatter: {selected_env} vs {selected_metric}</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(cor_df[selected_env], cor_df[metric_column], label="Data")
    m, b = np.polyfit(cor_df[selected_env], cor_df[metric_column], 1)
    r_squared = np.corrcoef(cor_df[selected_env], cor_df[metric_column])[0, 1] ** 2
    ax.plot(cor_df[selected_env], m * cor_df[selected_env] + b, linestyle='--', color='navy', label="Best Fit")
    ax.set_xlabel(selected_env)
    ax.set_ylabel(metric_column)
    ax.set_title(f"$R^2$ = {r_squared:.2f}")
    ax.legend()
    st.pyplot(fig)

# Third row: Distribution + Correlation
col4, col5 = st.columns([1.2, 1])

with col4:
    latest_year = df['Year'].max()
    cancer_latest_df = df[
        (df['Sex'] == gender) &
        (df['Country'] == country_name) &
        (df['Year'] == latest_year)
    ]

    if not cancer_latest_df.empty:
        st.markdown(f"<h4 style='text-align: center'>Distribution of {selected_metric} by Cancer Type in {country_name} ({latest_year})</h4>", unsafe_allow_html=True)
        treemap = px.treemap(
            cancer_latest_df,
            path=['Cancer label'],
            values=metric_column,
            color=metric_column,
            color_continuous_scale='Blues'
        )
        treemap.update_traces(textfont_size=24)
        treemap.update_layout(
            margin=dict(t=40, l=0, r=0, b=0),
            height=600
        )
        st.plotly_chart(treemap, use_container_width=True)

with col5:
    cancer_pivot = df[
        (df['Sex'] == gender) &
        (df['Country'] == country_name) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ].pivot_table(index='Year', columns='Cancer label', values=metric_column)

    env_df = df[
        (df['Country'] == country_name) &
        (df['Year'] >= year_range[0]) &
        (df['Year'] <= year_range[1])
    ][available_env + ['Year']].drop_duplicates().set_index('Year')

    combined = pd.concat([cancer_pivot, env_df], axis=1).dropna()
    corr = combined.corr()
    corr_subset = corr.loc[cancer_pivot.columns, available_env]  # cancer on y-axis, geospatial on x-axis

    st.markdown("<h4 style='text-align: center'>Correlation Matrix</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_subset, cmap='Blues', annot=True, fmt=".2f", ax=ax, annot_kws={"size": 10})
    st.pyplot(fig)