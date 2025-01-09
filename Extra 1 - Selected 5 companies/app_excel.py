import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import folium
from streamlit_folium import folium_static
from mpl_toolkits.mplot3d import Axes3D
import squarify

# Set page config
st.set_page_config(page_title="Pharma Industry Analysis", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        height: auto;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
        height: auto;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stDownloadButton>button:hover {
        background-color: #218838;
    }
    </style>
    """,
    unsafe_allow_html=True
)

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.prepare_data()

    def prepare_data(self):
        # Create binary columns for Yes/No values
        binary_columns = ['Manufacturer', 'Brand', 'Distributor', 'F&B (Food & Beverage)',
                          'Probiotics', 'Fortification']
        for col in binary_columns:
            self.df[f'{col}_Binary'] = (self.df[col] == 'Yes').astype(int)

        # Split multiple categories into lists
        list_columns = ['Product Categories', 'Health Segments', 'Certifications',
                       'Target Audience', 'Innovation/Research Areas']
        for col in list_columns:
            if col in self.df.columns:
                self.df[f'{col}_List'] = self.df[col].str.split(',')

    def calculate_company_scores(self):
        scores = pd.DataFrame()
        scores['Company'] = self.df['Company Name']

        # Calculate various scores
        if 'Certifications' in self.df.columns:
            scores['Certification_Score'] = self.df['Certifications'].str.count(',') + 1

        scores['Market_Presence'] = self.df['Geographical Presence'].apply(
            lambda x: 10 if 'Global' in x else 5)

        if 'Innovation/Research Areas' in self.df.columns:
            scores['Innovation_Score'] = self.df['Innovation/Research Areas'].str.count(',') + 1

        scores['Product_Diversity'] = self.df['Product Categories'].str.count(',') + 1

        scores['Total_Score'] = scores.select_dtypes(include=[np.number]).sum(axis=1)
        return scores

    def analyze_market_segments(self):
        segments = {}
        if 'Health Segments' in self.df.columns:
            for _, row in self.df.iterrows():
                if pd.notna(row['Health Segments']):
                    for segment in row['Health Segments'].split(','):
                        segment = segment.strip()
                        segments[segment] = segments.get(segment, 0) + 1
        return pd.DataFrame(list(segments.items()), columns=['Segment', 'Count'])

def create_matplotlib_figure(figure_func):
    fig, ax = plt.subplots(figsize=(10, 6))
    figure_func(ax)
    plt.tight_layout()
    return fig

def create_visualizations(analyzer):
    st.title("Pharmaceutical Industry Analysis Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters")
    company_type = st.sidebar.multiselect(
        "Select Company Type",
        options=analyzer.df['Company Type'].unique(),
        default=analyzer.df['Company Type'].unique()
    )

    # Filter data
    filtered_df = analyzer.df[analyzer.df['Company Type'].isin(company_type)]

    # Company Type Distribution
    st.subheader("Company Type Distribution")
    st.write("This chart shows the distribution of different company types in the dataset.")
    fig = create_matplotlib_figure(lambda ax:
        sns.countplot(data=filtered_df, x='Company Type', ax=ax)
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Manufacturing Capabilities
    st.subheader("Manufacturing Capabilities")
    st.write("This chart shows the count of companies with different manufacturing capabilities.")
    manufacturing_cols = ['Manufacturer_Binary', 'Brand_Binary', 'Distributor_Binary']
    manufacturing_data = filtered_df[manufacturing_cols].sum()
    fig = create_matplotlib_figure(lambda ax:
        manufacturing_data.plot(kind='bar', ax=ax)
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Geographic Presence
    st.subheader("Geographic Presence")
    st.write("This chart shows the geographic presence of the companies in the dataset.")
    presence_data = filtered_df['Geographical Presence'].value_counts()
    fig = create_matplotlib_figure(lambda ax:
        presence_data.plot(kind='barh', ax=ax)
    )
    st.pyplot(fig)

    # Health Segments Analysis
    st.subheader("Health Segments Distribution")
    st.write("This chart shows the distribution of health segments in the dataset.")
    segments = analyzer.analyze_market_segments()
    if not segments.empty:
        fig = create_matplotlib_figure(lambda ax:
            sns.barplot(data=segments, x='Count', y='Segment', ax=ax)
        )
        st.pyplot(fig)

    # Company Performance Metrics
    st.subheader("Company Performance Analysis")
    st.write("This chart shows the performance of companies based on market presence, innovation score, and product diversity.")
    scores = analyzer.calculate_company_scores()
    fig = create_matplotlib_figure(lambda ax:
        sns.scatterplot(data=scores, x='Market_Presence', y='Innovation_Score',
                       size='Product_Diversity', ax=ax)
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Top Companies Analysis
    st.subheader("Top Companies by Total Score")
    st.write("This chart shows the top companies based on their total scores.")
    top_companies = scores.nlargest(10, 'Total_Score')
    fig = create_matplotlib_figure(lambda ax:
        sns.barplot(data=top_companies, x='Total_Score', y='Company', ax=ax)
    )
    st.pyplot(fig)

    # Product Categories Analysis
    st.subheader("Product Categories Distribution")
    st.write("This chart shows the distribution of product categories in the dataset.")
    product_cats = filtered_df['Product Categories'].str.split(',').explode().value_counts()
    fig = create_matplotlib_figure(lambda ax:
        product_cats.head(10).plot(kind='barh', ax=ax)
    )
    st.pyplot(fig)

    # Detailed Analysis Tables
    st.header("Detailed Analysis")

    # Company Scores Table
    st.subheader("Company Scores")
    st.dataframe(scores)

    # Summary Statistics
    st.subheader("Summary Statistics")
    summary_stats = pd.DataFrame({
        'Total Companies': len(filtered_df),
        'Manufacturers': filtered_df['Manufacturer_Binary'].sum(),
        'Brands': filtered_df['Brand_Binary'].sum(),
        'Distributors': filtered_df['Distributor_Binary'].sum(),
    }, index=[0])
    st.dataframe(summary_stats)

    # Download Section
    st.header("Download Analysis")

    # Prepare download data
    download_df = pd.concat([
        filtered_df,
        scores.set_index('Company')
    ], axis=1)

    csv = download_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis as CSV",
        data=csv,
        file_name="pharma_analysis.csv",
        mime="text/csv",
    )

    # Additional Functionalities
    st.header("Additional Analysis")

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    st.write("This heatmap shows the correlation between different numeric variables in the dataset.")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig = create_matplotlib_figure(lambda ax:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    )
    st.pyplot(fig)

    # Box Plot for Product Diversity
    st.subheader("Box Plot for Product Diversity")
    st.write("This box plot shows the distribution of product diversity scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.boxplot(data=scores, x='Product_Diversity', ax=ax)
    )
    st.pyplot(fig)

    # Violin Plot for Innovation Score
    st.subheader("Violin Plot for Innovation Score")
    st.write("This violin plot shows the distribution and density of innovation scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.violinplot(data=scores, x='Innovation_Score', ax=ax)
    )
    st.pyplot(fig)

    # Histogram for Market Presence
    st.subheader("Histogram for Market Presence")
    st.write("This histogram shows the frequency distribution of market presence scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.histplot(data=scores, x='Market_Presence', kde=True, ax=ax)
    )
    st.pyplot(fig)

    # Bar Plot for Market Presence
    st.subheader("Bar Plot for Market Presence")
    st.write("This bar plot shows the count of companies based on their market presence scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.countplot(data=scores, x='Market_Presence', ax=ax)
    )
    st.pyplot(fig)

    # Line Plot for Certification Score
    st.subheader("Line Plot for Certification Score")
    st.write("This line plot shows the trend of certification scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.lineplot(data=scores, x=scores.index, y='Certification_Score', ax=ax)
    )
    st.pyplot(fig)

    # Bar Plot for Geographical Presence
    st.subheader("Bar Plot for Geographical Presence")
    st.write("This bar plot shows the count of companies in different geographical regions.")
    fig = create_matplotlib_figure(lambda ax:
        sns.countplot(data=filtered_df, x='Geographical Presence', ax=ax)
    )
    st.pyplot(fig)

    # Pie Chart for Company Type
    st.subheader("Pie Chart for Company Type")
    st.write("This pie chart shows the proportion of different company types.")
    company_type_counts = filtered_df['Company Type'].value_counts()
    fig = create_matplotlib_figure(lambda ax:
        company_type_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    )
    st.pyplot(fig)

    # Stacked Bar Chart for Manufacturing Capabilities
    st.subheader("Stacked Bar Chart for Manufacturing Capabilities")
    st.write("This stacked bar chart shows the count of manufacturing capabilities.")
    manufacturing_data_stacked = manufacturing_data.reset_index()
    manufacturing_data_stacked.columns = ['Capability', 'Count']
    fig = create_matplotlib_figure(lambda ax:
        manufacturing_data_stacked.set_index('Capability').T.plot(kind='bar', stacked=True, ax=ax)
    )
    st.pyplot(fig)

    # Area Plot for Product Diversity
    st.subheader("Area Plot for Product Diversity")
    st.write("This area plot shows the product diversity scores over time.")
    fig = create_matplotlib_figure(lambda ax:
        scores['Product_Diversity'].plot(kind='area', ax=ax)
    )
    st.pyplot(fig)

    # Treemap for Health Segments
    st.subheader("Treemap for Health Segments")
    st.write("This treemap shows the hierarchical structure of health segments.")
    fig, ax = plt.subplots(figsize=(10, 6))
    squarify.plot(sizes=segments['Count'], label=segments['Segment'], ax=ax)
    plt.axis('off')
    st.pyplot(fig)

    # Word Cloud for Product Categories
    st.subheader("Word Cloud for Product Categories")
    st.write("This word cloud shows the frequency of product categories in the dataset.")
    product_categories_text = ' '.join(filtered_df['Product Categories'].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(product_categories_text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Sunburst Chart for Geographical Presence
    st.subheader("Sunburst Chart for Geographical Presence")
    st.write("This sunburst chart shows the geographical presence of companies.")
    geo_presence_counts = filtered_df['Geographical Presence'].value_counts().reset_index()
    geo_presence_counts.columns = ['Geographical Presence', 'Count']
    fig = px.sunburst(geo_presence_counts, path=['Geographical Presence'], values='Count')
    st.plotly_chart(fig)

    # 3D Scatter Plot for Company Scores
    st.subheader("3D Scatter Plot for Company Scores")
    st.write("This 3D scatter plot shows the relationship between market presence, innovation score, and product diversity.")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scores['Market_Presence'], scores['Innovation_Score'], scores['Product_Diversity'])
    ax.set_xlabel('Market Presence')
    ax.set_ylabel('Innovation Score')
    ax.set_zlabel('Product Diversity')
    st.pyplot(fig)

    # Interactive Table for Detailed Analysis
    st.subheader("Interactive Table for Detailed Analysis")
    st.dataframe(filtered_df)

    # Interactive Map for Geographical Presence
    if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        st.subheader("Interactive Map for Geographical Presence")
        st.write("This interactive map shows the geographical presence of companies.")
        m = folium.Map(location=[20, 77], zoom_start=5)
        for _, row in filtered_df.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=row['Company Name']).add_to(m)
        folium_static(m)
    else:
        st.warning("Latitude and Longitude columns are missing. Unable to display the interactive map.")

    # Time Series Analysis for Company Scores
    st.subheader("Time Series Analysis for Company Scores")
    st.write("This time series analysis shows the trend of total scores over time.")
    scores['Date'] = pd.to_datetime(scores.index)
    scores.set_index('Date', inplace=True)
    fig = create_matplotlib_figure(lambda ax:
        scores['Total_Score'].plot(ax=ax)
    )
    st.pyplot(fig)

    # Heatmap for Correlation Matrix
    st.subheader("Heatmap for Correlation Matrix")
    st.write("This heatmap shows the correlation between different numeric variables in the dataset.")
    fig = create_matplotlib_figure(lambda ax:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    )
    st.pyplot(fig)

    # Box Plot for Innovation Score
    st.subheader("Box Plot for Innovation Score")
    st.write("This box plot shows the distribution of innovation scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.boxplot(data=scores, x='Innovation_Score', ax=ax)
    )
    st.pyplot(fig)

    # Violin Plot for Product Diversity
    st.subheader("Violin Plot for Product Diversity")
    st.write("This violin plot shows the distribution and density of product diversity scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.violinplot(data=scores, x='Product_Diversity', ax=ax)
    )
    st.pyplot(fig)

    # Histogram for Total Score
    st.subheader("Histogram for Total Score")
    st.write("This histogram shows the frequency distribution of total scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.histplot(data=scores, x='Total_Score', kde=True, ax=ax)
    )
    st.pyplot(fig)

    # Line Plot for Innovation Score
    st.subheader("Line Plot for Innovation Score")
    st.write("This line plot shows the trend of innovation scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.lineplot(data=scores, x=scores.index, y='Innovation_Score', ax=ax)
    )
    st.pyplot(fig)

    # Bar Plot for Certification Score
    st.subheader("Bar Plot for Certification Score")
    st.write("This bar plot shows the count of certification scores.")
    fig = create_matplotlib_figure(lambda ax:
        sns.countplot(data=scores, x='Certification_Score', ax=ax)
    )
    st.pyplot(fig)

    # Pie Chart for Product Diversity
    st.subheader("Pie Chart for Product Diversity")
    st.write("This pie chart shows the proportion of product diversity scores.")
    product_diversity_counts = scores['Product_Diversity'].value_counts()
    fig = create_matplotlib_figure(lambda ax:
        product_diversity_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    )
    st.pyplot(fig)

    # Stacked Bar Chart for Innovation Score
    st.subheader("Stacked Bar Chart for Innovation Score")
    st.write("This stacked bar chart shows the count of innovation scores.")
    innovation_score_stacked = scores['Innovation_Score'].value_counts().reset_index()
    innovation_score_stacked.columns = ['Innovation Score', 'Count']
    fig = create_matplotlib_figure(lambda ax:
        innovation_score_stacked.set_index('Innovation Score').T.plot(kind='bar', stacked=True, ax=ax)
    )
    st.pyplot(fig)

    # Area Plot for Market Presence
    st.subheader("Area Plot for Market Presence")
    st.write("This area plot shows the market presence scores over time.")
    fig = create_matplotlib_figure(lambda ax:
        scores['Market_Presence'].plot(kind='area', ax=ax)
    )
    st.pyplot(fig)

    # Treemap for Product Diversity
    st.subheader("Treemap for Product Diversity")
    st.write("This treemap shows the hierarchical structure of product diversity scores.")
    fig, ax = plt.subplots(figsize=(10, 6))
    squarify.plot(sizes=scores['Product_Diversity'], label=scores.index, ax=ax)
    plt.axis('off')
    st.pyplot(fig)

    # Word Cloud for Innovation/Research Areas
    st.subheader("Word Cloud for Innovation/Research Areas")
    st.write("This word cloud shows the frequency of innovation/research areas in the dataset.")
    innovation_areas_text = ' '.join(filtered_df['Innovation/Research Areas'].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(innovation_areas_text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Sunburst Chart for Certification Score
    st.subheader("Sunburst Chart for Certification Score")
    st.write("This sunburst chart shows the certification scores of companies.")
    cert_score_counts = scores['Certification_Score'].value_counts().reset_index()
    cert_score_counts.columns = ['Certification Score', 'Count']
    fig = px.sunburst(cert_score_counts, path=['Certification Score'], values='Count')
    st.plotly_chart(fig)

    # 3D Scatter Plot for Innovation Score
    st.subheader("3D Scatter Plot for Innovation Score")
    st.write("This 3D scatter plot shows the relationship between market presence, product diversity, and innovation score.")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scores['Market_Presence'], scores['Product_Diversity'], scores['Innovation_Score'])
    ax.set_xlabel('Market Presence')
    ax.set_ylabel('Product Diversity')
    ax.set_zlabel('Innovation Score')
    st.pyplot(fig)

def main():
    st.sidebar.title("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            analyzer = DataAnalyzer(df)
            create_visualizations(analyzer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload a file to begin analysis")

if __name__ == "__main__":
    main()
