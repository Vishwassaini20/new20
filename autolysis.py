import os
import sys
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import chardet  # To detect file encoding
import argparse

# Load environment variables from .env file
load_dotenv()

# Fetch AI Proxy token from .env file
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

# Define headers for API request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

# Function to request AI to generate the narrative story
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    prompt = f"""
    Below is a detailed summary and analysis of a dataset. Please generate a **rich and engaging narrative** about this dataset analysis, including:

    1. **The Data Received**: Describe the dataset with vivid language. What does the data represent? What are its features? Create a story around it.
    2. **The Analysis Carried Out**: Explain the analysis methods used—highlighting techniques like missing value handling, outlier detection, clustering, etc.
    3. **The Insights Discovered**: What were the key findings? What trends or patterns emerged that can be interpreted as discoveries?
    4. **The Implications of Findings**: How do these insights influence decisions? What actions can be taken based on the analysis? What recommendations would you give?

    **Dataset Summary**:
    {dataset_summary}
    
    **Dataset Info**:
    {dataset_info}
    
    **Visualizations**:
    {visualizations}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Will raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(1)

    return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else None

# Function to load dataset with automatic encoding detection
def load_data(file_path):
    try:
        # Detect encoding using chardet
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the encoding of the file
        encoding = result['encoding']   # Get the detected encoding

        # Load the dataset with the detected encoding
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Data loaded with {encoding} encoding.")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Perform basic analysis
def basic_analysis(data):
    summary = data.describe(include='all').to_dict()   # compute summary statistics
    missing_values = data.isnull().sum().to_dict()  # Missing values
    column_info = data.dtypes.to_dict()  # Column types
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Outlier detection using IQR (Interquartile Range)
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)  # Select numeric data
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Function to generate correlation matrix heatmap
def generate_correlation_matrix(data):
    # Ensure the output directory exists
    data = data.select_dtypes(include=[np.number])
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    corr_path = "correlation_matrix.png"
    plt.savefig(corr_path, bbox_inches='tight')  # Save in the current directory
    plt.close()
    return corr_path

# Pairplot visualization
def generate_pairplot(data):
    # Select numeric data
    numeric_data = data.select_dtypes(include=[np.number])
    pairplot_path = "pairplot.png"
    sns.pairplot(numeric_data)
    plt.title("Pairplot of Numeric Features")
    plt.savefig(pairplot_path, bbox_inches='tight')
    plt.close()
    return pairplot_path

# Feature importance plot (if applicable)
def feature_importance(data):
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if len(numeric_data.columns) > 1:  # Ensure there are at least two columns
        X = numeric_data.dropna(axis=1, how='any')
        y = data.iloc[:, -1]  # Assuming the last column is the target

        model = RandomForestClassifier()
        model.fit(X, y)

        feature_importance = model.feature_importances_
        feature_names = X.columns

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_names, y=feature_importance)
        plt.title("Feature Importance")
        feature_importance_path = "feature_importance.png"
        plt.xticks(rotation=90)
        plt.savefig(feature_importance_path, bbox_inches='tight')
        plt.close()

        return feature_importance_path
    return None

# DBSCAN clustering
def dbscan_clustering(data):
    # Ensure the output directory exists
    numeric_data = data.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="viridis")
    plt.title("DBSCAN Clustering")
    dbscan_path = "dbscan_clusters.png"
    plt.savefig(dbscan_path, bbox_inches='tight')  # Save in the current directory
    plt.close()
    return dbscan_path

# Hierarchical clustering
def hierarchical_clustering(data):
    # Ensure the output directory exists
    numeric_data = data.select_dtypes(include=np.number).dropna()
    linked = linkage(numeric_data, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    hc_path = "hierarchical_clustering.png"
    plt.savefig(hc_path, bbox_inches='tight')  # Save in the current directory
    plt.close()
    return hc_path

# Function to save results to README.md in the current directory
def save_readme(content):
    try:
        readme_path = "README.md"
        with open(readme_path, "w") as f:
            f.write(content)
        print(f"README saved in the current directory.")
    except Exception as e:
        print(f"Error saving README: {e}")
        sys.exit(1)

# Function to analyze and generate output
def analyze_and_generate_output(file_path):
    # Load data
    data = load_data(file_path)
    print("Data loaded")
    
    # Perform basic analysis
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    # Generate visualizations and save file paths
    image_paths = {}
    image_paths['correlation_matrix'] = generate_correlation_matrix(data)
    image_paths['pairplot'] = generate_pairplot(data)
    image_paths['feature_importance'] = feature_importance(data) if len(data.select_dtypes(include=[np.number]).columns) > 1 else None
    image_paths['dbscan_clusters'] = dbscan_clustering(data)
    image_paths['hierarchical_clustering'] = hierarchical_clustering(data)
    print("Images created:\n", image_paths)

    # Send data to LLM for analysis and suggestions
    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }
    
    prompt = (
        "You are a creative storyteller. "
        "Craft a compelling narrative based on this dataset analysis:\n\n"
        f"Data Summary: {data_info['summary']}\n\n"
        f"Missing Values: {data_info['missing_values']}\n\n"
        f"Outlier Analysis: {data_info['outliers']}\n\n"
        "Create a narrative covering these points:\n"
        f"Correlation matrix: {image_paths['correlation_matrix']},\n"
        f"Pairplot: {image_paths['pairplot']},\n"
        f"Feature Importance: {image_paths['feature_importance'] if image_paths['feature_importance'] else 'None'},\n"
        f"DBSCAN Clusters: {image_paths['dbscan_clusters']},\n"
        f"Hierarchical Clustering: {image_paths['hierarchical_clustering']}\n"
    )
    
    # Get the AI-generated story
    narrative = get_ai_story(data_info['summary'], data_info['missing_values'], image_paths)
    
    # Save the results to README
    save_readme(f"Dataset Analysis: {narrative}")
    
    return narrative, image_paths

# Main execution
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()
