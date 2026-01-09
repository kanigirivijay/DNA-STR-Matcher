# DNA-STR-Matcher
A next-gen forensic DNA STR analysis platform with ML classification, unique individual detection, and criminal matching, featuring interactive visualizations and downloadable reports.




STR Classification Using Machine Learning
Overview

This project focuses on classifying Short Tandem Repeats (STRs) using machine learning algorithms. STR classification is critical in forensic genetics and criminal investigations, where accurate identification of allele patterns can provide valuable insights.

The system leverages ensemble, kernel-based, and neural network models to handle nonlinear relationships and complex patterns in STR data. An interactive web interface allows visualization, exploration, and report generation.

Features

Machine Learning Models:

Random Forest (RF): Ensemble classifier using multiple decision trees.

Support Vector Machine (SVM): Kernel-based model to distinguish allele peak patterns.

Multi-Layer Perceptron (MLP): Neural network for capturing nonlinear relationships.

Interactive Web Interface: Built using Streamlit for user-friendly STR data exploration.

Visualization:

Interactive charts using Plotly

Criminal-individual connection networks using NetworkX

Data Handling: Label encoding, feature normalization, and tabular data management using pandas and NumPy

Report Generation: Downloadable reports with encoded outputs using base64

Installation

Clone the repository:

git clone https://github.com/your-username/str-classification.git


Navigate to the project directory:

cd str-classification


Install required packages:

pip install -r requirements.txt

Usage

Run the Streamlit app:

streamlit run app.py


Upload your STR dataset (CSV format).

Explore visualizations and predictions through the interactive UI.

Download classification reports directly from the app.

Workflow

Load STR dataset

Encode labels using LabelEncoder

Normalize input features with StandardScaler

Train and evaluate ML models on test data

Select the best model based on F1-score

Visualize results and generate reports

Libraries and Packages
Library	Purpose
Streamlit	Interactive web interface
pandas	Data handling
scikit-learn	ML model training & evaluation
plotly	Interactive charts
networkx	Visualizing connections
numpy	Data manipulation
base64	Encoding downloadable reports
time	Simulation & UI animations
Results

High accuracy in STR classification using ensemble and neural models

Interactive visualizations for allele patterns and connections

Easy-to-use web interface with downloadable reports

License

This project is licensed under the MIT License.
