# Manufacturing Data Analysis Dashboard

A comprehensive data visualization and analysis tool designed for engineering datasets, featuring AI-powered insights and advanced statistical analysis capabilities.

## 🚀 Features

### 📊 Interactive Data Visualization
- **Scatter Plot Analysis**: Create dynamic scatter plots with customizable X and Y axes
- **Real-time Data Filtering**: Interactive sliders to filter data ranges for focused analysis
- **Downloadable Charts**: Export visualizations as PNG files for reports and presentations

### 🔄 AI-Powered Insights
- **Smart Column Suggestions**: AI analyzes your dataset and recommends meaningful column pairings for visualization
- **Intelligent Title Generation**: Automatically generates professional dashboard titles based on your data
- **Multi-Provider Support**: Compatible with OpenAI, Google Gemini, and OpenRouter APIs
- **Interactive Suggestion Display**: View AI recommendations with embedded trend charts and detailed reasoning

### 📈 Principal Component Analysis (PCA)
- **Automated PCA Processing**: Reduce dimensionality while preserving data variance
- **Scree Plot Visualization**: Identify optimal number of components
- **Cumulative Variance Analysis**: Track explained variance across components
- **Mathematical Formulas**: View detailed PCA component formulas with LaTeX rendering
- **95% Variance Optimization**: Automatically determine minimum components for 95% variance retention

### 🔧 Data Processing
- **Automatic Data Cleaning**: Removes missing values and identifies numeric columns
- **Smart Data Sampling**: Optimizes performance by intelligently sampling large datasets
- **Statistical Summaries**: Comprehensive statistics including mean, standard deviation, min/max values
- **Export Capabilities**: Download cleaned data and PCA-enhanced datasets as CSV files

### 💡 User Experience
- **Responsive Design**: Clean, modern interface with dark/light theme support
- **Tabbed Navigation**: Organized workflow with dedicated tabs for different analysis types
- **Real-time Updates**: Dynamic interface that responds to data changes instantly
- **Performance Optimized**: Cached operations and efficient data handling for smooth experience

## 🛠 Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly Express with interactive charts
- **AI Integration**: OpenAI API, Google Gemini, OpenRouter with Instructor for structured responses
- **Statistical Analysis**: Principal Component Analysis, StandardScaler normalization

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly (≥6.2.0)
- Scikit-learn
- OpenAI
- Instructor
- Pydantic

## 🚀 Quick Start

1. Upload a CSV file with numeric data
2. Configure API key for AI features (optional)
3. Explore automated PCA analysis
4. Get AI-powered column comparison suggestions
5. Create interactive scatter plots with filtering
6. Download visualizations and processed data

---

*Made by [@AutumnVulpes](https://github.com/AutumnVulpes/manufacturing-analysis-app)*
