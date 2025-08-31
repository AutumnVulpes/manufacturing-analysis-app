# Engineering Data Analysis Dashboard

**AI-powered data visualization and analysis platform for engineering datasets**

![Dashboard Demo](https://i.imgur.com/G33RVjW.gif)

_Transform your engineering data into actionable insights with advanced statistical analysis, machine learning, and intelligent AI assistance._
_Try out the app [here!](https://vulpes-data-visualization.streamlit.app/)

---

## Quick Start 🚀

### Requirements

*   Python 3.8+
*   Modern web browser

### Install & Run

```
git clone https://github.com/AutumnVulpes/manufacturing-analysis-app
cd manufacturing-analysis-app
pip install -r requirements.txt
streamlit run app.py
```

### Get Started

1.  **Upload your CSV** - Drag and drop your engineering dataset
2.  **Configure AI** (optional) - Add API key for intelligent insights
3.  **Explore data** - Use PCA, visualizations, and AI chat

---

## Core Features

### 📊 **Smart Data Processing**

*   **Automatic data cleaning** and validation
*   **Intelligent sampling** for large datasets (10k+ rows)
*   **Real-time preview** with statistical summaries

<img src="https://i.imgur.com/mYdhXB0.gif" alt="Data Upload Demo" width="50%" height="50%">

### 🤖 **AI-Powered Analysis**

*   **Automatic title generation** based on your dataset
*   **Column pairing suggestions** with reasoning
*   **Conversational data assistant** for real-time insights

<img src="https://i.imgur.com/BIiDxVm.gif" alt="AI assistant demo" width="50%" height="50%">

### 📈 **Advanced Analytics**

*   **Principal Component Analysis** with 95% variance optimization
*   **Interactive scatter plots** with real-time filtering
*   **Mathematical formulas** with LaTeX rendering
*   **Deeper insight** with scree and cumulative variance plots

<img src="https://i.imgur.com/y0SmMtl.gif" alt="PCA analysis demo" width="50%" height="50%">

### 💬 **Interactive Insights**

*   **Real-time chat interface** for data exploration
*   **Statistical analysis on-demand**

---

## Advanced Usage

### AI Configuration

Configure your preferred AI provider for enhanced features:

```python
# Supported providers
providers = ["OpenAI", "Google Gemini", "OpenRouter"]
```

Add your API key in the dashboard to enable:

*   Intelligent column suggestions
*   Automatic title generation
*   Conversational data analysis

### PCA Optimization

The system automatically determines optimal principal components:

*   **Standardized data scaling** for accurate analysis
*   **95% variance threshold** for component selection
*   **Enhanced dataset export** with principal components

### Data Export

Download your processed data:

*   **Cleaned datasets** with missing values removed
*   **PCA-enhanced data** with principal components
*   **High-quality visualizations** as PNG files

---

## Use Cases

**Engineering Design & Testing**

*   Analyze relationships between design parameters and performance
*   Identify key variables affecting system behavior
*   Optimize engineering processes through data insights

**Research & Development**

*   Explore experimental datasets with multiple variables
*   Discover hidden patterns through principal component analysis
*   Generate data-driven hypotheses with AI assistance

---

## Technology Stack

*   **Frontend**: Streamlit with custom CSS
*   **Visualization**: Plotly Express for interactive charts
*   **Analytics**: Pandas, NumPy, Scikit-learn
*   **AI Integration**: OpenAI, Google Gemini, OpenRouter APIs
*   **Data Processing**: Automated cleaning and PCA optimization

---

## Privacy & Security

*   **Local processing** - Your data stays in your environment
*   **Session-only storage** - No persistent data retention
*   **API key security** - Keys stored only in session memory
*   **Open source** - Full transparency of data handling

---

## Contributing

Found an issue or have a feature request?

*   🐛 [Report bugs](https://github.com/AutumnVulpes/manufacturing-analysis-app/issues)
*   💡 [Request features](https://github.com/AutumnVulpes/manufacturing-analysis-app/discussions)
*   ⭐ [Star this repo](https://github.com/AutumnVulpes/manufacturing-analysis-app)

---

**Built by** [**@AutumnVulpes**](https://github.com/AutumnVulpes)
