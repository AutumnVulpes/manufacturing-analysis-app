# 🔬 Engineering Data Analysis Dashboard

A comprehensive AI-powered data visualization and analysis platform designed for engineering datasets, featuring advanced statistical analysis, machine learning insights, and intelligent data exploration capabilities.

---

## ✨ Core Features

### 📤 **Intelligent Data Upload & Processing**

**Smart CSV Processing**
- **Drag-and-drop file upload** with instant data validation
- **Automatic data cleaning** - removes missing values and identifies numeric columns
- **Intelligent data sampling** for optimal performance with large datasets (10k+ rows)
- **Real-time data preview** with statistical summaries

> **[GIF: Drag-and-drop CSV upload process showing file validation and data preview]**

**Data Quality Assurance**
- Automatic detection of numeric vs categorical columns
- Missing value handling with transparent reporting
- Data type validation and conversion
- Export capabilities for cleaned datasets

---

### 🤖 **AI-Powered Data Intelligence**

**Smart Dashboard Personalization**
- **Automatic title generation** - AI analyzes your dataset and creates professional dashboard titles
- **Context-aware naming** based on column patterns and data characteristics
- **Multi-provider AI support** (OpenAI, Google Gemini, OpenRouter)

> **[GIF: AI generating custom dashboard title from uploaded engineering data]**

**Intelligent Column Analysis**
- **AI-powered column pairing suggestions** with detailed reasoning
- **Embedded trend visualizations** showing data relationships at a glance
- **Statistical correlation insights** for meaningful data comparisons
- **Interactive suggestion display** with mini-charts and analysis rationale

> **[GIF: AI generating column comparison suggestions with embedded trend charts and reasoning]**

**Conversational Data Assistant**
- **Real-time AI chat interface** for data exploration
- **Context-aware responses** using your specific dataset
- **Statistical analysis on-demand** - ask questions about patterns, trends, correlations
- **Intelligent query filtering** - only responds to data-relevant questions
- **Persistent chat history** throughout your analysis session

> **[GIF: Chat interface showing user asking questions about data patterns and receiving AI insights]**

---

### 📊 **Advanced Principal Component Analysis (PCA)**

**Automated Dimensionality Reduction**
- **One-click PCA processing** with standardized data scaling
- **95% variance optimization** - automatically determines optimal component count
- **Interactive component selection** with real-time variance tracking
- **Enhanced dataset export** with principal components included

> **[GIF: PCA configuration process showing column selection and automated analysis]**

**Comprehensive PCA Visualization**
- **Interactive Scree Plots** - identify optimal number of components with elbow detection
- **Cumulative Variance Charts** - track explained variance across components
- **95% variance threshold indicators** with optimal component highlighting
- **Downloadable PCA visualizations** (PNG export)

> **[GIF: Navigating between scree plot and cumulative variance visualizations]**

**Mathematical Transparency**
- **Detailed PCA formulas** with LaTeX rendering for each principal component
- **Component coefficient display** showing exact mathematical transformations
- **Standardization formulas** with mean and standard deviation calculations
- **Expandable formula sections** for in-depth mathematical understanding

> **[IMAGE: Mathematical formulas display showing PCA component calculations with LaTeX formatting]**

---

### 📈 **Interactive Data Visualization**

**Dynamic Scatter Plot Analysis**
- **Customizable axis selection** from any numeric columns
- **Real-time plot updates** as you change configurations
- **Professional styling** with automatic axis labeling and scaling
- **High-resolution PNG export** for reports and presentations

> **[GIF: Creating scatter plots with different axis combinations and customization options]**

**Advanced Data Filtering**
- **Interactive range sliders** for precise data subset selection
- **Multi-column filtering** with independent range controls
- **Real-time plot updates** as filters are adjusted
- **Visual feedback** showing filtered data points and statistics

> **[GIF: Using filter sliders to focus on specific data ranges and seeing real-time plot updates]**

**Export & Sharing**
- **One-click PNG downloads** for all visualizations
- **High-quality image generation** suitable for professional reports
- **Batch export capabilities** for multiple chart types
- **Consistent styling** across all exported visualizations

---

### 🎛️ **Advanced Analytics Dashboard**

**Organized Workflow Interface**
- **Tabbed navigation system** for different analysis types:
  - **PCA Configuration** - dimensionality reduction setup
  - **PCA Formulas** - mathematical component details  
  - **Visualization Config** - scatter plot and filtering controls
  - **AI Data Assistant** - intelligent analysis and chat
- **Responsive design** adapting to different screen sizes
- **Persistent state management** - your settings are remembered

**Performance Optimization**
- **Intelligent caching** for faster repeated operations
- **Optimized data processing** for large datasets
- **Responsive UI updates** with minimal loading times
- **Memory-efficient operations** for sustained analysis sessions

**Statistical Insights**
- **Comprehensive data summaries** (mean, std dev, min/max, count)
- **Real-time statistics updates** as you filter data
- **Correlation analysis** through AI-powered suggestions
- **Pattern recognition** via principal component analysis

---

## 🛠️ **Technology Stack**

**Frontend & Visualization**
- **Streamlit** - Modern web app framework with custom CSS styling
- **Plotly Express** - Interactive, publication-quality charts
- **Responsive design** with dark/light theme support

**Data Processing & Analytics**
- **Pandas** - High-performance data manipulation and analysis
- **NumPy** - Numerical computing foundation
- **Scikit-learn** - Machine learning and statistical analysis
- **StandardScaler** - Data normalization for PCA

**AI Integration**
- **OpenAI API** - GPT models for intelligent analysis
- **Google Gemini** - Advanced language model support
- **OpenRouter** - Multi-provider AI access
- **Instructor** - Structured AI responses with Pydantic validation
- **Tenacity** - Robust retry mechanisms for API reliability

---

## 🚀 **Quick Start Guide**

### **1. Upload Your Data**
- Drag and drop your CSV file or click to browse
- Automatic data validation and cleaning
- Instant preview of your dataset

### **2. Configure AI Assistant (Optional)**
- Add your API key (OpenAI, Gemini, or OpenRouter)
- Get intelligent dashboard title generation
- Enable AI-powered column suggestions

### **3. Explore with PCA**
- Automatic principal component analysis
- View scree plots and variance explanations
- Export enhanced dataset with PC components

### **4. Create Visualizations**
- Select X and Y axes for scatter plots
- Apply interactive filters to focus your analysis
- Download high-quality PNG exports

### **5. Get AI Insights**
- Ask questions about your data patterns
- Receive context-aware statistical analysis
- Build persistent analysis conversations

---

## 🎯 **Use Cases**

**Engineering Design & Analysis**
- Explore relationships between design parameters and performance
- Identify key variables affecting system behavior
- Optimize engineering processes and configurations

**Quality Control & Testing**
- Analyze test metrics and identify quality patterns
- Correlate process parameters with output characteristics
- Reduce dimensionality of complex sensor data

**Research & Development**
- Analyze experimental datasets with multiple variables
- Discover hidden patterns through PCA
- Generate hypotheses through AI-powered insights

---

## 🔒 **Privacy & Security**

- **Local processing** - your data never leaves your environment
- **API key security** - keys are stored only in session memory
- **No data persistence** - uploaded files are processed in memory only
- **Open source** - full transparency of data handling

---

<div align="center">

**Made with ❤️ by [@AutumnVulpes](https://github.com/AutumnVulpes)**

[🌟 Star this repo](https://github.com/AutumnVulpes/manufacturing-analysis-app) | [🐛 Report Issues](https://github.com/AutumnVulpes/manufacturing-analysis-app/issues) | [💡 Request Features](https://github.com/AutumnVulpes/manufacturing-analysis-app/discussions)

</div>
