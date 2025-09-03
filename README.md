# Spark for Machine Learning & AI
### LinkedIn Learning Course Materials

[![Course Rating](https://img.shields.io/badge/Rating-4.5%2F5-brightgreen)](https://www.linkedin.com/learning/spark-for-machine-learning-ai/)
[![Duration](https://img.shields.io/badge/Duration-1h%2051m-blue)](https://www.linkedin.com/learning/spark-for-machine-learning-ai/)
[![Level](https://img.shields.io/badge/Level-Beginner-green)](https://www.linkedin.com/learning/spark-for-machine-learning-ai/)

> *A comprehensive introduction to Apache Spark's machine learning capabilities using MLlib*

**Instructor:** Dan Sullivan - Data Architect, Author, and Instructor  
**Released:** November 7, 2017  
**Platform:** LinkedIn Learning

---

## 🎯 Course Overview

This course provides hands-on experience with Apache Spark's MLlib library, focusing on practical machine learning workflows rather than theoretical implementations. Designed for data scientists and analysts who want to leverage Spark's distributed computing power for machine learning tasks without building custom ML libraries from scratch.

### Learning Objectives

- **Machine Learning Workflows** - End-to-end ML pipeline development
- **DataFrame Operations** - Efficient data organization and manipulation
- **Data Preprocessing** - Feature engineering and data preparation techniques
- **Clustering Analysis** - Unsupervised learning for data exploration
- **Classification Methods** - Supervised learning for categorical predictions
- **Regression Techniques** - Numerical prediction algorithms
- **Recommendation Systems** - Collaborative filtering approaches

---

## 📁 Repository Structure

```
spark-ml-course/
├── Ch01/
│   └── 01_04/
│       └── employee.txt              # Employee dataset for DataFrame operations
├── Ch03/
│   └── 03_02/
│       └── clustering_dataset.csv    # Numerical data for clustering exercises
├── Handout - Spark Reference.pdf    # Quick reference guide for Spark ML
└── README.md                        # This file
```

---

## 📊 Datasets

### Employee Dataset (`employee.txt`)
A comprehensive employee information dataset featuring:

| Field | Type | Description | ML Use Case |
|-------|------|-------------|-------------|
| `id` | Integer | Unique employee identifier | Index/Key |
| `last_name` | String | Employee surname | Feature engineering |
| `email` | String | Contact information | Text processing |
| `gender` | Categorical | Male/Female | Classification target |
| `department` | Categorical | Business unit | Classification/Clustering |
| `start_date` | Date | Employment start date | Feature derivation |
| `salary` | Numerical | Annual compensation | Regression target |
| `job_title` | String | Position description | Text analysis |
| `region_id` | Integer | Geographic region | Categorical feature |

**Sample Data:**
```csv
id,last_name,email,gender,department,start_date,salary,job_title,region_id
1,'Kelley','rkelley0@soundcloud.com','Female','Computers','10/2/2009',67470,'Structural Engineer',2
2,'Armstrong','sarmstrong1@infoseek.co.jp','Male','Sports','3/31/2008',71869,'Financial Advisor',2
```

### Clustering Dataset (`clustering_dataset.csv`)
A clean numerical dataset optimized for clustering algorithm demonstrations:

| Field | Type | Description |
|-------|------|-------------|
| `col1` | Float | Numerical feature 1 |
| `col2` | Float | Numerical feature 2 |
| `col3` | Float | Numerical feature 3 |

**Sample Data:**
```csv
col1,col2,col3
7,4,1
7,7,9
7,9,6
```

---

## 🛠 Technical Reference

### Core Spark ML Modules

| Module | Purpose |
|--------|---------|
| `pyspark.ml.classification` | Classification algorithms |
| `pyspark.ml.clustering` | Clustering algorithms |
| `pyspark.ml.regression` | Regression algorithms |
| `pyspark.ml.feature` | Preprocessing functionality |
| `pyspark.ml.evaluation` | Model evaluation metrics |
| `pyspark.ml.linalg` | Vector operations |

### Essential DataFrame Operations

```python
# Data Loading
df = spark.read.csv("path/to/file.csv", header=True, inferSchema=True)

# Data Exploration
df.printSchema()           # Display schema
df.columns                 # Get column names
df.count()                 # Row count
df.take(10)               # First 10 rows

# Data Manipulation
df.filter("salary >= 100000")  # Boolean filtering
df.select("column").show()      # Column selection and display
```

### Preprocessing Pipeline

| Function | Purpose | Use Case |
|----------|---------|----------|
| `MinMaxScaler` | Normalize to [0,1] range | Neural networks, distance-based algorithms |
| `StandardScaler` | Standardize (μ=0, σ=1) | Linear models, PCA |
| `StringIndexer` | Categorical to numeric mapping | ML algorithm compatibility |
| `VectorAssembler` | Combine features into vector | MLlib requirement |
| `Bucketizer` | Continuous to categorical | Histogram creation, discretization |
| `Tokenizer` | Text to word tokens | NLP preprocessing |

### Algorithm Selection Guide

#### Clustering
- **KMeans**: Standard choice for exploratory data analysis
- **Bisecting KMeans**: Hierarchical approach, efficient for large datasets

#### Classification
- **NaiveBayes**: Fast, effective with independent features
- **DecisionTreeClassifier**: Interpretable, handles mixed data types
- **MultilayerPerceptronClassifier**: Complex patterns, non-linear separation

#### Regression
- **LinearRegression**: Baseline algorithm for numerical prediction
- **DecisionTreeRegressor**: Non-parametric alternative
- **GBTRegressor**: Gradient-boosted trees for complex relationships

---

## 🚀 Getting Started

### Prerequisites
- Apache Spark installation
- Python 3.x with PySpark
- Basic familiarity with Python and DataFrames

### Quick Start
```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Initialize Spark
spark = SparkSession.builder.appName("SparkML").getOrCreate()

# Load data
df = spark.read.csv("employee.txt", header=True, inferSchema=True)

# Basic exploration
df.printSchema()
df.show(5)
```

### Development Environment
```bash
# Start PySpark shell
pyspark

# Start Spark-Scala shell  
spark-shell

# Start SparkR (requires R installation)
sparkR
```

---

## 📚 Course Skills Covered

- **Apache Spark ML** - Distributed machine learning workflows
- **Data Engineering** - ETL processes for ML pipelines  
- **Feature Engineering** - Data preparation and transformation
- **Model Selection** - Algorithm choice based on problem type
- **Scalable Computing** - Distributed processing for large datasets

---

## 🎯 Learning Outcomes

Upon completion, you will be able to:

✅ Load and manipulate data using Spark DataFrames  
✅ Implement preprocessing pipelines for ML-ready data  
✅ Apply clustering algorithms for data exploration  
✅ Build classification models for categorical prediction  
✅ Develop regression models for numerical forecasting  
✅ Design recommendation systems using collaborative filtering  
✅ Evaluate model performance using appropriate metrics

---

## 📖 Additional Resources

- [Apache Spark Official Documentation](https://spark.apache.org/docs/latest/)
- [MLlib Programming Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)

---

## 📄 License

Course materials are provided under LinkedIn Learning's standard license terms. Please refer to the original course for usage guidelines.

---

*This repository contains supplementary materials for educational purposes. For the complete course experience, please access the full LinkedIn Learning course.*
