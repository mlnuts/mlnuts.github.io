var ptx_lunr_search_style = "textbook";
var ptx_lunr_docs = [
{
  "id": "front-colophon",
  "level": "1",
  "url": "front-colophon.html",
  "type": "Colophon",
  "number": "",
  "title": "Colophon",
  "body": "  "
},
{
  "id": "sec-Descriptive-Statistics",
  "level": "1",
  "url": "sec-Descriptive-Statistics.html",
  "type": "Section",
  "number": "1.1",
  "title": "Descriptive Statistics",
  "body": " Descriptive Statistics   Descriptive statistics summarize key features of a dataset, providing insights into its central tendency, dispersion, and shape. This process, known as Exploratory Data Analysis (EDA) , helps identify patterns and trends before applying advanced statistical methods. Common measures include mean, median, mode, variance, standard deviation, range, quartiles, and visualizations like histograms and boxplots. These tools are essential for understanding data in fields like education, finance, and science.    Measures of Central Tendency  Measures of central tendency describe the \"typical\" value in a dataset.     Mean (Average) : The mean, denoted , is the sum of all data points divided by their count. For a dataset with points, the mean is: where .  Example: For student grades , the mean is:   The mean is sensitive to outliers. For income data , the mean is , skewed by the outlier, while the median better represents the typical value.     Median : The median is the middle value in a sorted dataset, where 50% of the data lies below and above. For odd , it’s the middle value; for even , it’s the average of the two middle values.  Example: For (sorted), median = 90. For , median = (90 + 91)\/2 = 90.5. For incomes , median = 35000, robust to the outlier.     Mode : The mode is the most frequent value. A dataset may have no mode, one mode (unimodal), or multiple modes (bimodal or multimodal).  Example: has mode 90. is bimodal (90, 95). has no mode.     Comparison : Consider incomes . Mean = 225000, median = 35000, mode = none. The median best reflects the typical income due to the outlier. See for a visual comparison.   Density plot of incomes with mean, median, and no mode.   Density plot showing central tendency measures.     Code for density plot of central tendency measures (new)  import numpy as np import matplotlib.pyplot as plt from scipy.stats import gaussian_kde data = [20000, 30000, 35000, 40000, 1000000] kde = gaussian_kde(data, bw_method=0.5) x = np.linspace(0, 1100000, 1000) y = kde(x) plt.figure(figsize=(8, 5)) plt.plot(x, y, label='Density') plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean = {np.mean(data):.0f}') plt.axvline(np.median(data), color='green', linestyle='-', label=f'Median = {np.median(data):.0f}') plt.xlabel('Income') plt.ylabel('Density') plt.title('Central Tendency Measures for Income Data') plt.legend() plt.grid(True, alpha=0.3) plt.savefig('central-tendency.png') plt.show()     Measures of Dispersion  Dispersion measures how spread out data is around the central tendency.     Variance and Standard Deviation : Variance ( ) measures average squared deviation from the mean; standard deviation ( ) is its square root, in the same units as the data.  For a population: For a sample (unbiased estimate): where is the sample mean.  Example: For grades , . Population variance:  . Sample variance: , .  Low variance (e.g., ) vs. high variance (e.g., ) shows tighter vs. wider spread as illustrated in .   Comparing low and high variance datasets.   Comparison of low and high variance data.     Code for variance comparison histograms (new)  import matplotlib.pyplot as plt import numpy as np from scipy.stats import norm x = np.linspace(-10, 10, 1000) low_sig, high_sig = 0.2, 2.0 low_var = norm.pdf(x, loc=0, scale=low_sig) high_var = norm.pdf(x, loc=0, scale=high_sig) fig, ax = plt.subplots(1, 1, figsize=(10, 4)) ax.plot(x, low_var, 'ob', markersize = 2, label=f\"low variance, var = {low_sig**2:.2f}\") ax.plot(x, high_var, 'xr', markersize = 2, label=f\"high variance, var = {high_sig**2:.2f}\") ax.set_title(f'Variance Comparison') ax.set_xlabel('Grade') ax.set_ylabel('Scaled Frequency') plt.tight_layout() plt.legend() plt.show()      Range and Quartiles : Range = max - min. Quartiles divide sorted data into four parts: Q1 (25th percentile), Q2 (median, 50th), Q3 (75th). Use linear interpolation: position = , where .  Example: For grades , . Median (Q2) = 85. Q1 = 75, Q3 = 95. Range = 100 - 70 = 30. IQR = Q3 - Q1 = 20. Outliers: below Q1 - 1.5*IQR = 45 or above Q3 + 1.5*IQR = 125. These grades have no outliers.      Distribution Shape   Histogram : Histograms show frequency distributions by grouping data into bins.  Example: For grades , with bin size 10 from 70 to 110, see .   Histogram of Grades    Bin  Range  Data  Count  Frequency    1    3  0.333    2    3  0.333    3    2  0.222    4    1  0.111     See for a histogram with mean and median.   Histogram of grades with mean and median.   Histogram with mean and median lines.     Updated histogram with mean and median  import matplotlib.pyplot as plt import numpy as np data = [70, 75, 75, 80, 80, 85, 90, 95, 100] bin_size = 10 bins = np.arange(70, 110, bin_size) plt.figure(figsize=(8, 5)) plt.hist(data, bins=bins, edgecolor='black', alpha=0.7) mean = np.mean(data) median = np.median(data) plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.1f}') plt.axvline(median, color='green', linestyle='-', label=f'Median = {median:.1f}') plt.xlabel('Grade') plt.ylabel('Frequency') plt.title('Histogram of Student Grades') plt.xticks(bins) plt.grid(axis='y', alpha=0.3) plt.legend() plt.savefig('.\/images\/essential-probability-and-statistics\/histogram.png') plt.show()    Boxplot : Boxplots show min, Q1, median, Q3, max (whiskers), and outliers (points beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR).  Example: For grades with an outlier , Q1 = 77.5, Q2 = 87.5, Q3 = 97.5, IQR = 20. Outliers: ≤150 is above Q3 + 1.5*IQR = 127.5. See .   Boxplot of grades with annotated quartiles and outlier.   Boxplot with one outlier.     Updated boxplot with annotations  import matplotlib.pyplot as plt import numpy as np data = [70, 75, 80, 85, 90, 95, 100, 150] plt.figure(figsize=(8, 4)) bp = plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red')) q1, median, q3 = np.percentile(data, [25, 50, 75]) plt.text(q1 - 5, 1.1, 'Q1', ha='right') plt.text(median, 1.1, 'Median', ha='center') plt.text(q3 + 5, 1.1, 'Q3', ha='left') plt.text(150, 1.3, 'Outlier', ha='center') plt.title('Boxplot of Student Grades') plt.xlabel('Grade') plt.grid(True, alpha=0.3) plt.savefig('boxplot.png') plt.show()    Skewness : Skewness measures asymmetry. Positive skew (right tail longer) is common in incomes; negative skew (left tail longer) in exam scores. For incomes , skewness is positive. See .   Histograms comparing normal and right-skewed distributions.   Normal vs. skewed distributions.     Code for skewness comparison (new)  import matplotlib.pyplot as plt import numpy as np from scipy.stats import norm, skewnorm np.random.seed(42) normal_data = np.random.normal(50, 10, 1000) skewed_data = np.random.exponential(20000, 1000) fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) ax1.hist(normal_data, bins=30, edgecolor='black', alpha=0.7) ax1.set_title('Normal Distribution') ax1.set_xlabel('Value') ax1.set_ylabel('Frequency') ax2.hist(skewed_data, bins=30, edgecolor='black', alpha=0.7) ax2.set_title('Right-Skewed Distribution') ax2.set_xlabel('Value') ax2.set_ylabel('Frequency') plt.tight_layout() plt.savefig('.\/images\/essential-probability-and-statistics\/skewness-comparison.png') plt.show()     Numerical Summary  Computing descriptive statistics numerically is efficient with Python. Below is a program to calculate mean, median, mode, variance, standard deviation, quartiles, and skewness for a dataset. This program will print out the following results   Mean: 83.33   Median: 80.00   Mode: 75   Population Variance: 88.89   Population Std Dev: 9.43   Sample Variance: 100.00   Sample Std Dev: 10.00   Q1: 75.00, Q3: 90.00   Skewness: 0.39     Code for computing descriptive statistics (new)  import numpy as np from scipy import stats data = [70, 75, 75, 80, 80, 85, 90, 95, 100] mean = np.mean(data) median = np.median(data) mode = stats.mode(data, keepdims=True)[0][0] pop_var = np.var(data) pop_std = np.std(data) sam_var = np.var(data, ddof=1) sam_std = np.std(data, ddof=1) q1, q3 = np.percentile(data, [25, 75]) skew = stats.skew(data) print(f\"Mean: {mean:.2f}\") print(f\"Median: {median:.2f}\") print(f\"Mode: {mode}\") print(f\"Population Variance: {pop_var:.2f}\") print(f\"Population Std Dev: {pop_std:.2f}\") print(f\"Sample Variance: {sam_var:.2f}\") print(f\"Sample Std Dev: {sam_std:.2f}\") print(f\"Q1: {q1:.2f}, Q3: {q3:.2f}\") print(f\"Skewness: {skew:.2f}\")    "
},
{
  "id": "fig-central-tendency",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#fig-central-tendency",
  "type": "Figure",
  "number": "1.1.1",
  "title": "",
  "body": " Density plot of incomes with mean, median, and no mode.   Density plot showing central tendency measures.   "
},
{
  "id": "fig-variance-comparison",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#fig-variance-comparison",
  "type": "Figure",
  "number": "1.1.2",
  "title": "",
  "body": " Comparing low and high variance datasets.   Comparison of low and high variance data.   "
},
{
  "id": "tab-Histogram-Table",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#tab-Histogram-Table",
  "type": "Table",
  "number": "1.1.3",
  "title": "Histogram of Grades",
  "body": " Histogram of Grades    Bin  Range  Data  Count  Frequency    1    3  0.333    2    3  0.333    3    2  0.222    4    1  0.111    "
},
{
  "id": "fig-descriptive-statistics-histogram",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#fig-descriptive-statistics-histogram",
  "type": "Figure",
  "number": "1.1.4",
  "title": "",
  "body": " Histogram of grades with mean and median.   Histogram with mean and median lines.   "
},
{
  "id": "fig-descriptive-statistics-boxplots",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#fig-descriptive-statistics-boxplots",
  "type": "Figure",
  "number": "1.1.5",
  "title": "",
  "body": " Boxplot of grades with annotated quartiles and outlier.   Boxplot with one outlier.   "
},
{
  "id": "fig-skewness-comparison",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#fig-skewness-comparison",
  "type": "Figure",
  "number": "1.1.6",
  "title": "",
  "body": " Histograms comparing normal and right-skewed distributions.   Normal vs. skewed distributions.   "
},
{
  "id": "sec-useful-descriptive-statistics-tools",
  "level": "1",
  "url": "sec-useful-descriptive-statistics-tools.html",
  "type": "Section",
  "number": "1.2",
  "title": "Useful Tools for Descriptive Statistics and Exploratory Data Analysis",
  "body": " Useful Tools for Descriptive Statistics and Exploratory Data Analysis   Exploratory Data Analysis (EDA) is a critical step in understanding your data before applying advanced techniques like machine learning. It involves summarizing the main characteristics of a dataset, often using visual methods, to uncover patterns, spot anomalies, test hypotheses, and check assumptions. In this section, we focus on Python-based tools that enable efficient and effective data analysis, tailored for machine learning workflows. While languages like R are powerful for statistics, we emphasize Python due to its widespread use in data science and machine learning communities. Key tools include NumPy for numerical computations, Pandas for data manipulation, and Matplotlib\/Seaborn for visualization. These libraries integrate seamlessly, allowing you to load, clean, analyze, and visualize data in a streamlined manner.  A typical EDA workflow includes: loading data, inspecting its structure, handling missing values, computing summary statistics, exploring distributions, and visualizing relationships. Using Jupyter notebooks ensures reproducibility and documentation of your analysis.    NumPy: Foundation for Numerical Operations  NumPy (Numerical Python) is the backbone of scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with mathematical functions for efficient operations. It’s foundational for Pandas and essential for computing descriptive statistics.   Key Features:   Efficient array creation and manipulation (e.g., reshaping, slicing).  Statistical functions: mean, median, standard deviation, variance, percentiles.  Integration with Pandas DataFrames and visualization libraries.     Example: Compute descriptive statistics for exam scores.   Computing statistics with NumPy  import numpy as np # Sample data: exam scores scores = np.array([85, 92, 78, 95, 88, 76, 90, 82]) # Basic statistics mean = np.mean(scores) median = np.median(scores) std_dev = np.std(scores, ddof=1) # Sample std dev variance = np.var(scores, ddof=1) # Sample variance percentiles = np.percentile(scores, [25, 50, 75]) print(f\"Mean: {mean:.2f}\") print(f\"Median: {median:.2f}\") print(f\"Sample Standard Deviation: {std_dev:.2f}\") print(f\"Sample Variance: {variance:.2f}\") print(f\"25th, 50th, 75th Percentiles: {percentiles}\")   Output: Mean: 85.75 Median: 86.50 Sample Standard Deviation: 6.82 Sample Variance: 46.50 25th, 50th, 75th Percentiles: [80.5 86.5 91.25]    NumPy is fast for numerical operations but lacks labeled data handling, which is where Pandas excels.     Pandas: Data Manipulation and Analysis   Pandas is a powerful, flexible library for data manipulation and analysis, built on NumPy. Its core data structures are:  Series : A one-dimensional labeled array for sequences of data.  DataFrame : A two-dimensional labeled table, similar to a spreadsheet or SQL table, ideal for tabular data.    Pandas is designed for cleaning, transforming, analyzing, and visualizing data. It supports multiple file formats (CSV, Excel, JSON, SQL) and integrates with NumPy, Matplotlib, Seaborn, and Scikit-learn, making it a cornerstone for EDA in machine learning.    Why Use Pandas?   Handles structured data efficiently (e.g., tabular data).  Supports data cleaning (missing values, duplicates, outliers).  Enables grouping, aggregation, and statistical summaries.  Scales to large datasets with optimized performance.     EDA Workflow with Pandas:   Load data ( pd.read_csv() , pd.read_excel() ).  Inspect structure ( head() , info() , describe() ).  Clean data (handle missing values, remove duplicates).  Compute statistics and explore distributions.  Visualize (integrate with Matplotlib\/Seaborn).     Example: Analyzing Student Data Let’s use a realistic dataset of student scores, including a missing value, loaded from a CSV file.   Creating and loading sample student data  import pandas as pd # Create sample CSV data (in practice, load from disk) data = \"\"\"Name,Age,Score,Passed Alice,25,85.5,True Bob,30,90.0,True Carol,27,88.0,True Dave,22,76.5,False Eve,28,,True\"\"\" with open('students.csv', 'w') as f: f.write(data) # Load data df = pd.read_csv('students.csv') print(df)   Output as a table:   Student DataFrame    Name  Age  Score  Passed    Alice  25  85.5  True    Bob  30  90.0  True    Carol  27  88.0  True    Dave  22  76.5  False    Eve  28  NaN  True     Inspect the data using common Pandas methods:   Inspecting DataFrame  # Inspect data print(\"First 3 rows:\") print(df.head(3)) print(\"\\nLast 2 rows:\") print(df.tail(2)) print(\"\\nShape:\", df.shape) # (5, 4) print(\"\\nColumns:\", df.columns.tolist()) print(\"\\nInfo:\") print(df.info()) print(\"\\nDescriptive Statistics:\") print(df.describe())   Output of df.describe() :   Descriptive Statistics from df.describe()     Age  Score    count  5.000000  4.000000    mean  26.400000  85.000000    std  3.209361  5.958188    min  22.000000  76.500000    25%  24.250000  83.250000    50%  26.000000  86.750000    75%  27.750000  88.500000    max  30.000000  90.000000     Clean and transform the data (e.g., handle missing values, filter, add columns, group, sort):   Data cleaning and transformation  # Handle missing values print(\"Missing values:\\n\", df.isnull()) df['Score'] = df['Score'].fillna(df['Score'].mean()) # Fill NaN with mean # Filter rows high_scorers = df[df['Score'] > 85] print(\"\\nHigh scorers:\\n\", high_scorers) # Add new column df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C') print(\"\\nDataFrame with Grade:\\n\", df) # Group and aggregate grouped = df.groupby('Passed')['Score'].agg(['mean', 'count']) print(\"\\nGrouped by Passed:\\n\", grouped) # Sort by Score df_sorted = df.sort_values(by='Score', ascending=False) print(\"\\nSorted by Score:\\n\", df_sorted) # Chain operations result = df[df['Age'] > 25][['Name', 'Score']].sort_values(by='Score') print(\"\\nChained operations (Age > 25, select columns, sort):\\n\", result)   Visualize the score distribution:   Histogram of student scores using Pandas and Matplotlib.   Histogram from Pandas DataFrame.     Generating histogram from Pandas  import matplotlib.pyplot as plt import pandas as pd # Assuming df from previous code df['Score'].hist(bins=5, edgecolor='black', alpha=0.7) plt.xlabel('Score') plt.ylabel('Frequency') plt.title('Distribution of Student Scores') plt.grid(True, alpha=0.3) plt.savefig('pandas-histogram.png', dpi=300) plt.show()   For further learning, explore Python for Data Analysis by Wes McKinney (free online) and Kaggle’s Pandas course .    Visualization with Matplotlib and Seaborn  Visualization is a cornerstone of EDA, making patterns and relationships in data intuitive. Matplotlib provides customizable, low-level plotting, while Seaborn, built on Matplotlib, offers high-level statistical visualizations with attractive defaults.   Matplotlib Key Features:   Flexible plots: histograms, boxplots, scatter plots, line plots.  Customizable axes, labels, and styles.  Integration with Pandas for direct plotting.     Seaborn Advantages:   Statistical plots: histplot with KDE, boxplot, pairplot for correlations.  Attractive themes and color palettes.  Simplified syntax for complex visualizations.     Example: Visualizing Student Data Using the student DataFrame, create a histogram and boxplot with Matplotlib, and a histplot with KDE and pairplot with Seaborn.   Matplotlib histogram and boxplot of student scores.   Matplotlib plots from Pandas.     Matplotlib histogram and boxplot  import matplotlib.pyplot as plt import pandas as pd # Assuming df from previous code fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) df['Score'].hist(bins=5, ax=ax1, edgecolor='black', alpha=0.7) ax1.set_title('Histogram of Scores') ax1.set_xlabel('Score') ax1.set_ylabel('Frequency') ax1.grid(True, alpha=0.3) df.boxplot(column='Score', ax=ax2) ax2.set_title('Boxplot of Scores') ax2.set_ylabel('Score') ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('matplotlib-plots.png', dpi=300) plt.show()    Seaborn histplot with KDE and pairplot of student data.   Seaborn plots from Pandas.     Seaborn histplot and pairplot  # --- NEW CODE FOR SEABORN PLOTS --- import seaborn as sns import pandas as pd # Assuming df from previous code sns.set_theme(style=\"whitegrid\") fig, ax = plt.subplots(figsize=(8, 5)) sns.histplot(df['Score'], kde=True, ax=ax) ax.set_title('Histogram with KDE of Scores') ax.set_xlabel('Score') ax.set_ylabel('Count') plt.savefig('seaborn-histplot.png', dpi=300) plt.show() # Pairplot for relationships sns.pairplot(df[['Age', 'Score']], diag_kind='kde') plt.savefig('seaborn-pairplot.png', dpi=300) plt.show() # --- END NEW CODE ---    Real-World Example: Load a larger dataset (e.g., from Kaggle) and visualize distributions and correlations.   EDA with a larger dataset  import pandas as pd import seaborn as sns import matplotlib.pyplot as plt # Sample larger dataset (simulated for book) np.random.seed(42) n = 100 data = pd.DataFrame({ 'Age': np.random.normal(25, 5, n), 'Score': np.random.normal(85, 10, n), 'Hours_Studied': np.random.normal(20, 5, n) }) data['Score'] = data['Score'].clip(0, 100) # Ensure valid scores # Basic EDA print(data.describe()) print(\"\\nMissing values:\\n\", data.isnull().sum()) # Correlation matrix print(\"\\nCorrelation matrix:\\n\", data.corr()) # Visualization plt.figure(figsize=(10, 4)) plt.subplot(1, 2, 1) sns.histplot(data['Score'], kde=True) plt.title('Distribution of Scores') plt.subplot(1, 2, 2) sns.scatterplot(x='Hours_Studied', y='Score', data=data) plt.title('Score vs. Hours Studied') plt.tight_layout() plt.savefig('.\/images\/essential-probability-and-statistics\/eda-large-dataset.png', dpi=300) plt.show() # Pairplot sns.pairplot(data, diag_kind='kde') plt.savefig('.\/images\/essential-probability-and-statistics\/eda-pairplot.png', dpi=300) plt.show()    EDA on a larger dataset: histogram and scatter plot.   EDA visualizations for larger dataset.     Pairplot showing relationships in larger dataset.   Pairplot for larger dataset.      NumPy, Pandas, Matplotlib, and Seaborn form a powerful toolkit for EDA. Start with NumPy for numerical operations, use Pandas for data manipulation and cleaning, and leverage Matplotlib\/Seaborn for insightful visualizations. Practice with real datasets (e.g., from Kaggle) in Jupyter notebooks to build skills. For advanced machine learning pipelines, you can explore TensorFlow’s Data API later, but mastering these foundational tools is key for beginners. Resources like Python for Data Analysis and Kaggle’s Pandas course offer hands-on learning.   "
},
{
  "id": "tab-students-dataframe",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#tab-students-dataframe",
  "type": "Table",
  "number": "1.2.1",
  "title": "Student DataFrame",
  "body": " Student DataFrame    Name  Age  Score  Passed    Alice  25  85.5  True    Bob  30  90.0  True    Carol  27  88.0  True    Dave  22  76.5  False    Eve  28  NaN  True    "
},
{
  "id": "tab-students-describe",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#tab-students-describe",
  "type": "Table",
  "number": "1.2.2",
  "title": "Descriptive Statistics from df.describe()",
  "body": " Descriptive Statistics from df.describe()     Age  Score    count  5.000000  4.000000    mean  26.400000  85.000000    std  3.209361  5.958188    min  22.000000  76.500000    25%  24.250000  83.250000    50%  26.000000  86.750000    75%  27.750000  88.500000    max  30.000000  90.000000    "
},
{
  "id": "fig-pandas-histogram",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-pandas-histogram",
  "type": "Figure",
  "number": "1.2.3",
  "title": "",
  "body": " Histogram of student scores using Pandas and Matplotlib.   Histogram from Pandas DataFrame.   "
},
{
  "id": "fig-matplotlib-plots",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-matplotlib-plots",
  "type": "Figure",
  "number": "1.2.4",
  "title": "",
  "body": " Matplotlib histogram and boxplot of student scores.   Matplotlib plots from Pandas.   "
},
{
  "id": "fig-seaborn-plots",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-seaborn-plots",
  "type": "Figure",
  "number": "1.2.5",
  "title": "",
  "body": " Seaborn histplot with KDE and pairplot of student data.   Seaborn plots from Pandas.   "
},
{
  "id": "fig-eda-large-dataset",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-eda-large-dataset",
  "type": "Figure",
  "number": "1.2.6",
  "title": "",
  "body": " EDA on a larger dataset: histogram and scatter plot.   EDA visualizations for larger dataset.   "
},
{
  "id": "fig-eda-pairplot",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-eda-pairplot",
  "type": "Figure",
  "number": "1.2.7",
  "title": "",
  "body": " Pairplot showing relationships in larger dataset.   Pairplot for larger dataset.   "
},
{
  "id": "sec-data-types-for-machine-learning",
  "level": "1",
  "url": "sec-data-types-for-machine-learning.html",
  "type": "Section",
  "number": "1.3",
  "title": "Data Types for Machine Learning",
  "body": " Data Types for Machine Learning   Understanding the types of data you encounter in machine learning is crucial for effective preprocessing and model building. Data types determine how you clean, transform, and encode data for algorithms, which typically require numerical inputs. This section covers three key distinctions: structured vs. unstructured data, sequence vs. non-sequence data, and numerical vs. categorical (including ordinal) data. We’ll use Pandas to demonstrate handling these data types, focusing on practical examples relevant to machine learning workflows.  Machine learning models often require numerical representations, so preprocessing steps like encoding categorical data, scaling numerical data, or embedding unstructured data are essential. A sample dataset of student records will illustrate these concepts throughout.    Structured vs. Unstructured Data   Structured Data : Organized in a tabular format (rows and columns), like a spreadsheet or database table. Each row represents a data point (e.g., a student), and each column represents a feature (e.g., age, score). Structured data is common in machine learning for tasks like classification or regression.   Unstructured Data : Lacks a predefined format, such as text, images, audio, or video. For example, student essays or profile pictures are unstructured. Machine learning often requires transforming unstructured data into structured formats (e.g., word embeddings for text).   Example: A student dataset (structured) vs. student essays (unstructured).   Structured student data in Pandas  import pandas as pd # Structured data: student records data = { 'Name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'], 'Gender': ['F', 'M', 'F', 'M', 'F'], 'Education_Level': ['Bachelor', 'Master', 'High School', 'PhD', 'Bachelor'], 'Age': [25, 30, 27, 22, 28], 'Score': [85.5, 90.0, 88.0, 76.5, None] } df = pd.DataFrame(data) print(df) # Unstructured data: student essays (example text) essays = [\"Alice's essay on AI ethics...\", \"Bob's essay on machine learning...\"] print(\"\\nSample essay:\", essays[0][:20])   Output (structured data as table):   Student DataFrame    Name  Gender  Education_Level  Age  Score    Alice  F  Bachelor  25  85.5    Bob  M  Master  30  90.0    Carol  F  High School  27  88.0    Dave  M  PhD  22  76.5    Eve  F  Bachelor  28  NaN     Unstructured data like essays requires preprocessing (e.g., tokenization, embeddings) for machine learning, covered later in this section.    Sequence vs. Non-Sequence Data   Sequence Data : Data where the order of data points matters, such as time series, text, or speech. For example, a student’s daily study hours over a month is sequence data, where temporal order affects analysis.   Non-Sequence Data : Data where order is irrelevant, such as static student records. Most tabular datasets in machine learning are non-sequence.   Example: A time series of a student’s study hours.   Sequence data example  import pandas as pd import numpy as np import matplotlib.pyplot as plt # Sequence data: daily study hours dates = pd.date_range('2025-01-01', periods=10, freq='D') study_hours = pd.Series([2, 3, 0, 4, 2, 5, 1, 3, 2, 4], index=dates, name='Study_Hours') print(study_hours) # Plot plt.figure(figsize=(8, 4)) study_hours.plot() plt.xlabel('Date') plt.ylabel('Study Hours') plt.title('Daily Study Hours (Sequence Data)') plt.grid(True, alpha=0.3) plt.savefig('time-series.png', dpi=300) plt.show()    Time series of daily study hours.   Time series plot of study hours.    Sequence data requires models like RNNs or LSTMs, which account for temporal dependencies, unlike non-sequence data used in standard regression or classification.    Numerical, Categorical, and Ordinal Data  Machine learning datasets often combine numerical, categorical, and ordinal data, each requiring specific preprocessing.     Numerical Data : Represents measurable quantities, either discrete (countable, e.g., number of courses taken) or continuous (infinite precision, e.g., test scores, age). Numerical data often needs scaling (e.g., standardization) for machine learning algorithms like SVMs or neural networks.  Example: In the student dataset, Age and Score are numerical. Let’s visualize the score distribution.   Histogram of student scores (numerical data).   Histogram of numerical data.     Histogram of numerical data  import pandas as pd import matplotlib.pyplot as plt # Assuming df from previous example df['Score'].hist(bins=5, edgecolor='black', alpha=0.7) plt.xlabel('Score') plt.ylabel('Frequency') plt.title('Distribution of Student Scores') plt.grid(True, alpha=0.3) plt.savefig('numerical-histogram.png', dpi=300) plt.show()   Preprocessing: Scale numerical data to ensure equal contribution to models.   Scaling numerical data  from sklearn.preprocessing import StandardScaler # Scale Age and Score scaler = StandardScaler() df[['Age', 'Score']] = scaler.fit_transform(df[['Age', 'Score']].fillna(df['Score'].mean())) print(df[['Age', 'Score']])      Categorical Data : Represents discrete, unordered labels (e.g., Gender: {F, M}). Machine learning requires numerical encoding, typically via one-hot encoding.   One-Hot Encoding : For a categorical variable with categories, each category is represented by a -dimensional binary vector. For Gender (F, M), :   Example: Encode Gender in the student dataset.   One-hot encoding categorical data  # One-hot encoding df_encoded = pd.get_dummies(df, columns=['Gender'], prefix='Gender') print(df_encoded)   Visualize Gender counts:   Bar plot of Gender counts (categorical data).   Bar plot of categorical data.     Bar plot of categorical data  import pandas as pd import matplotlib.pyplot as plt # Assuming df from previous example df['Gender'].value_counts().plot(kind='bar', edgecolor='black', alpha=0.7) plt.xlabel('Gender') plt.ylabel('Count') plt.title('Distribution of Gender') plt.grid(True, alpha=0.3) plt.savefig('categorical-bar.png', dpi=300) plt.show()   For high-cardinality categorical data (e.g., thousands of categories like city names), one-hot encoding creates too many features. Instead, use embeddings (e.g., word2vec for text), discussed below.     Ordinal Data : Categorical data with a defined order (e.g., Education Level: High School ≤ Bachelor ≤ Master ≤ PhD). The order matters, but differences are not necessarily quantitative.  Example: Encode Education Level as an ordered category.   Encoding ordinal data  from pandas.api.types import CategoricalDtype # Define ordered category cat_type = CategoricalDtype(categories=['High School', 'Bachelor', 'Master', 'PhD'], ordered=True) df['Education_Level'] = df['Education_Level'].astype(cat_type) print(df['Education_Level']) # Integer encoding for ML education_order = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4} df['Education_Level_Code'] = df['Education_Level'].map(education_order) print(df[['Education_Level', 'Education_Level_Code']])   Visualize Education Level:   Ordered bar plot of Education Level.   Ordered bar plot of ordinal data.     Ordered bar plot of ordinal data  import pandas as pd import matplotlib.pyplot as plt # Assuming df from previous example df['Education_Level'].value_counts().loc[['High School', 'Bachelor', 'Master', 'PhD']].plot(kind='bar', edgecolor='black', alpha=0.7) plt.xlabel('Education Level') plt.ylabel('Count') plt.title('Distribution of Education Level (Ordered)') plt.grid(True, alpha=0.3) plt.savefig('ordinal-bar.png', dpi=300) plt.show()    Safe Ordinal Encoding : Use integer encoding for tree-based models (e.g., Random Forest), which respect order without assuming equal intervals. Avoid one-hot encoding, as it discards order.     Combined Example: Preprocess the student dataset and visualize relationships.   Preprocessing mixed data types  # --- MIXED DATA PREPROCESSING --- import pandas as pd from sklearn.preprocessing import StandardScaler # Assuming df from previous example # Handle missing values df['Score'] = df['Score'].fillna(df['Score'].mean()) # One-hot encode Gender df = pd.get_dummies(df, columns=['Gender'], prefix='Gender') # Integer encode Education_Level education_order = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4} df['Education_Level_Code'] = df['Education_Level'].map(education_order) # Scale numerical columns scaler = StandardScaler() df[['Age', 'Score']] = scaler.fit_transform(df[['Age', 'Score']]) print(df)   Visualize relationships:   Pairplot of numerical and encoded ordinal data.   Pairplot of mixed data types.     Pairplot of mixed data  import seaborn as sns import pandas as pd # Assuming df from previous example sns.set_theme(style=\"whitegrid\") sns.pairplot(df[['Age', 'Score', 'Education_Level_Code']], diag_kind='kde') plt.savefig('pairplot.png', dpi=300) plt.show()     Handling Unstructured Text Data  Text data, a common form of unstructured data, requires special preprocessing. For example, student essays can be converted to numerical representations using word embeddings (e.g., word2vec, BERT) for high-cardinality categorical data or text analysis.   Example: Tokenize and embed a sample essay.   Basic text preprocessing  # --- TEXT PREPROCESSING --- from sklearn.feature_extraction.text import CountVectorizer # Sample essays essays = [\"AI ethics is critical for fairness\", \"Machine learning improves predictions\"] vectorizer = CountVectorizer() X = vectorizer.fit_transform(essays) print(\"Feature names:\", vectorizer.get_feature_names_out()) print(\"Bag-of-words matrix:\\n\", X.toarray())   For advanced text processing, use libraries like Hugging Face’s Transformers to generate embeddings for machine learning models.    Machine learning requires careful handling of data types: structured data for tabular analysis, unstructured data like text for specialized preprocessing, sequence data for temporal models, and numerical\/categorical\/ordinal data for appropriate encoding. Use Pandas for structured data, scale numerical features, encode categorical and ordinal data thoughtfully, and preprocess unstructured\/sequence data with libraries like scikit-learn or Hugging Face. Practice with datasets from Kaggle to master these techniques.   "
},
{
  "id": "tab-student-dataframe",
  "level": "2",
  "url": "sec-data-types-for-machine-learning.html#tab-student-dataframe",
  "type": "Table",
  "number": "1.3.1",
  "title": "Student DataFrame",
  "body": " Student DataFrame    Name  Gender  Education_Level  Age  Score    Alice  F  Bachelor  25  85.5    Bob  M  Master  30  90.0    Carol  F  High School  27  88.0    Dave  M  PhD  22  76.5    Eve  F  Bachelor  28  NaN    "
},
{
  "id": "fig-time-series",
  "level": "2",
  "url": "sec-data-types-for-machine-learning.html#fig-time-series",
  "type": "Figure",
  "number": "1.3.2",
  "title": "",
  "body": " Time series of daily study hours.   Time series plot of study hours.   "
},
{
  "id": "fig-numerical-histogram",
  "level": "2",
  "url": "sec-data-types-for-machine-learning.html#fig-numerical-histogram",
  "type": "Figure",
  "number": "1.3.3",
  "title": "",
  "body": " Histogram of student scores (numerical data).   Histogram of numerical data.   "
},
{
  "id": "fig-categorical-bar",
  "level": "2",
  "url": "sec-data-types-for-machine-learning.html#fig-categorical-bar",
  "type": "Figure",
  "number": "1.3.4",
  "title": "",
  "body": " Bar plot of Gender counts (categorical data).   Bar plot of categorical data.   "
},
{
  "id": "fig-ordinal-bar",
  "level": "2",
  "url": "sec-data-types-for-machine-learning.html#fig-ordinal-bar",
  "type": "Figure",
  "number": "1.3.5",
  "title": "",
  "body": " Ordered bar plot of Education Level.   Ordered bar plot of ordinal data.   "
},
{
  "id": "fig-pairplot",
  "level": "2",
  "url": "sec-data-types-for-machine-learning.html#fig-pairplot",
  "type": "Figure",
  "number": "1.3.6",
  "title": "",
  "body": " Pairplot of numerical and encoded ordinal data.   Pairplot of mixed data types.   "
},
{
  "id": "sec-Basic-Probability",
  "level": "1",
  "url": "sec-Basic-Probability.html",
  "type": "Section",
  "number": "1.4",
  "title": "Basic Probability for Machine Learning",
  "body": " Basic Probability for Machine Learning   Probability is the backbone of machine learning, helping us model uncertainty in data, predictions, and outcomes. In machine learning, probability underpins tasks like classification (e.g., predicting labels), evaluating model confidence, and handling noisy data. This section introduces probability concepts such as sample spaces, events, and axioms—and connects them to practical machine learning applications using Python. We will use the student dataset from to illustrate ideas.  An event is a specific outcome or set of outcomes from an experiment, represented as a set. For a coin toss, \"heads\" is , \"tails\" is , and \"heads or tails\" is . Each trial answers whether an event occurred (yes\/no). For a die roll yielding 2, events like or occur if they include 2. Sets allow combining events via union ( ) or intersection ( ), such as .    Axiomatic View of Probability  In 1933, Andrey Kolmogorov formalized probability with three axioms, providing a mathematical framework. Think of these as rules that ensure probabilities make sense, like ensuring a weather forecast never predicts negative rain or more than 100% chance.   Sample Space : The set of all possible outcomes. For a six-sided die, . For a student passing an exam, .   Event Space : All possible subsets of , including the empty set (impossible event) and (event certain to happen). For a coin toss ( ), . With outcomes, has events.   Probability Measure : Assigns a number to each event , representing its likelihood. For example, for a fair die, .  The probability space is the triplet . Kolmogorov’s axioms are:  Non-negativity : for all .  Normalization : , ensuring total certainty for event .  Additivity : For disjoint events ( ), .    Derived results:   The union formula accounts for overlap, as shown in . For a die, if (odd numbers), , then .   Venn diagram illustrating events (odd numbers) and for 1,000 simulated rolls of a fair six-sided die. The left region (181) counts rolls of 1 (in only), the right region (326) counts rolls of 6 (in only), and the overlap (155) counts rolls of 3 or 5 (in ). These counts estimate probabilities, with theoretical values , , , and . Deviations (e.g., 326 vs. expected 167 for only) reflect random variation. This visualization supports the union formula , used in machine learning for feature probability calculations.   Venn diagram of intersecting events.     Simulating die roll for union probability  # --- DIE ROLL VENN DIAGRAM --- import numpy as np from matplotlib_venn import venn2 import matplotlib.pyplot as plt np.random.seed(42) n_trials = 1000 rolls = np.random.randint(1, 7, n_trials) # Events e1 = np.isin(rolls, [1, 3, 5]) # Odd numbers e2 = np.isin(rolls, [3, 5, 6]) # 3,5,6 e1_only = np.sum(e1 \\amp; ~e2) e2_only = np.sum(e2 \\amp; ~e1) both = np.sum(e1 \\amp; e2) # Venn diagram plt.figure(figsize=(6, 4)) venn2(subsets=(e1_only, e2_only, both), set_labels=('E1 (Odd)', 'E2 (3,5,6)')) plt.title('Venn Diagram of Die Roll Events') plt.savefig('venn-diagram-E1-E2.png', dpi=300) plt.show() # Probabilities p_e1 = np.mean(e1) p_e2 = np.mean(e2) p_inter = np.mean(e1 \\amp; e2) p_union = np.mean(e1 | e2) print(f\"P(E1): {p_e1:.3f}, P(E2): {p_e2:.3f}, P(E1 ∩ E2): {p_inter:.3f}, P(E1 ∪ E2): {p_union:.3f}\") # --- END CODE ---     Sum and Product Rules for Probability  The sum and product rules are foundational for computing probabilities of combined events, essential for machine learning tasks like feature engineering and Bayesian inference.   Sum Rule : The probability of the union of two events and is given by: This accounts for overlapping events to avoid double-counting. In machine learning, the sum rule is used to compute marginal probabilities from joint distributions.   Product Rule : The joint probability of two events and is: where is the conditional probability of given . This rule is key for factoring joint probabilities in models like Naive Bayes.  Example: For a fair die, let (numbers ≤ 3) and . The sum rule gives , as . For the product rule, consider the student dataset from : the probability of passing and studying >20 hours is .   Sum and product rules with student data  # --- SUM AND PRODUCT RULES --- import pandas as pd import numpy as np import seaborn as sns import matplotlib.pyplot as plt # Student data np.random.seed(42) data = pd.DataFrame({ 'Hours_Studied': np.random.normal(20, 5, 100).clip(0, 40), 'Passed': np.random.binomial(1, 0.7, 100) }) data['High_Study'] = data['Hours_Studied'] > 20 # Compute probabilities p_pass = np.mean(data['Passed']) p_high_study = np.mean(data['High_Study']) p_pass_and_high = np.mean(data['Passed'] \\amp; data['High_Study']) p_pass_or_high = p_pass + p_high_study - p_pass_and_high # Sum rule p_pass_given_high = p_pass_and_high \/ p_high_study # For product rule print(f\"P(Pass): {p_pass:.3f}, P(High Study): {p_high_study:.3f}\") print(f\"P(Pass ∩ High Study): {p_pass_and_high:.3f}\") print(f\"P(Pass ∪ High Study): {p_pass_or_high:.3f}\") print(f\"Product Rule: P(Pass ∩ High Study) = P(Pass | High Study) * P(High Study) = {p_pass_given_high:.3f} * {p_high_study:.3f} = {p_pass_given_high * p_high_study:.3f}\") # Heatmap of joint probabilities joint_table = pd.crosstab(data['Passed'], data['High_Study'], normalize='all') sns.heatmap(joint_table, annot=True, cmap='Blues', fmt='.3f') plt.xlabel('High Study Hours (>20)') plt.ylabel('Passed') plt.title('Joint Probability of Pass and High Study Hours') plt.savefig('joint-probability-heatmap.png', dpi=300) plt.show() # --- END CODE ---    Heatmap displaying the joint probability distribution of passing an exam and studying more than 20 hours for 100 students, based on synthetic data (Hours_Studied: normal, mean 20, std 5; Passed: Bernoulli, p=0.7). Each cell shows the probability , with darker shades indicating higher probabilities. Marginal probabilities and are derived by summing rows or columns, illustrating the sum rule. The product rule is verified as . This visualization is relevant to machine learning for estimating joint feature probabilities in classification models.   Heatmap of joint probabilities for pass and study hours.      Conditional Probability and Independence   Conditional Probability : The probability of an event given that has occurred, denoted , where . For example, the probability a student passes given they studied over 20 hours.   Independence : Events and are independent if , meaning one event doesn’t affect the other.  Example: Using the student dataset from , estimate the probability of passing given high study hours.   Conditional probability with student data  # --- CONDITIONAL PROBABILITY --- import pandas as pd import numpy as np # Student data np.random.seed(42) data = pd.DataFrame({ 'Hours_Studied': np.random.normal(20, 5, 100).clip(0, 40), 'Passed': np.random.binomial(1, 0.7, 100) }) data['High_Study'] = data['Hours_Studied'] > 20 # Conditional probability p_pass = np.mean(data['Passed']) p_high_study = np.mean(data['High_Study']) p_pass_and_high = np.mean(data['Passed'] \\amp; data['High_Study']) p_pass_given_high = p_pass_and_high \/ p_high_study print(f\"P(Pass | High Study): {p_pass_given_high:.3f}\") # Bar plot counts = data.groupby(['High_Study', 'Passed']).size().unstack() counts.plot(kind='bar', stacked=True) plt.xlabel('High Study Hours (>20)') plt.ylabel('Count') plt.title('Pass\/Fail by Study Hours') plt.legend(['Fail', 'Pass']) plt.savefig('.\/images\/essential-probability-and-statistics\/conditional-bar.png', dpi=300) plt.show() # --- END CODE ---    Stacked bar plot showing the distribution of pass\/fail outcomes for 100 students, based on whether they studied more than 20 hours (High_Study = True) or not. The dataset is synthetic, with Hours_Studied drawn from a normal distribution (mean 20, std 5) and Passed from a Bernoulli distribution (p=0.7). The plot illustrates conditional probability , showing a higher proportion of passes among students with high study hours. This visualization is relevant to machine learning for feature analysis in classification tasks, such as predicting student success based on study habits.   Bar plot of conditional probability.      Probability Distributions  Probability distributions describe how probabilities are distributed over outcomes. In machine learning, distributions model data or predictions.   Bernoulli Distribution : Models a binary outcome (e.g., pass\/fail) with probability . For passing an exam, , .   Binomial Distribution : Counts successes in independent Bernoulli trials. For 10 students, the number who pass follows a binomial distribution.   Binomial distribution for student passes  # --- BINOMIAL DISTRIBUTION --- import numpy as np import matplotlib.pyplot as plt from scipy.stats import binom n, p = 10, 0.7 # 10 students, P(Pass) = 0.7 k = np.arange(0, 11) pmf = binom.pmf(k, n, p) plt.bar(k, pmf) plt.xlabel('Number of Passes') plt.ylabel('Probability') plt.title('Binomial Distribution (n=10, p=0.7)') plt.grid(True, alpha=0.3) plt.savefig('.\/images\/essential-probability-and-statistics\/binomial-dist.png', dpi=300) plt.show() # --- END CODE ---    Bar plot of the binomial probability mass function (PMF) for the number of students passing an exam out of 10, with a pass probability . Each bar represents the probability of students passing, calculated as . The peak around 7 passes reflects the high likelihood of most students passing given . This distribution is critical in machine learning for modeling binary outcomes, such as predicting the number of successful predictions in a classification task.   Bar plot of binomial distribution.      Three Types of Probabilities  Probability can be approached theoretically, empirically (frequentist), or subjectively (Bayesian).     Theoretical Probability : Uses symmetry. For a fair die, . For even numbers, .     Frequentist Probability : Estimates probability from trial frequencies: .   Frequentist estimation for fair and biased dice  # --- FREQUENTIST SIMULATION --- import numpy as np import matplotlib.pyplot as plt np.random.seed(42) n_trials = 1000 fair_rolls = np.random.randint(1, 7, n_trials) biased_rolls = np.random.choice([1, 2, 3, 4, 5, 6], n_trials, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]) # Cumulative probabilities cum_fair = np.cumsum(fair_rolls == 1) \/ np.arange(1, n_trials + 1) cum_biased = np.cumsum(biased_rolls == 1) \/ np.arange(1, n_trials + 1) plt.plot(cum_fair, label='Fair Die (P=1\/6)') plt.plot(cum_biased, label='Biased Die (P=0.2)') plt.axhline(1\/6, color='red', linestyle='--', label='Theoretical P=1\/6') plt.xlabel('Trials') plt.ylabel('Estimated P(1)') plt.title('Frequentist Estimates: Fair vs. Biased Die') plt.legend() plt.grid(True, alpha=0.3) plt.savefig('frequentist-convergence.png', dpi=300) plt.show() # --- END CODE ---    Plot showing the convergence of frequentist probability estimates for rolling a 1 on a fair die ( ) and a biased die ( ) over 1,000 trials. The fair die’s estimate (blue) fluctuates but approaches 1\/6 (red dashed line), while the biased die’s estimate (orange) converges to 0.2, reflecting the higher probability of rolling a 1. This visualization demonstrates how empirical frequencies approximate true probabilities in large samples, a technique used in machine learning to estimate probabilities from training data.   Convergence plot for frequentist estimates.       Bayesian Probability : Updates prior beliefs with data using Bayes’ theorem: . For die face 1, use a Beta prior, updated to Beta( ).  Example: Estimate for students using the dataset, starting with a Beta(1,1) prior.   Bayesian update for student pass probability  # --- BAYESIAN UPDATE --- import numpy as np import matplotlib.pyplot as plt from scipy.stats import beta import pandas as pd # Student data np.random.seed(42) data = pd.DataFrame({ 'Passed': np.random.binomial(1, 0.7, 10) }) # Prior: Beta(1,1) a, b = 1, 1 n, n1 = len(data), data['Passed'].sum() a_post, b_post = a + n1, b + n - n1 # Plot prior and posterior x = np.linspace(0, 1, 1000) plt.plot(x, beta.pdf(x, a, b), label='Prior Beta(1,1)', color='blue') plt.plot(x, beta.pdf(x, a + 1, b + 1), label='After 1 Pass', color='orange') plt.plot(x, beta.pdf(x, a_post, b_post), label=f'Posterior Beta({a_post},{b_post})', color='green') plt.axvline(n1\/n, color='red', linestyle='--', label='Frequentist Est.') plt.xlabel('P(Pass)') plt.ylabel('Density') plt.title('Bayesian Update for P(Pass)') plt.legend() plt.grid(True, alpha=0.3) plt.savefig('.\/images\/essential-probability-and-statistics\/bayesian-update.png', dpi=300) plt.show() # --- END CODE ---    Plot showing the Bayesian update of the probability of a student passing an exam, starting with a uniform Beta(1,1) prior (blue). After observing one pass (orange) and 10 trials with 3 passes (green, posterior Beta(4,8)), the distribution shifts, with the posterior mean at 4\/12 ≈ 0.333. The frequentist estimate (red dashed line, 3\/10 = 0.3) is shown for comparison. This visualization illustrates how Bayesian methods incorporate prior beliefs and data to refine probability estimates, a technique used in machine learning for probabilistic models and uncertainty quantification.   Bayesian update plot.        Probability provides the foundation for modeling uncertainty in machine learning. Axioms define the rules, while theoretical, frequentist, and Bayesian approaches offer different perspectives. Conditional probability and distributions like binomial are key for models like Naive Bayes. Practice with datasets from and libraries from to apply these concepts. Explore Probability Course for further learning.   "
},
{
  "id": "fig-venn-diagram-E1-E2",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-venn-diagram-E1-E2",
  "type": "Figure",
  "number": "1.4.1",
  "title": "",
  "body": " Venn diagram illustrating events (odd numbers) and for 1,000 simulated rolls of a fair six-sided die. The left region (181) counts rolls of 1 (in only), the right region (326) counts rolls of 6 (in only), and the overlap (155) counts rolls of 3 or 5 (in ). These counts estimate probabilities, with theoretical values , , , and . Deviations (e.g., 326 vs. expected 167 for only) reflect random variation. This visualization supports the union formula , used in machine learning for feature probability calculations.   Venn diagram of intersecting events.   "
},
{
  "id": "fig-joint-probability-heatmap",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-joint-probability-heatmap",
  "type": "Figure",
  "number": "1.4.2",
  "title": "",
  "body": " Heatmap displaying the joint probability distribution of passing an exam and studying more than 20 hours for 100 students, based on synthetic data (Hours_Studied: normal, mean 20, std 5; Passed: Bernoulli, p=0.7). Each cell shows the probability , with darker shades indicating higher probabilities. Marginal probabilities and are derived by summing rows or columns, illustrating the sum rule. The product rule is verified as . This visualization is relevant to machine learning for estimating joint feature probabilities in classification models.   Heatmap of joint probabilities for pass and study hours.   "
},
{
  "id": "fig-conditional-bar",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-conditional-bar",
  "type": "Figure",
  "number": "1.4.3",
  "title": "",
  "body": " Stacked bar plot showing the distribution of pass\/fail outcomes for 100 students, based on whether they studied more than 20 hours (High_Study = True) or not. The dataset is synthetic, with Hours_Studied drawn from a normal distribution (mean 20, std 5) and Passed from a Bernoulli distribution (p=0.7). The plot illustrates conditional probability , showing a higher proportion of passes among students with high study hours. This visualization is relevant to machine learning for feature analysis in classification tasks, such as predicting student success based on study habits.   Bar plot of conditional probability.   "
},
{
  "id": "fig-binomial-dist",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-binomial-dist",
  "type": "Figure",
  "number": "1.4.4",
  "title": "",
  "body": " Bar plot of the binomial probability mass function (PMF) for the number of students passing an exam out of 10, with a pass probability . Each bar represents the probability of students passing, calculated as . The peak around 7 passes reflects the high likelihood of most students passing given . This distribution is critical in machine learning for modeling binary outcomes, such as predicting the number of successful predictions in a classification task.   Bar plot of binomial distribution.   "
},
{
  "id": "fig-frequentist-convergence",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-frequentist-convergence",
  "type": "Figure",
  "number": "1.4.5",
  "title": "",
  "body": " Plot showing the convergence of frequentist probability estimates for rolling a 1 on a fair die ( ) and a biased die ( ) over 1,000 trials. The fair die’s estimate (blue) fluctuates but approaches 1\/6 (red dashed line), while the biased die’s estimate (orange) converges to 0.2, reflecting the higher probability of rolling a 1. This visualization demonstrates how empirical frequencies approximate true probabilities in large samples, a technique used in machine learning to estimate probabilities from training data.   Convergence plot for frequentist estimates.   "
},
{
  "id": "fig-bayesian-update",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-bayesian-update",
  "type": "Figure",
  "number": "1.4.6",
  "title": "",
  "body": " Plot showing the Bayesian update of the probability of a student passing an exam, starting with a uniform Beta(1,1) prior (blue). After observing one pass (orange) and 10 trials with 3 passes (green, posterior Beta(4,8)), the distribution shifts, with the posterior mean at 4\/12 ≈ 0.333. The frequentist estimate (red dashed line, 3\/10 = 0.3) is shown for comparison. This visualization illustrates how Bayesian methods incorporate prior beliefs and data to refine probability estimates, a technique used in machine learning for probabilistic models and uncertainty quantification.   Bayesian update plot.   "
},
{
  "id": "sec-Joint-Conditional-and-Marginal-Probabilities",
  "level": "1",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html",
  "type": "Section",
  "number": "1.5",
  "title": "Joint, Marginal, and Conditional Probabilities",
  "body": " Joint, Marginal, and Conditional Probabilities   This section explores probability concepts for multiple random variables, essential for modeling relationships in data. We cover joint and marginal probabilities, covariance and correlation, conditional probability, Bayes’ rule, and independence, with a focus on rigorous mathematical formulations and practical applications. Random variables are denoted by capital letters (e.g., ), with specific values in lowercase, sometimes with indices (e.g., ). Examples use a patient dataset to illustrate concepts, and Python visualizations reinforce intuition.     Random Variables, Probabilities, and Expectations  A random variable is a quantity whose value is determined by a random experiment, unknown until observed. We denote random variables by capital letters (e.g., ) and their values by lowercase with or without indices (e.g., ).  Random variables are:  Discrete : Takes values in a countable set (e.g., die outcomes ).  Continuous : Takes values in a real interval (e.g., patient temperature).    For a discrete random variable with values ( ), the probability mass function (PMF) is:   The expectation of a function is: The mean is:   The variance measures spread: The standard deviation is .   Die roll expectation and variance  For a fair six-sided die, for . The mean is: The second moment is: The variance is:     Probability mass function (PMF) of a fair six-sided die, showing equal probabilities ( ) for outcomes . This bar chart visualizes the discrete distribution, useful for understanding expected values in games of chance.   Bar chart of die PMF.     PMF of a fair die  # === CODE: PMF of a fair die === import matplotlib.pyplot as plt x = [1, 2, 3, 4, 5, 6] p = [1\/6] * 6 plt.figure() plt.bar(x, p) plt.xlabel(\"Die Outcome\") plt.ylabel(\"Probability\") plt.title(\"PMF of a Fair Die\") plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"die_pmf.png\", dpi=300) plt.show()      Joint and Marginal Probabilities  For two discrete random variables and , the joint probability assigns probabilities to pairs :   The marginal probability of is obtained by summing over all values of : Similarly, .   Patient disease and test results  Consider a dataset of 1,000 patients, with indicating disease status ( : disease, : no disease) and indicating test result ( : positive, : negative). The joint counts are given in . Joint probabilities are:  Marginals are:      Heatmap of joint probabilities for disease status ( ) and test result ( ) from 1,000 patients ( ). Darker shades indicate higher probabilities. Marginal probabilities ( , ) are shown in the margins, illustrating the summation process .   Heatmap of joint probabilities with marginals.     Joint probability heatmap  import numpy as np import matplotlib.pyplot as plt import seaborn as sns # Joint probabilities from patient table joint = np.array([[0.15, 0.05], [0.30, 0.50]]) # Rows: X=D, X=N; Cols: Y=+, Y=- marginal_x = np.sum(joint, axis=1) # P(D), P(N) marginal_y = np.sum(joint, axis=0) # P(+), P(-) # Heatmap with marginals fig, ax = plt.subplots() sns.heatmap(joint, annot=True, fmt=\".3f\", cmap=\"Blues\", cbar=False, xticklabels=[\"$Y=+$\", \"$Y=-$\"], yticklabels=[\"$X=D$\", \"$X=N$\"], annot_kws={\"size\": 16}, ax=ax) for i, m in enumerate(marginal_x): ax.text(2.1, i + 0.5, f\"{m:.3f}\", va=\"center\") for j, m in enumerate(marginal_y): ax.text(j + 0.33, 2.2, f\"{m:.3f}\", ha=\"center\") ax.text(2.2, 2.2, \"1.000\", va=\"center\", ha=\"center\") plt.title(\"Joint Probability Heatmap with Marginals\") plt.tight_layout() plt.savefig(\"joint_heatmap.png\", dpi=300) plt.show()      Covariance and Correlation  The covariance between random variables and measures their joint variability:   The correlation normalizes covariance: where , . Positive correlation means and tend to increase together; negative means they move oppositely. Correlation does not imply causation.   Patient symptom severity and test score  For 1,000 patients, let be symptom severity (normal, mean 5 for disease, 3 for no disease, std 1.5) and be test score (normal, mean 80 for disease, 60 for no disease, std 10). The code below computes covariance and correlation, showing a positive relationship.    Scatter plot of symptom severity vs. test score for 1,000 patients, with regression lines for disease (D) and no disease (N) groups. The computed correlation coefficient is shown, indicating a positive relationship between features, relevant for medical diagnostics.   Scatter plot with regression lines for correlation.     Correlation scatter plot  # === Scatter plot with correlation and regression lines === import numpy as np import pandas as pd import matplotlib.pyplot as plt np.random.seed(42) # Synthetic patient data counts = {\"D+\": 150, \"D-\": 50, \"N+\": 300, \"N-\": 500} data = [] for status, _, count in [(\"D\", 1, 200), (\"N\", 0, 800)]: severity = np.random.normal(5 if status == \"D\" else 3, 1.5, count) score = np.random.normal(80 if status == \"D\" else 60, 10, count) data.extend([[s, t, 1 if status == \"D\" else 0] for s, t in zip(severity, score)]) data = pd.DataFrame(data, columns=[\"Severity\", \"Score\", \"Disease\"]) # Compute correlation corr = data[[\"Severity\", \"Score\"]].corr().iloc[0, 1] # Plot with regression lines plt.figure(figsize=(8, 6)) for d, color, label in [(1, \"red\", \"Disease\"), (0, \"blue\", \"No Disease\")]: subset = data[data[\"Disease\"] == d] plt.scatter(subset[\"Severity\"], subset[\"Score\"], c=color, alpha=0.5, label=label) z = np.polyfit(subset[\"Severity\"], subset[\"Score\"], 1) p = np.poly1d(z) plt.plot(subset[\"Severity\"], p(subset[\"Severity\"]), color=color, linestyle=\"--\") plt.xlabel(\"Symptom Severity\") plt.ylabel(\"Test Score\") plt.title(f\"Correlation: {corr:.3f}\") plt.legend() plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"correlation_examples.png\", dpi=300) plt.show()      Conditional Probability  The conditional probability of given is: By the product rule:   Conditional probabilities form a distribution over for fixed , summing to 1: .   Patient disease and test results  Using the patient dataset ( ), compute all conditional probabilities:  Patient counts (disease vs. test)   Marginal  150 50  300 500  Marginal   Joint probabilities: Marginals: , , , . Conditional probabilities:     Grouped bar chart of conditional probabilities for disease status ( ) given test result ( ) and test result given disease status, based on . Each group shows a probability distribution (summing to 1), illustrating how conditional probabilities slice the joint distribution.   Grouped bar chart of conditional probabilities.     Conditional probability bar chart  # === Grouped bar chart for all conditional probabilities === import matplotlib.pyplot as plt import numpy as np # Joint counts and probabilities counts = {\"D+\": 150, \"D-\": 50, \"N+\": 300, \"N-\": 500} total = 1000 P_D_plus = counts[\"D+\"]\/total P_D_minus = counts[\"D-\"]\/total P_N_plus = counts[\"N+\"]\/total P_N_minus = counts[\"N-\"]\/total P_plus = P_D_plus + P_N_plus P_minus = P_D_minus + P_N_minus P_D = P_D_plus + P_D_minus P_N = P_N_plus + P_N_minus # Conditional probabilities probs = { \"P(D|+)\": P_D_plus \/ P_plus, \"P(N|+)\": P_N_plus \/ P_plus, \"P(D|-)\": P_D_minus \/ P_minus, \"P(N|-)\": P_N_minus \/ P_minus, \"P(+|D)\": P_D_plus \/ P_D, \"P(-|D)\": P_D_minus \/ P_D, \"P(+|N)\": P_N_plus \/ P_N, \"P(-|N)\": P_N_minus \/ P_N } # Grouped bar plot labels = [\"Given $Y=+$\", \"Given $Y=-$\", \"Given $X=D$\", \"Given $X=N$\"] values = [[probs[\"P(D|+)\"], probs[\"P(N|+)\"]], [probs[\"P(D|-)\"], probs[\"P(N|-)\"]], [probs[\"P(+|D)\"], probs[\"P(-|D)\"]], [probs[\"P(+|N)\"], probs[\"P(-|N)\"]]] x = np.arange(len(labels)) width = 0.2 fig, ax = plt.subplots(figsize=(10, 6)) ax.bar(x - width\/2, [v[0] for v in values], width, label=\"First Outcome\", color=\"blue\") ax.bar(x + width\/2, [v[1] for v in values], width, label=\"Second Outcome\", color=\"red\") ax.set_xticks(x) ax.set_xticklabels(labels) ax.set_ylabel(\"Conditional Probability\") ax.set_title(\"Conditional Probabilities from Patient Table\") ax.legend([\"$P(D|·)$, $P(+|·)$\", \"$P(N|·)$, $P(-|·)$\"]) ax.set_ylim(0, 1) ax.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"patient_conditional_bars.png\", dpi=300) plt.show()      Bayes’ Rule  Bayes’ theorem updates probabilities based on new evidence, derived from the product rule: where the denominator is the marginal probability:    Breast cancer screening  For a population with breast cancer prevalence , a mammogram has:  Sensitivity : .  Specificity : , so .  Compute . First, the marginal probability: Then: Thus, a positive test increases the probability of cancer from 1% to 13.91%.    Tree diagram for breast cancer screening, showing prior probabilities ( , ), likelihoods ( , etc.), and joint probabilities leading to . This visualizes Bayes’ rule for medical diagnostics.   Tree diagram for Bayes’ rule in screening.     Bayes’ rule tree diagram  # === NEW CODE: Tree diagram using graphviz === from graphviz import Digraph dot = Digraph(comment=\"Bayes Tree\") dot.attr(rankdir=\"LR\") dot.node(\"A\", \"Population\\nP(C)=0.01\\nP(N)=0.99\", shape=\"box\") dot.node(\"B\", \"Cancer\\nP(C)=0.01\", shape=\"box\") dot.node(\"C\", \"No Cancer\\nP(N)=0.99\", shape=\"box\") dot.node(\"D\", \"Positive\\nP(+|C)=0.80\", shape=\"box\") dot.node(\"E\", \"Negative\\nP(-|C)=0.20\", shape=\"box\") dot.node(\"F\", \"Positive\\nP(+|N)=0.05\", shape=\"box\") dot.node(\"G\", \"Negative\\nP(-|N)=0.95\", shape=\"box\") dot.edges([\"AB\", \"AC\", \"BD\", \"BE\", \"CF\", \"CG\"]) dot.edge(\"B\", \"D\", label=\"P(+|C)=0.80\\nP(+,C)=0.008\") dot.edge(\"B\", \"E\", label=\"P(-|C)=0.20\\nP(-,C)=0.002\") dot.edge(\"C\", \"F\", label=\"P(+|N)=0.05\\nP(+,N)=0.0495\") dot.edge(\"C\", \"G\", label=\"P(-|N)=0.95\\nP(-,N)=0.9405\") dot.render(\"bayes_tree\", format=\"png\", cleanup=True)      Continuous Random Variables   For continuous random variables, probabilities are defined via density functions, extending discrete concepts to uncountable outcome spaces.    Probability Density  A continuous random variable has a probability density function (PDF)  such that the probability over an interval is: The density normalizes: Expectation and variance are:   For two continuous variables , , the joint density  gives: The marginal density of is:    Bivariate normal distribution  A bivariate normal distribution for , with means , variances , and correlation has joint density: The marginal density of is the standard normal: .    Contour plot of the joint density of a bivariate normal distribution ( , , ). Marginal densities of is shown on the right, illustrating marginalization by integrating over .   Contour plot of bivariate normal joint density.     Bivariate normal joint density  # === Bivariate normal contour plot === import numpy as np import matplotlib.pyplot as plt from scipy.stats import multivariate_normal # Bivariate normal parameters mu = [0, 0] cov = [[1, 0.5], [0.5, 1]] # Correlation rho = 0.5 rv = multivariate_normal(mean=mu, cov=cov) x = np.linspace(-3, 3, 100) y = np.linspace(-3, 3, 100) X, Y = np.meshgrid(x, y) pos = np.dstack((X, Y)) Z = rv.pdf(pos) # Marginal density for X marginal_x = np.exp(-x**2 \/ 2) \/ np.sqrt(2 * np.pi) # Plot fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) ax1.contour(X, Y, Z, cmap=\"Blues\") ax1.set_xlabel(\"X\") ax1.set_ylabel(\"Y\") ax1.set_title(\"Bivariate Normal Joint Density (ρ=0.5)\") ax1.grid(True, alpha=0.3) ax2.plot(x, marginal_x) ax2.set_xlabel(\"X\") ax2.set_ylabel(\"Density\") ax2.set_title(\"Marginal Density of X\") ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"bivariate_normal.png\", dpi=300) plt.show()     Cumulative Distribution Function  The cumulative distribution function (CDF) of a continuous random variable is: The density is the derivative: , where differentiable.  For a standard normal distribution ( , ): Its CDF is , the standard normal CDF.   PDF and CDF of a standard normal distribution ( , ). The left panel shows the bell-shaped density, and the right panel shows the cumulative probability, illustrating .   Two-panel plot of standard normal PDF and CDF.     Standard normal PDF and CDF  # === Standard normal PDF and CDF === import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm mu, sigma = 0.0, 1.0 dist = norm(loc=mu, scale=sigma) x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000) pdf_values = dist.pdf(x) cdf_values = dist.cdf(x) fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5)) ax1.plot(x, pdf_values, label=\"PDF\") ax1.fill_between(x, pdf_values, alpha=0.2) ax1.set_xlabel(\"x\") ax1.set_ylabel(\"Density\") ax1.set_title(\"Standard Normal PDF\") ax1.grid(True, alpha=0.3) ax1.legend() ax2.plot(x, cdf_values, label=\"CDF\") ax2.set_xlabel(\"x\") ax2.set_ylabel(\"Cumulative Probability\") ax2.set_title(\"Standard Normal CDF\") ax2.grid(True, alpha=0.3) ax2.legend() plt.tight_layout() plt.savefig(\"normal_pdf_cdf.png\", dpi=300) plt.show()       Independent Random Variables  Random variables and are independent if their joint distribution factorizes: For continuous variables: Equivalently, or .  Independence implies , but zero covariance does not imply independence, except for jointly normal variables.   Independent vs. dependent variables  Consider the patient dataset with (disease status) and (test result), which are dependent (see ). Compare with , a patient’s age group (young\/old), assumed independent of . The plot below shows joint probabilities for dependent and independent cases.    Heatmaps comparing joint probabilities for dependent ( : disease, : test) and independent ( : disease, : age) variables. The left heatmap shows non-factorized probabilities, indicating dependence; the right shows factorized probabilities, confirming independence.   Heatmaps comparing dependent and independent joint probabilities.     Independence heatmap comparison  # === Heatmap for independent vs. dependent variables === import numpy as np import matplotlib.pyplot as plt import seaborn as sns # Dependent: X (disease), Y (test) joint_xy = np.array([[0.15, 0.05], [0.30, 0.50]]) # Independent: X (disease), Z (age: young\/old, P(young)=0.4, P(old)=0.6) P_X = np.array([0.2, 0.8]) # P(D), P(N) P_Z = np.array([0.4, 0.6]) # P(Young), P(Old) joint_xz = np.outer(P_X, P_Z) # Plot fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) sns.heatmap(joint_xy, annot=True, fmt=\".3f\", cmap=\"Blues\", cbar=False, xticklabels=[\"Y=+\", \"Y=-\"], yticklabels=[\"X=D\", \"X=N\"], ax=ax1) ax1.set_title(\"Dependent: P(X,Y)\") sns.heatmap(joint_xz, annot=True, fmt=\".3f\", cmap=\"Blues\", cbar=False, xticklabels=[\"Z=Young\", \"Z=Old\"], yticklabels=[\"X=D\", \"X=N\"], ax=ax2) ax2.set_title(\"Independent: P(X)P(Z)\") plt.tight_layout() plt.savefig(\"independence_heatmap.png\", dpi=300) plt.show()      Application: Medical Diagnosis  Joint and conditional probabilities are critical in medical diagnosis. Using the patient dataset, we estimate the probability of disease given symptoms and test results, combining multiple evidence sources.   Diagnosing disease with test and severity  Augment the patient dataset with symptom severity ( , normal, mean 5 for D, 3 for N, std 1.5). For a patient with a positive test ( ) and high severity ( ), compute using conditional probabilities.    Bar chart of conditional probabilities for combinations of test result ( ) and symptom severity ( or ). This illustrates how multiple features refine disease probability estimates in medical diagnostics.   Bar chart of conditional probabilities for diagnosis.     Conditional probability for diagnosis  import numpy as np import pandas as pd import matplotlib.pyplot as plt np.random.seed(42) # Synthetic patient data counts = {\"D+\": 150, \"D-\": 50, \"N+\": 300, \"N-\": 500} data = [] for status, test, count in [(\"D\", 1, 150), (\"D\", 0, 50), (\"N\", 1, 300), (\"N\", 0, 500)]: severity = np.random.normal(5 if status == \"D\" else 3, 1.5, count) data.extend([[1 if status == \"D\" else 0, test, s] for s in severity]) data = pd.DataFrame(data, columns=[\"Disease\", \"Test\", \"Severity\"]) data[\"High_Severity\"] = data[\"Severity\"] >= 4 # Compute conditional probabilities probs = [] for test, hs in [(1, True), (1, False), (0, True), (0, False)]: subset = data[(data[\"Test\"] == test) \\amp; (data[\"High_Severity\"] == hs)] if len(subset) > 0: p_d = np.mean(subset[\"Disease\"]) probs.append(p_d) else: probs.append(0) # Plot labels = [\"Y=+, S≥4\", \"Y=+, S<4\", \"Y=-, S≥4\", \"Y=-, S<4\"] plt.figure(figsize=(8, 6)) plt.bar(labels, probs, color=\"blue\") plt.ylabel(\"P(D | Y, S)\") plt.title(\"Conditional Probability of Disease\") plt.ylim(0, 1) plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"diagnosis_conditional.png\", dpi=300) plt.show()     Joint, marginal, and conditional probabilities form the foundation for modeling relationships between random variables. Joint probabilities capture co-occurrence, marginals summarize individual variables, and conditionals refine probabilities based on evidence. Bayes’ rule updates beliefs, while independence simplifies joint distributions. Visualizations like heatmaps, scatter plots, and tree diagrams clarify these concepts. Apply these tools in fields like medical diagnostics, as shown, or explore further with resources like Probability Course .   "
},
{
  "id": "subsec-Random-Variables-7",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-Random-Variables-7",
  "type": "Example",
  "number": "1.5.1",
  "title": "Die roll expectation and variance.",
  "body": " Die roll expectation and variance  For a fair six-sided die, for . The mean is: The second moment is: The variance is:   "
},
{
  "id": "fig-die-pmf",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-die-pmf",
  "type": "Figure",
  "number": "1.5.2",
  "title": "",
  "body": " Probability mass function (PMF) of a fair six-sided die, showing equal probabilities ( ) for outcomes . This bar chart visualizes the discrete distribution, useful for understanding expected values in games of chance.   Bar chart of die PMF.   "
},
{
  "id": "subsec-Joint-Probability-4",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-Joint-Probability-4",
  "type": "Example",
  "number": "1.5.3",
  "title": "Patient disease and test results.",
  "body": " Patient disease and test results  Consider a dataset of 1,000 patients, with indicating disease status ( : disease, : no disease) and indicating test result ( : positive, : negative). The joint counts are given in . Joint probabilities are:  Marginals are:    "
},
{
  "id": "fig-joint-heatmap",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-joint-heatmap",
  "type": "Figure",
  "number": "1.5.4",
  "title": "",
  "body": " Heatmap of joint probabilities for disease status ( ) and test result ( ) from 1,000 patients ( ). Darker shades indicate higher probabilities. Marginal probabilities ( , ) are shown in the margins, illustrating the summation process .   Heatmap of joint probabilities with marginals.   "
},
{
  "id": "subsec-Covariance-and-Correlation-4",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-Covariance-and-Correlation-4",
  "type": "Example",
  "number": "1.5.5",
  "title": "Patient symptom severity and test score.",
  "body": " Patient symptom severity and test score  For 1,000 patients, let be symptom severity (normal, mean 5 for disease, 3 for no disease, std 1.5) and be test score (normal, mean 80 for disease, 60 for no disease, std 10). The code below computes covariance and correlation, showing a positive relationship.  "
},
{
  "id": "fig-correlation-scatter",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-correlation-scatter",
  "type": "Figure",
  "number": "1.5.6",
  "title": "",
  "body": " Scatter plot of symptom severity vs. test score for 1,000 patients, with regression lines for disease (D) and no disease (N) groups. The computed correlation coefficient is shown, indicating a positive relationship between features, relevant for medical diagnostics.   Scatter plot with regression lines for correlation.   "
},
{
  "id": "subsec-Conditional-Probability-4",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-Conditional-Probability-4",
  "type": "Example",
  "number": "1.5.7",
  "title": "Patient disease and test results.",
  "body": " Patient disease and test results  Using the patient dataset ( ), compute all conditional probabilities:  Patient counts (disease vs. test)   Marginal  150 50  300 500  Marginal   Joint probabilities: Marginals: , , , . Conditional probabilities:   "
},
{
  "id": "fig-patient-bar-conditional",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-patient-bar-conditional",
  "type": "Figure",
  "number": "1.5.9",
  "title": "",
  "body": " Grouped bar chart of conditional probabilities for disease status ( ) given test result ( ) and test result given disease status, based on . Each group shows a probability distribution (summing to 1), illustrating how conditional probabilities slice the joint distribution.   Grouped bar chart of conditional probabilities.   "
},
{
  "id": "subsec-conditional-probability-from-joint-probability-3",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-conditional-probability-from-joint-probability-3",
  "type": "Example",
  "number": "1.5.10",
  "title": "Breast cancer screening.",
  "body": " Breast cancer screening  For a population with breast cancer prevalence , a mammogram has:  Sensitivity : .  Specificity : , so .  Compute . First, the marginal probability: Then: Thus, a positive test increases the probability of cancer from 1% to 13.91%.  "
},
{
  "id": "fig-bayes-tree",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-bayes-tree",
  "type": "Figure",
  "number": "1.5.11",
  "title": "",
  "body": " Tree diagram for breast cancer screening, showing prior probabilities ( , ), likelihoods ( , etc.), and joint probabilities leading to . This visualizes Bayes’ rule for medical diagnostics.   Tree diagram for Bayes’ rule in screening.   "
},
{
  "id": "subsubsec-Probability-Density-4",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsubsec-Probability-Density-4",
  "type": "Example",
  "number": "1.5.12",
  "title": "Bivariate normal distribution.",
  "body": " Bivariate normal distribution  A bivariate normal distribution for , with means , variances , and correlation has joint density: The marginal density of is the standard normal: .  "
},
{
  "id": "fig-bivariate-normal",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-bivariate-normal",
  "type": "Figure",
  "number": "1.5.13",
  "title": "",
  "body": " Contour plot of the joint density of a bivariate normal distribution ( , , ). Marginal densities of is shown on the right, illustrating marginalization by integrating over .   Contour plot of bivariate normal joint density.   "
},
{
  "id": "fig-normal-pdf-cdf",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-normal-pdf-cdf",
  "type": "Figure",
  "number": "1.5.14",
  "title": "",
  "body": " PDF and CDF of a standard normal distribution ( , ). The left panel shows the bell-shaped density, and the right panel shows the cumulative probability, illustrating .   Two-panel plot of standard normal PDF and CDF.   "
},
{
  "id": "subsec-Independent-Variables-4",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-Independent-Variables-4",
  "type": "Example",
  "number": "1.5.15",
  "title": "Independent vs. dependent variables.",
  "body": " Independent vs. dependent variables  Consider the patient dataset with (disease status) and (test result), which are dependent (see ). Compare with , a patient’s age group (young\/old), assumed independent of . The plot below shows joint probabilities for dependent and independent cases.  "
},
{
  "id": "fig-independence-heatmap",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-independence-heatmap",
  "type": "Figure",
  "number": "1.5.16",
  "title": "",
  "body": " Heatmaps comparing joint probabilities for dependent ( : disease, : test) and independent ( : disease, : age) variables. The left heatmap shows non-factorized probabilities, indicating dependence; the right shows factorized probabilities, confirming independence.   Heatmaps comparing dependent and independent joint probabilities.   "
},
{
  "id": "subsec-Application-Medical-Diagnosis-3",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#subsec-Application-Medical-Diagnosis-3",
  "type": "Example",
  "number": "1.5.17",
  "title": "Diagnosing disease with test and severity.",
  "body": " Diagnosing disease with test and severity  Augment the patient dataset with symptom severity ( , normal, mean 5 for D, 3 for N, std 1.5). For a patient with a positive test ( ) and high severity ( ), compute using conditional probabilities.  "
},
{
  "id": "fig-diagnosis-conditional",
  "level": "2",
  "url": "sec-Joint-Conditional-and-Marginal-Probabilities.html#fig-diagnosis-conditional",
  "type": "Figure",
  "number": "1.5.18",
  "title": "",
  "body": " Bar chart of conditional probabilities for combinations of test result ( ) and symptom severity ( or ). This illustrates how multiple features refine disease probability estimates in medical diagnostics.   Bar chart of conditional probabilities for diagnosis.   "
},
{
  "id": "sec-Example-Discrete-Probability-Distributions",
  "level": "1",
  "url": "sec-Example-Discrete-Probability-Distributions.html",
  "type": "Section",
  "number": "1.6",
  "title": "Example Discrete Probability Distributions",
  "body": " Example Discrete Probability Distributions   Distributions describe how probabilities are spread across values of a random variable. We will given examples of Probability Mass Function (PMF) for the discrete random variables. We will discuss examples of Probability Density Function (PDF) for continuous random variable in the next section.    Bernoulli Distribution  the outcome of each elementary event of a Bernoulli trial is either a failure or success of something 9false or true of some statement, or any myriads of two states problems, which is usually represented by a random variable having values and . That is We call such random variables Bernoulli variables . The values of probability of each value of gives us the Probability Mass Function (PMF) of the Bernoulli distribution. Suppose probability of is . Then, probability of will be . Sometimes is denote by , with .  The separate listing of the probability of the two values of in Eqs. and can actually be written more conveniently in one formula. From this formula, you will get and by substitting appropriate value of .   Graphically, Bernoulli distribution is plotted as bars with a dot at the top of the bar. Figure to the side shows an illustration with for , and of course, for .     Bernoulli distribution with .    For any distribution, we can find the mean of variable by weighing each value of with its probability. Similarly we can find the expectation value of any power of . For instance, the expectation value of the power of will be Thus, variance of a Bernoulli variable will be Therefore, the standard deviation, ,of Bernoulli variable is     Binomial Distribution  Imagine tossing a single coin a fixed number number of times, say times. You might get no Heads at all or 1 Heads and 9 Tails, or 2 Heads and 8 Tails, etc. Record how many Heads you got in this trial, say you got 3 Heads. Now, toss the same coin 10 times again. This will be the second trial of experiment \"tossing a particular coin 10 times\". In the second trial, you might get a different number of Heads, say this time you got 8 Heads.  If you repeated the experiment above hundreds or thousands of times, you can build a table of number of trials that resulted in a total of Heads, Heads, Heads, , Heads, which are all the possibilities. This table, an example shown in for 2000 trials, will be our Frequency Table . By dividing each of these numbers by the total number of trials you performed, you will get an estimate of probabilities of each outcome. The exact formula that gives the distribution you found is called Bernoulli distribution for the -toss experiment .   Binomial Experiment: Frequency and Approximate Probability    Total Number of Heads  Frequency  Approximate Probability    0  8  P(0) \\approx 0.004    1  45  P(1) \\approx 0.0225    2  120  P(2) \\approx 0.06    3  220  P(3) \\approx 0.11    4  300  P(4) \\approx 0.15    5  350  P(5) \\approx 0.175    6  300  P(6) \\approx 0.15    7  250  P(7) \\approx 0.125    8  200  P(8) \\approx 0.10    9  150  P(9) \\approx 0.075    10  57  P(10) \\approx 0.0285    Total Number of Trials:  2000     In general, Binomial distribution gives us the probability of different number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success . In each trial of a Binomial experiments you have a sequence of Bernoulli repeats of (for success) and (for failure). Suppose this sequence has Heads ( ) and Tails ( ). The probability of this sequence of Bernoulli outcomes will be But the and could have occurred in any order.To get the probability of getting a total of Heads in any order, we multiply number of different orders in which we could have got the same total number of Heads. That turns out to be the Binomial coefficient, and hence the name Binomial distribution. Note that this distribution has two fixed parameters , the number of independent Bernoulli trials in each Binomial trial and , the probability of success in each Bernoulli trial. Beware of the importance of ; you can think of there being infinitely many Binomial distributions, each corresponding to different values of . For instance, in Table , if you had conducted the experiment with Bernoulli trials in each Binomial experiment, instead of , you would have gotten much different probabilities for . This is illustrated in the following figure, .   Illustrating that different values in Binomial distribution correspond to different distributions. Here, with and . See that the probabilities for same value are different for the two distributions.   llustrating that different values in Binomial distribution correspond to different distributions. Here, with and . See that the probabilities for same value are different for the two distributions.    Following code was used to create the plot above.   import numpy as np import matplotlib.pyplot as plt from scipy.stats import binom # Parameters p = 0.5 # probability of success N1 = 20 # number of trials for first distribution N2 = 10 # number of trials for second distribution # Support for each distribution x1 = np.arange(0, N1+1) x2 = np.arange(0, N2+1) # PMFs pmf1 = binom.pmf(x1, N1, p) pmf2 = binom.pmf(x2, N2, p) # Plot fig, ax = plt.subplots(figsize=(8,5)) # Binomial N1 ax.vlines(x1, 0, pmf1, colors='blue', lw=2, label=f'Bin{N1}') ax.plot(x1, pmf1, 'o', color='blue') # Binomial N2 ax.vlines(x2, 0, pmf2, colors='orange', lw=2, label=f'Bin{N2}') ax.plot(x2, pmf2, 'o', color='orange') # Labels and grid ax.set_title(f'Binomial Distribution PMFs (p={p})') ax.set_xlabel('Number of Successes') ax.set_ylabel('Probability') ax.grid(axis='y', linestyle='--', alpha=0.6) ax.legend() plt.show()   Another way to improve your intuition about the Binomial distribution is to look at the impact of changing value for the Bernoulli trials themselves - what impact do they have on a -Binomial? It is shown in . These plots show that low skews the PMF toward fewer successes; produces a symmetric distribution centered at ; high skews toward more successes.   Illustrating that different values in Binomial distribution correspond to different distributions but with .   Illustrating that different values in Binomial distribution correspond to different distributions but with .    For doing analytical calculations with the Binomial distribution, it is important to recall the following algebraic identity, called Binomial expansion. Using this it is straightforward to show that Binomial distribution is normalized properly since The mean of the Binomial random variable can be obtained by weighing each value of by the corresponding probability. A simple method of showing the result involves taking an appropriate derivative appropriately. The variance is similarly shown to be And, the standard deviation is just the square root.   Binomial distribution plays important role in understanding average of several Bernoulli random variables, say , which have the same . Suppose, we denote Bernoulli variables by . Then, their sum will be a Binomial random variable, if Bernoulli random variables take or as awe have discussed above. The average will be a scaled Binomial variable. We will denote this random variable by with a bar above the symbol and a reminder that it is average of Bernoulli variables. This random variable is called sample mean . It will take the following values: With being probability of any of the individual Bernoulli variables to produce a success, i.e., for every one of the . But the variance is quite interesting which translates to the standard deviation  Where does this matter?       Estimation: In statistics, we often estimate by from data.     Interpretation: If you run multiple experiments, your average success rate will be centered at and become more concentrated as grows since the standard deviation drops as . This is illustrated in .     Connection to the Central Limit Theorem: For large , it can be shown that the probability distribution of the random variable tends to become Gaussian with the mean and variance . We write this as even though each is a discrete random variable. This goes by the name Central Limit Theorem .     Illustrating that for large the distribution of the average of Bernoulli variables of the same tends towards a Gaussian distribution.   Illustrating that for large the distribution of the average of Bernoulli variables of the same tends towards a Gaussian distribution.       Poisson Distribution   The Poisson distribution models the number of times an event occurs in a fixed interval of time or space, given that:   Events occur independently.    Events happen at a constant average rate, usually denoted by Greek letter lambda .    Two events cannot occur at the exact same instant. That means we are usually interested in events that are rare within the interval we choose to work with so that it can be safely assumed that two events do not coincide.   A Poisson random variable can take any non-negative integer values since it's just a count. The probability mass function for a Poisson random variable will give probabilities for for each non-negative value for a constant average rate is given by It is obviously normalized since Therefore The mean of Poisson distribution is the average count, \\lambda. as we can show by the following calculations. The variance of Poisson distribution is similarly shown to be also . where the missing steps are left for the student to practice, using the same type of argument as introducing operators appropriately.  A mathematically interesting result is that in an appropriate limit, a Binomial distribution can be shown to become same as Poisson distribution. I will just state the result without giving you the detailed calculations. (hint: You can get factors of from ).    Example Radioactivity is one of the classic and most intuitive real-life examples of the Poisson distribution. Let's look at it a little closely. Radioactive decay is a random process. Each atom has a constant probability of decaying in a fixed time interval. The decays are:    Independent (one decay does not affect another).     Rare events relative to the huge number of atoms.    Occurring with a constant average rate .   These aspects make the Poisson distribution a perfect model for studying the statistics of radioactivity.  Suppose we measure the number of particles emitted from a radioactive source in 10-second intervals. From past experiments, we know that the detector records on average 3 decays per 10 seconds. So, per second, we expect on-average decays. That completely specifies the Poisson distribution. Therefore, we can immediately calculate all sorts of things for the phenomenon. For instance, probability of seeing exactly decays in seconds will be Probability of exactly 3 decays in a second will be Now, for a trick question. What will be the probability of 10 decays in one minute? Well, we will convert our lambda per second to a new lambda per minute. Let's label lambda's by the intervals they refer to. Then   A visual representation of the PMF often helps to build intuition. A simple program in Python can be used to to do that. The plot with is shown in .  import numpy as np import matplotlib.pyplot as plt from scipy.stats import poisson # Average decay rate lam = 3 # Range of possible counts x = np.arange(0, 15) pmf = poisson.pmf(x, lam) plt.figure(figsize=(8,5)) plt.vlines(x, 0, pmf, colors='darkred', lw=2, alpha=0.7, label=f'λ={lam}') plt.plot(x, pmf, 'o', color='black', markersize=5) plt.xlabel('Number of Decays in One Interval') plt.ylabel('Probability P(X=k)') plt.title('Poisson Distribution of Radioactive Decays (λ=3)') plt.grid(axis='y', linestyle='--', alpha=0.6) plt.legend() plt.show()     Poisson Distribution for . The most likely outcome is 3 decays per 10 seconds (the mean). Note that 0 or 1 decay is possible, but much less likely. Seeing 6 or more decays is rare, but not impossible.   Poisson Distribution for .      Poisson Process  A Poisson process is a stochastic process used to model the occurrence of events that happen independently and at a constant average rate over time or space.  Formally, a Poisson process is a counting process , where represents the number of events that have occurred up to time  , and it satisfies the following properties:    Initial Condition : , meaning the process starts with no events at time zero.     Independent Increments : The number of events in non-overlapping time intervals is independent. For example, the number of events in is independent of the number in if the intervals do not overlap.     Stationary Increments : The number of events in a time interval of length , i.e., , depends only on the length and not on the starting point .     Poisson Distribution : The number of events in any interval of length follows a Poisson distribution with mean , where is the rate parameter (average number of events per unit time). The probability of events in an interval of length is:      No Simultaneous Events : The probability of two or more events occurring at exactly the same time is negligible (technically, the probability of multiple events in an infinitesimally small interval is zero).       Examples :    Queueing Systems : Customers arriving at a store at an average rate of customers per hour.     Telecommunications : Phone calls arriving at a call center with a constant average rate.     Reliability : Failures of a machine occurring randomly at an average rate of failures per hour.     Traffic : Cars passing a checkpoint on a highway at a constant average rate.       "
},
{
  "id": "tab-Binomial-frequency-and-prob",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#tab-Binomial-frequency-and-prob",
  "type": "Table",
  "number": "1.6.1",
  "title": "Binomial Experiment: Frequency and Approximate Probability",
  "body": " Binomial Experiment: Frequency and Approximate Probability    Total Number of Heads  Frequency  Approximate Probability    0  8  P(0) \\approx 0.004    1  45  P(1) \\approx 0.0225    2  120  P(2) \\approx 0.06    3  220  P(3) \\approx 0.11    4  300  P(4) \\approx 0.15    5  350  P(5) \\approx 0.175    6  300  P(6) \\approx 0.15    7  250  P(7) \\approx 0.125    8  200  P(8) \\approx 0.10    9  150  P(9) \\approx 0.075    10  57  P(10) \\approx 0.0285    Total Number of Trials:  2000    "
},
{
  "id": "fig-binomial-10N20",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-binomial-10N20",
  "type": "Figure",
  "number": "1.6.2",
  "title": "",
  "body": " Illustrating that different values in Binomial distribution correspond to different distributions. Here, with and . See that the probabilities for same value are different for the two distributions.   llustrating that different values in Binomial distribution correspond to different distributions. Here, with and . See that the probabilities for same value are different for the two distributions.   "
},
{
  "id": "fig-binomial-N10p2p5pp8",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-binomial-N10p2p5pp8",
  "type": "Figure",
  "number": "1.6.3",
  "title": "",
  "body": " Illustrating that different values in Binomial distribution correspond to different distributions but with .   Illustrating that different values in Binomial distribution correspond to different distributions but with .   "
},
{
  "id": "fig-bernoulli-to-CLT",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-bernoulli-to-CLT",
  "type": "Figure",
  "number": "1.6.4",
  "title": "",
  "body": " Illustrating that for large the distribution of the average of Bernoulli variables of the same tends towards a Gaussian distribution.   Illustrating that for large the distribution of the average of Bernoulli variables of the same tends towards a Gaussian distribution.   "
},
{
  "id": "fig-poisson-distribution",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-poisson-distribution",
  "type": "Figure",
  "number": "1.6.5",
  "title": "",
  "body": " Poisson Distribution for . The most likely outcome is 3 decays per 10 seconds (the mean). Note that 0 or 1 decay is possible, but much less likely. Seeing 6 or more decays is rare, but not impossible.   Poisson Distribution for .   "
},
{
  "id": "sec-Example-Continuous-Probability-Distributions",
  "level": "1",
  "url": "sec-Example-Continuous-Probability-Distributions.html",
  "type": "Section",
  "number": "1.7",
  "title": "Example Continuous Probability Distributions",
  "body": " Example Continuous Probability Distributions   Continuous random variables take values on an interval of the real line. Unlike discrete random variables, we do not assign probability to single points (which would be zero). Instead, we assign probability to intervals, using probability density functions ( PDFs ).  In this section, we will discuss three distributions that are important for ML use.    Uniform: All values equally likely     Normal (Gaussian): Bell-shaped curve     Exponential: Time until an event occurs   There are, of course, other distributions such as beta and gamma distributions that are also in common use. By presenting just these three here, I hope to give the reader enough feel for what to look for when studying other distributions.    Uniform Distribution   When a random variable is equally likely to take any value between two real numbers and , we say that the distribution is uniform between these values. The distribution is usually designated by with , and say that which means that where the part without the is the PDF of the distribution, which we denote by the Greek letter rho, .  The mean value of the distribution will clearly be half way between and as can easily be computed by performing the simple integral. The variance, which is square of the standard deviation , is similarly calculated to yield where Thus, standard deviation of is:   The cumulative distribution function, , which gives the probability that is easily calculated for uniform distribution. where I used for the dummy variable since now is a particular value. This will be a step function since the formula resulting from the integration depends on where the point happens to lie. The last line says that probability that has any value less than any value in is 1.0 since obviously the entire range is included in this case. The second line says that the probability increases linearly between and .   shows plots of PDF and CDF of uniform distribution . You can see that as we scan through the interval of the uniform PDF, the probability accumulates in the CDF and eventually, CDF becomes , which represents probability of any of the values in the interval.   PDF and CDF of Uniform distribution . Note the value of PDF is uniformly while that of CDF increases linear in the interval.   PDF and CDF of Uniform distribution . Note the value of PDF is uniformly while that of CDF increases linear in the interval.    To generate these plots import: from scipy.stats import uniform and then use the methods uniform.pdf() and uniform.cdf().    Inverse Uniform CDF  Inverse of a CDF is used for sampling from a distribution. Although, we will see below that inverse CDF of the uniform distribution is trivial, the inverse CDF is highly useful when you need to generate samples from some other distributions such as Normal, Exponential, Gamma, etx.  The inverse of CDF is written as - it's not a negative one power of ; that is just the name of the inverse function. As the name implies when you successively apply an to some number , you would get that number back. So, what does it look like for the uniform distribution ? The CDF in the range is The inverse will be Let's check if that's true. So, if you wanted to generate samples in the range that act like they are sampled from the uniform distribution , you would first generate pseudo-random numbers in unit interval using algorithms lke Mersenne Twister. Suppose, you have such sample . Then, you will plug them in the inverse CDF to generate samples from . This is trivial here since we have an analytic expression for the inverse CDF. In other districutions, such as the Gaussian, i.e., Normal distribution, the inverse can only be computed numerically. The stats packages usually have functions that do it for you. For instance, ppf() method in Python\/scipy.stats is used for that purpose. In case of uniform distribution, the command is  from scipy.stats import uniform samples = uniform.ppf(u, loc=a, scale=b-a).       Normal (Gaussian) Distribution   The PDF of a Gaussian or Normal distribution is a bell-shaped curve with only two parameters, a mean and a standard deviation . The name Gaussian is preferred in Physics and Engineering, and Normal is preferred in statistics and data science. In these notes I will use both of them, just for fun.  The PDF of the Gaussian distribution of mean and standard deviation for a scalar variable is defined by where The reason I am writing rather than is that exponent in the latter expression usually prints too small on the screen. While doing calculations by hand, you should stick to notation.  When the distribution of a random variable is Gaussian of mean and standard deviation , i.e., variance we denote this as a short hand notation by The special case of and is called standard normal distribution . A standard normal random variable will obey   As always, the PDF in Eq. has the probability interpretation in an infinitesimal interval around . Thus, if you want the probability of , you will just integrate it. The CDF is just such an integral for the probability of . If , then entire real line is included. That would make The integral is unwieldy and only done numerically. The Fundamental Theorem of Calculus gives us an analytic expression of the derivative of CDF, which is used in formal analytical work. This is, of course, a general result and applies to all PDF\/CDF and forms a powerful tool of formal work. We are not going to do much in that direction.   shows plots of PDF and CDF of Gaussian distribution .   PDF and CDF of Gaussian distribution . Note the bell-shape of the PDF and the soft step of the CDF which goes from to .   PDF and CDF of Gaussian distribution . Note the bell-shape of the PDF and the soft step of the CDF which goes from to .    To generate these plots import: from scipy.stats import uniform and then use the methods uniform.pdf() and uniform.cdf().  The mean and variance of a Gaussian is in the definition itself and can be readily checked if you know how do Gaussian integrals. Here are couple of tricks of doing Gaussian integrals.     Inverse CDF and Sampling  As we defined for the uniform distribution above, inverse of the Cumulative Distribution Function (CDF) is another function, denoted by (NotE: it's not . The negative power in the symbol is just a symbol.) The functions and are inverses so that when you act by them, one does the effect of the other. But unlike the case was with the uniform distribution, here we only know in the integral form. However, you can find of Gaussian distribution numerically. It is already programmed in stats packages. For instance, we can compute the percent point function (PPF), which is the numerical inverse CDF by scipy.stats.norm.ppf() function by passing the appropriate parameters.  The CDF function is a mapping from to . The inverse will map to . To sample randomly from Gaussian distribution means producing random values of . That means if we obtain random values and feed that into , the result will be random values of sampled according to the distribution of the , which is Gaussian in the present case. Obtaining random values can be done by just sampling from the uniform distribution .  Thus the steps of sampling from a distribution is:   Generate uniform random numbers .    Apply , if using inverse of standard normal. If using scipy.stats pacakge, the code for will be scipy.stats.norm.ppf( , loc = , scale = ) . In scipy.stats, you can include the actual loc and scale in the argument itself as shown in the program listing below.    Use as your Gaussian samples.   Let us look at an example of drawing from a Gaussian distribution and how the samples match up with the theoretical distribution. This is shown in . It was produced by the code below. Clearly, the histogram based on the samples is very representative of the theoretical curve.   import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Parameters mu, sigma = 0, 1 n_samples = 1000 # Step 1: uniform samples u = np.random.rand(n_samples) # Step 2: transform with inverse CDF (ppf) samples = norm.ppf(u, loc=mu, scale=sigma) # Plot histogram vs theoretical PDF x = np.linspace(-4, 4, 200) pdf = norm.pdf(x, mu, sigma) plt.figure(figsize=(7,5)) plt.hist(samples, bins=30, density=True, alpha=0.6, label=\"Sampled (inverse CDF)\") plt.plot(x, pdf, 'r-', lw=2, label=\"Theoretical PDF\") plt.xlabel(\"x\") plt.ylabel(\"Density\") plt.title(\"Sampling from Normal(0,1) using Inverse CDF\") plt.legend() plt.show()    Samples from a Gaussian distribution and the theoretical curve. The histogram is based on 1000 sample points.   Samples from a Gaussian distribution and the theoretical curve. The histogram is based on 1000 sample points.       Exponential Distribution  Exponential distribution is commonly used to model the time between successive events in a Poisson process, where events occur independently and at a constant average rate. The distribution is defined for non-negative values ( ) and is characterized by a single parameter, (lambda), which is the rate parameter, as described in the section on the Poisson distribution.  The PDF of Exponential distribution is given by As usual, it has the following probability interpretation. You can verify that the PDF in Eq. is properly normalized to give probability over the entire range f values, i.e, is . The Cumulative Distribution Function, (CDF), is the probability for . Therefore,  shows the PDF and CDF of the exponential distribution for .   PDF and CDF of Exponential distribution . Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from to .   PDF and CDF of Exponential distribution . Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from to .    The complement of the CDF, i.e., the probability that is called survival function (SF).   From the survival function, it is possible to prove and important property of exponential distribution: that it is meomryless , meaning that the probability of an event occurring in the next time interval does not depend on how much time has already elapsed. In formulas, this will be a condition on the conditional probability. That is, whether you wait upto and then look at the next units of time ot you don't wait and look at the next interval of time, the two will give the same probability - same probability for intervals of time will be independent of the starting instant.   Proof of Memoryless Property : Let's write the left side of Eq. in terms of joint and prior, based on the definition of conditional probabilities given in an earlier section. Since the joint probability, you see that joint probability will simply equal . Therefore, Now, we use the survival function given in Eq. to write the right hand side quantities and then simplify.   Finally, let's go over the mean and variance of exponential distribution. Mean as usual is the expectation value of the random variable . The variance will be from which we get the standard deviation .    Real-World Examples :   Radioactive Decay:  Time until the next decay event for a particle with decay rate .    Queueing Systems:  Time until the next customer arrives at a store, assuming arrivals follow a Poisson process with rate customers per hour. Mean waiting time = hours.    Reliability:  Lifetime of a lightbulb that fails at a constant rate failures per hour. The probability it lasts more than hours is .      "
},
{
  "id": "fig-uniform-pdf-cdf",
  "level": "2",
  "url": "sec-Example-Continuous-Probability-Distributions.html#fig-uniform-pdf-cdf",
  "type": "Figure",
  "number": "1.7.1",
  "title": "",
  "body": " PDF and CDF of Uniform distribution . Note the value of PDF is uniformly while that of CDF increases linear in the interval.   PDF and CDF of Uniform distribution . Note the value of PDF is uniformly while that of CDF increases linear in the interval.   "
},
{
  "id": "fig-gaussian-pdf-cdf",
  "level": "2",
  "url": "sec-Example-Continuous-Probability-Distributions.html#fig-gaussian-pdf-cdf",
  "type": "Figure",
  "number": "1.7.2",
  "title": "",
  "body": " PDF and CDF of Gaussian distribution . Note the bell-shape of the PDF and the soft step of the CDF which goes from to .   PDF and CDF of Gaussian distribution . Note the bell-shape of the PDF and the soft step of the CDF which goes from to .   "
},
{
  "id": "fig-gaussian-sampling",
  "level": "2",
  "url": "sec-Example-Continuous-Probability-Distributions.html#fig-gaussian-sampling",
  "type": "Figure",
  "number": "1.7.3",
  "title": "",
  "body": " Samples from a Gaussian distribution and the theoretical curve. The histogram is based on 1000 sample points.   Samples from a Gaussian distribution and the theoretical curve. The histogram is based on 1000 sample points.   "
},
{
  "id": "fig-exponential-pdf-cdf",
  "level": "2",
  "url": "sec-Example-Continuous-Probability-Distributions.html#fig-exponential-pdf-cdf",
  "type": "Figure",
  "number": "1.7.4",
  "title": "",
  "body": " PDF and CDF of Exponential distribution . Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from to .   PDF and CDF of Exponential distribution . Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from to .   "
},
{
  "id": "sec-LLN-and-CLT",
  "level": "1",
  "url": "sec-LLN-and-CLT.html",
  "type": "Section",
  "number": "1.8",
  "title": "Law of Large Numbers and Central Limit Theorem",
  "body": " Law of Large Numbers and Central Limit Theorem   The Law of Large Numbers (LLN) and Central Limit Theorem (CLT) are foundational results in probability and statistics, underpinning many inferential techniques. The LLN ensures that sample averages converge to the population mean, while the CLT describes the distribution of those averages as approximately normal for large samples. This section provides a detailed yet accessible exploration of these theorems, their assumptions, and their applications.   The Setup:   Consider a sequence of independent and identically distributed (i.i.d.) random variables , each with the same probability distribution, mean , and variance , where denotes the expected value. These are called the true mean and true variance . We define the sample mean as: The sample mean is itself a random variable with its own distribution. For example, if is a Bernoulli random variable with and , then takes values in . The LLN and CLT describe the behavior of as increases.  The LLN and CLT require that the are i.i.d. with finite mean . The CLT additionally requires finite variance , and the Strong LLN requires a slightly stronger condition on the moments of .    Law of Large Numbers  The LLN addresses the question: What happens to the sample mean as the sample size grows large? It comes in two forms: the Weak LLN and the Strong LLN, which differ in their modes of convergence.   Weak LLN: The Weak LLN states that the probability that deviates from the true mean by more than any positive amount approaches zero as : This is called convergence in probability , denoted:    Strong LLN: The Strong LLN states that the sequence of sample means converges to with probability 1: This is called almost sure convergence , a stronger condition implying that almost all sample paths of converge to .   Intuitive Example: Consider tossing a fair coin with , so for heads and for tails, with . For , is either 0 or 1. For , suppose we observe 7 heads, so . For , we might get , and for , . As increases, gets closer to , illustrating the LLN. The Strong LLN guarantees this convergence occurs almost surely.   Visualization: The following figure shows the sample mean of rolls of a fair six-sided die (with true mean ) for increasing , converging to .  Illustration of the Law of Large Numbers: Sample means of fair six-sided die rolls converge to the true mean as increases.   Sample means of die rolls converging to 3.5.       Central Limit Theorem  The CLT states that for i.i.d. random variables with finite mean and variance , the distribution of the sample mean becomes approximately normal as increases: where denotes convergence in distribution. Equivalently, the standardized sample mean: converges to a standard normal distribution: The probability density of for large is approximately: The cumulative distribution function (CDF) of is:   The CLT is remarkable because it holds regardless of the underlying distribution of (e.g., Bernoulli, exponential, or normal), as long as and are finite. This explains why normal distributions appear in phenomena like measurement errors, test scores, or heights, which are aggregates of many small random effects.   Example: For an exponential distribution with rate (mean , variance ), the sample mean of observations is approximately . The CLT allows us to compute probabilities, such as , using the normal distribution.   Visualization:  shows histograms of for a fair six-sided die for different sample sizes, illustrating convergence to a normal distribution.  Histograms of sample means of fair six-sided die rolls for , visually showing convergence to a normal distribution per the CLT.   Histograms showing CLT convergence for die rolls.       LLN vs. CLT  The LLN and CLT are complementary:  LLN: Ensures that (in probability or almost surely), describing the convergence of the sample mean to the true mean.  CLT: Describes the distribution of fluctuations around , stating that , with deviations of order .  In essence, the LLN tells us where the sample mean goes, while the CLT tells us how it fluctuates around that value.    Berry-Esseen Theorem  The CLT states that is approximately normal for large , but how large must be? The Berry-Esseen Theorem quantifies the rate of convergence. Let be the standardized sum, with CDF . The theorem states: where is the third absolute moment, is the standard normal CDF, and is a constant ( , Shevtsova 2011). This implies that the error in the normal approximation decreases as , modulated by the “tail-heaviness” factor .  For example, for a fair six-sided die, is finite, and for , the approximation error is small, ensuring reliable use of the CLT in practice.    Why LLN and CLT Matter   LLN Applications: The LLN justifies using sample means to estimate population means. For example, in polling, we survey a sample to estimate voter preferences. The LLN ensures that with a large enough sample, the sample mean is close to the true population mean.  Consider rolling a fair six-sided die with true mean and variance . For rolls, suppose we compute . The LLN suggests . illustrates this convergence.  In statistical mechanics, the LLN applies to time averages in ergodic systems, ensuring that long-term observations of a particle’s behavior approximate the population average.   CLT and Confidence Intervals: The CLT enables us to quantify uncertainty in sample means via confidence intervals. For the die example, suppose we roll times and compute and sample standard deviation . The CLT implies: Using the sample standard deviation , the standard error is . A 95% confidence interval is: This interval suggests that we are 95% confident that lies between 3.265 and 3.935, consistent with the true mean .   Simulation: The following Python code simulates 10,000 trials of 100 die rolls, computes sample means, and plots a histogram with a 95% confidence interval:  Simulating Dice Rolls and Confidence Interval  import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Step 1: Simulate dice rolls np.random.seed(42) n_trials = 10000 n_rolls = 100 data = np.random.randint(1, 7, size=(n_trials, n_rolls)) sample_means = data.mean(axis=1) # Step 2: Compute 95% CI mean_est = np.mean(sample_means) std_est = np.std(sample_means, ddof=1) z_975 = norm.ppf(0.975) margin = z_975 * std_est ci_lower, ci_upper = mean_est - margin, mean_est + margin print(f\"Estimated mean = {mean_est:.3f}\") print(f\"95% CI ≈ [{ci_lower:.3f}, {ci_upper:.3f}]\") # Step 3: Plot histogram with CI fig, ax = plt.subplots(figsize=(8,5)) counts, bins, patches = ax.hist(sample_means, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black') plt.axvline(mean_est, color='red', linestyle='--', label=\"Mean of sample means\") plt.axvline(ci_lower, color='black', linestyle='dashed', label=\"95% CI\") plt.axvline(ci_upper, color='black', linestyle='dashed') y_arrow = counts.max() \/ 3 plt.annotate('', xy=(ci_lower, y_arrow), xytext=(ci_upper, y_arrow), arrowprops=dict(arrowstyle='<->', color='black', lw=2)) plt.text(mean_est, y_arrow * 1.1, \"95% CI\", ha='center', fontsize=12) plt.title(\"Sampling Distribution of Dice Means (100 rolls, 10,000 trials)\") plt.xlabel(\"Sample Mean\") plt.ylabel(\"Density\") plt.legend() plt.show()    Histogram of sample means from 10,000 trials of 100 fair six-sided die rolls, with a 95% confidence interval (dashed lines and arrow) around the estimated mean.   Histogram of die roll sample means with 95% CI.      Other Applications: The CLT is crucial for hypothesis testing (e.g., z-tests) and approximating probabilities for sums of random variables, such as total customer purchases in a store.   "
},
{
  "id": "fig-die-rolls-sample-mean-vs-true-mean",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#fig-die-rolls-sample-mean-vs-true-mean",
  "type": "Figure",
  "number": "1.8.1",
  "title": "",
  "body": " Illustration of the Law of Large Numbers: Sample means of fair six-sided die rolls converge to the true mean as increases.   Sample means of die rolls converging to 3.5.   "
},
{
  "id": "fig-clt-convergence",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#fig-clt-convergence",
  "type": "Figure",
  "number": "1.8.2",
  "title": "",
  "body": " Histograms of sample means of fair six-sided die rolls for , visually showing convergence to a normal distribution per the CLT.   Histograms showing CLT convergence for die rolls.   "
},
{
  "id": "fig-confidence-interval-die-roll",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#fig-confidence-interval-die-roll",
  "type": "Figure",
  "number": "1.8.3",
  "title": "",
  "body": " Histogram of sample means from 10,000 trials of 100 fair six-sided die rolls, with a 95% confidence interval (dashed lines and arrow) around the estimated mean.   Histogram of die roll sample means with 95% CI.   "
},
{
  "id": "sec-Inferential-Statistics",
  "level": "1",
  "url": "sec-Inferential-Statistics.html",
  "type": "Section",
  "number": "1.9",
  "title": "Inferential Statistics",
  "body": " Inferential Statistics   Inferential statistics is the branch of statistics that enables us to draw conclusions about a population based on data from a sample. Unlike descriptive statistics, which summarize observed data, inferential statistics use probability theory to make generalizations about the population from which the sample is drawn. This is essential in fields such as medicine, economics, and social sciences, where measuring an entire population is often impractical. For example, a survey of 1,000 voters can estimate the preferences of millions. This section focuses on frequentist methods, which rely on hypothesis testing and confidence intervals, but Bayesian methods, which incorporate prior knowledge to compute probabilities of hypotheses, are also used in statistical inference.  The core concepts of inferential statistics include:   Population : The entire group of interest.  Sample : A subset of the population that is observed.  Parameter : A numerical characteristic of the population (e.g., population mean , variance ).  Statistic : A numerical characteristic computed from the sample (e.g., sample mean , sample proportion ).     Point Estimation  Point estimation involves using a sample statistic to estimate a population parameter. For example, the sample mean is used to estimate the population mean , and the sample proportion estimates the population proportion . A good point estimator should be unbiased (its expected value equals the true parameter) and have low variance (it is precise).  For instance, if we measure the heights of 50 randomly selected adults and find a sample mean of , we use as a point estimate for the population mean height . However, point estimates do not convey uncertainty, which is why confidence intervals are often used to provide a range of plausible values.    Sampling Distributions  A sampling distribution is the probability distribution of a statistic over repeated samples of the same size from a population. For example, if we repeatedly draw samples of size and compute the sample mean , the distribution of values is the sampling distribution of the mean.  The Central Limit Theorem (CLT) states that for large sample sizes, the sampling distribution of the sample mean or proportion is approximately normal, regardless of the population distribution, provided certain conditions are met (e.g., and for proportions). This property underpins hypothesis testing and confidence intervals, allowing us to use the normal distribution for inference.    Hypothesis Testing  Hypothesis testing is a cornerstone of inferential statistics, allowing us to test claims about population parameters using sample data. It involves a structured process to determine whether the data provide sufficient evidence to reject a default claim, called the null hypothesis.  The steps of hypothesis testing are:   Null hypothesis ( ): The default claim, often stating no effect or no difference (e.g., ).  Alternative hypothesis ( ): The claim we seek evidence for (e.g., ).  Choose a significance level (e.g., 0.05), the probability of rejecting when it is true.  Compute a test statistic from the sample data.  Find the p-value or critical region.  Decision: Reject if the p-value is less than ; otherwise, fail to reject . Failing to reject does not prove it is true, only that the evidence is insufficient to reject it.   A common misconception is that the p-value represents the probability that the null hypothesis is true. Instead, it is the probability of observing data as extreme as the sample, assuming is true. A small p-value (e.g., ) suggests strong evidence against , but it does not quantify the probability of or .   Worked Example: Coin Toss   Suppose we suspect a coin is biased because it lands on heads more often. To test if it is fair, we define: where is the probability of heads. We toss the coin 100 times and observe 52 heads, so the sample proportion is:   We set the significance level at (98% confidence). The Central Limit Theorem applies because and , ensuring the sample proportion is approximately normal. The standard error under is: The test statistic is:   For a two-tailed test, the p-value is: Since , we fail to reject . The significance level must be chosen before the experiment to avoid bias (e.g., adjusting to 0.7 after seeing the p-value). There is insufficient evidence to conclude the coin is biased, but this does not prove the coin is fair.   Visualization   The following Python code generates a plot of the standard normal distribution with the test statistic marked and the two-tailed p-value regions shaded:   import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Define the range for the x-axis x = np.linspace(-4, 4, 1000) # Standard normal distribution y = norm.pdf(x, 0, 1) # Test statistic z = 0.4 # Create the plot plt.figure(figsize=(8, 5)) plt.plot(x, y, 'b-', label='Standard Normal Distribution') # Shade the right tail (z > 0.4) x_right = np.linspace(z, 4, 100) plt.fill_between(x_right, norm.pdf(x_right, 0, 1), color='red', alpha=0.3, label='p-value region (two-tailed)') # Shade the left tail (z < -0.4) x_left = np.linspace(-4, -z, 100) plt.fill_between(x_left, norm.pdf(x_left, 0, 1), color='red', alpha=0.3) # Mark the test statistic plt.axvline(z, color='black', linestyle='--', label=f'Z = {z}') plt.axvline(-z, color='black', linestyle='--') plt.title('Standard Normal Distribution with Test Statistic and p-value') plt.xlabel('Z') plt.ylabel('Density') plt.legend() plt.grid(True) plt.show()    Plot of the standard normal distribution with the test statistic marked by dashed lines and the two-tailed p-value regions shaded in red.   Plot of the standard normal distribution with the test statistic and p-value regions shaded.    The plot in shows the test statistic close to the mean, indicating a large p-value and weak evidence against .  Alternatively, we can construct a 98% confidence interval for using the standard error of the proportion, : Since lies within this interval, we fail to reject , consistent with the p-value approach.   Real-World Application: Drug Efficacy   In medicine, hypothesis testing is used to evaluate drug efficacy. Suppose a new drug claims to reduce blood pressure by at least 10 mmHg. We test (no reduction) versus (reduction). Data from 30 patients show a mean reduction of with a sample standard deviation . Using a t-test (since is unknown and is moderate), we compute the t-statistic and p-value to determine if the evidence supports the drug’s efficacy.    Confidence Intervals  A confidence interval (CI) estimates a population parameter with a range of values, associated with a confidence level (e.g., 98%). For the coin toss example, the 98% CI for the true proportion is , meaning we are 98% confident that lies within this range. The confidence level indicates that if we repeated the sampling process many times, 98% of such intervals would contain the true parameter. It does not mean there is a 98% probability that lies in this specific interval.  For a population mean, the CI is: where is the sample mean, is the population standard deviation, is the sample size, and is the critical value (e.g., 2.326 for 98% confidence). If is unknown, the sample standard deviation is used, and for small samples ( ), a t-distribution is applied instead of the normal distribution.  CIs complement hypothesis testing by providing a range of plausible values for the parameter. For example, in the drug efficacy case, a 95% CI for the mean blood pressure reduction might be , suggesting the true effect is likely positive.    Types of Errors  Hypothesis testing carries risks of incorrect decisions. A Type I error occurs when we reject when it is true (false positive), with probability (e.g., 0.02 for a 2% chance). A Type II error occurs when we fail to reject when is true (false negative), with probability . Reducing increases unless the sample size or effect size is increased.  In the coin toss example, a Type I error would be concluding the coin is biased when it is fair. A Type II error would be failing to detect a bias when the coin is biased.   Visualization: Types of Errors   The following table illustrates the possible outcomes of a hypothesis test:   Hypothesis Testing Outcomes     True  False    Reject  Type I Error ( )  Correct (Power: )    Fail to Reject  Correct ( )  Type II Error ( )       Statistical Power and Effect Size  The power of a test is the probability of correctly rejecting a false null hypothesis ( ). Power depends on the sample size, effect size, and significance level . The effect size measures the magnitude of the difference or relationship, such as Cohen’s d for the difference in means or the odds ratio for proportions.  For example, in the drug efficacy example, the effect size might be the mean blood pressure reduction (e.g., 10 mmHg). A larger effect size or sample size increases power, making it easier to detect a true effect. Power analysis determines the sample size needed to achieve a desired power (e.g., 80%) for a given effect size and .    Multiple Testing  When conducting multiple hypothesis tests, the probability of at least one Type I error increases. For example, if 20 tests are performed at , the chance of at least one false positive is approximately . Methods like the Bonferroni correction adjust the significance level (e.g., for tests) to control the overall Type I error rate.  For instance, in a study testing 10 drugs, using per test ensures the overall Type I error rate remains near 0.05. However, this reduces power, so researchers must balance error control and test sensitivity.    Common Statistical Tests  Beyond the z-test, other common tests include:  t-test : Used for small samples or when the population variance is unknown. For example, to test if a new teaching method improves test scores, we collect scores from 20 students, compute the sample mean difference points and standard deviation , and use a t-test to compare against .  Chi-square test : Used for categorical data. For example, a survey of 200 people tests if political affiliation (Party A, Party B, Independent) is independent of age group (young, middle-aged, older) by comparing observed and expected frequencies.  ANOVA : Used to compare means across multiple groups, e.g., testing if different diets affect weight loss.  Non-parametric tests : Used when assumptions like normality are violated, such as the Mann-Whitney U test for comparing two groups or the Kruskal-Wallis test for multiple groups.    Each test relies on specific assumptions. For example, z-tests and t-tests assume approximately normal data (or large samples for the CLT) and independent observations. The chi-square test requires expected frequencies of at least 5 per category, and ANOVA assumes homogeneity of variances across groups. Violating these assumptions may necessitate non-parametric tests.   "
},
{
  "id": "sec-Inferential-Statistics-2-3-1-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#sec-Inferential-Statistics-2-3-1-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Population "
},
{
  "id": "sec-Inferential-Statistics-2-3-2-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#sec-Inferential-Statistics-2-3-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Sample "
},
{
  "id": "sec-Inferential-Statistics-2-3-3-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#sec-Inferential-Statistics-2-3-3-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Parameter "
},
{
  "id": "sec-Inferential-Statistics-2-3-4-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#sec-Inferential-Statistics-2-3-4-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Statistic "
},
{
  "id": "subsec-Hypothesis-Testing-4-1-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#subsec-Hypothesis-Testing-4-1-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Null hypothesis "
},
{
  "id": "subsec-Hypothesis-Testing-4-2-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#subsec-Hypothesis-Testing-4-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Alternative hypothesis "
},
{
  "id": "subsec-Hypothesis-Testing-4-5-1",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#subsec-Hypothesis-Testing-4-5-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "p-value "
},
{
  "id": "fig-p-value",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#fig-p-value",
  "type": "Figure",
  "number": "1.9.1",
  "title": "",
  "body": " Plot of the standard normal distribution with the test statistic marked by dashed lines and the two-tailed p-value regions shaded in red.   Plot of the standard normal distribution with the test statistic and p-value regions shaded.   "
},
{
  "id": "subsec-Types-of-Errors-2",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#subsec-Types-of-Errors-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Type I error Type II error "
},
{
  "id": "table-error-types",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#table-error-types",
  "type": "Table",
  "number": "1.9.2",
  "title": "Hypothesis Testing Outcomes",
  "body": " Hypothesis Testing Outcomes     True  False    Reject  Type I Error ( )  Correct (Power: )    Fail to Reject  Correct ( )  Type II Error ( )    "
},
{
  "id": "subsec-Power-Effect-Size-2",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#subsec-Power-Effect-Size-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "power effect size "
},
{
  "id": "subsec-Other-Tests-2",
  "level": "2",
  "url": "sec-Inferential-Statistics.html#subsec-Other-Tests-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "t-test Chi-square test ANOVA Non-parametric tests "
},
{
  "id": "backmatter-2",
  "level": "1",
  "url": "backmatter-2.html",
  "type": "Colophon",
  "number": "",
  "title": "Colophon",
  "body": " This book was authored in PreTeXt .  "
}
]

var ptx_lunr_idx = lunr(function () {
  this.ref('id')
  this.field('title')
  this.field('body')
  this.metadataWhitelist = ['position']

  ptx_lunr_docs.forEach(function (doc) {
    this.add(doc)
  }, this)
})
