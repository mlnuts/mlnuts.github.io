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
  "body": " Descriptive Statistics   Descriptive statistics summarize key features of a dataset, providing insights into its central tendency, dispersion, and shape. This process, known as Exploratory Data Analysis (EDA) , helps identify patterns and trends before applying advanced statistical methods. Common measures include mean, median, mode, variance, standard deviation, range, quartiles, and visualizations like histograms and boxplots. These tools are essential for understanding data in fields like education, finance, and science.    Measures of Central Tendency  Measures of central tendency describe the \"typical\" value in a dataset.     Mean (Average) : The mean, denoted , is the sum of all data points divided by their count. For a dataset with points, the mean is: where .  Example: For student grades , the mean is:   However, mean of a dataset can be misleading if you have a few ouliers since mean is very senisitve to outliers. For instance, say you have a dataset of income, which is , , , , . Clearly, most of the income is in the area, but the mean of this dataset is , skewed by the outlier. In this case, the median would better represents the typical value in the dataset.     Median : The median is the middle value in a sorted dataset, where of the data lies below and above. For odd , it's the middle value; for even , it's the average of the two middle values.  Example: For (sorted), median = . For , median . For incomes , , , , , median = , robust to the outlier.     Mode : The mode is the most frequent value. A dataset may have no mode, one mode (unimodal), or multiple modes (bimodal or multimodal).  Example: has mode 90. is bimodal . has no mode.     Comparison : Consider incomes , , , , . Mean = , median = , mode = none. The median best reflects the typical income due to the outlier. See for a visual comparison.   Density plot of incomes with mean, median, and no mode.   Density plot showing central tendency measures.       Measures of Dispersion  Dispersion measures how spread out data is around the central tendency.     Variance and Standard Deviation : Variance ( ) measures average squared deviation from the mean; standard deviation ( ) is its square root, in the same units as the data.  For a population: where is the population (true) mean. The data collected from a polulation is called sample. From the sample we can only calculate as estimate of the corresponding population quantities. We define estimate of sample variance by keeping the same divisor as in the true variance definitionor, define with a divisor , which is called an unbiased estimate of variance. where is the sample mean.   Example: For grades , . Population variance: This will give the standard deviation , : Sample variance, on the other hand, will be: , and sample standard deviation .   illustrated the tighter vs. wider spread for a low variance (e.g., ) vs. high variance (e.g., ).   Comparing low and high variance datasets.   Comparison of low and high variance data.        Range and Quartiles : Range = max - min. Quartiles divide sorted data into four parts: Q1 (25th percentile), Q2 (median, 50th), Q3 (75th). Use linear interpolation: position = , where .  Example: For grades , . Median . , . Range . . Outliers: These grades have no outliers.      Distribution Shape   Histogram : Histograms show frequency distributions by grouping data into bins of equal size from min to a bin that includes the max data. So, if you have data from to with a bin size . Then, bins will have , , , till you have exhausted all data. The last bin may extend beyond the data as in the example below.  Example: For grades  , with bin size from to , see .   Histogram of Grades    Bin  Range  Data  Count  Frequency    1    4  0.333    2    3  0.333    3    2  0.222    4    1  0.111     Many computer libraries have histogram plotting routines. For instance was generated from the Python program listed after it. The histogram has been decorated with the mean and median of the data also.   Histogram of grades with mean and median.   Histogram with mean and median lines.    Example Histogram import matplotlib.pyplot as plt import numpy as np data = [70, 72, 75, 75, 80, 82, 85, 93, 95, 100] bins = [70, 80, 90, 100, 110] freq_arr, bins_arr = np.histogram(data, bins) # returns frequency width = bins_arr[1:] - bins_arr[:-1] plt.figure(figsize=(8, 5)) plt.hist(data, bins=bins, edgecolor='black', alpha=0.7) # this is just plt.bar(bins_arr[:-1], freq_arr, width) mean = np.mean(data) median = np.median(data) plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.1f}') plt.axvline(median, color='green', linestyle='-', label=f'Median = {median:.1f}') plt.xlabel('Grade') plt.ylabel('Frequency') plt.title('Histogram of Student Grades') plt.xticks(bins) plt.grid(axis='y', alpha=0.3) plt.legend() plt.savefig('histogram.png') plt.show()   Boxplot : Boxplots show min, Q1, median, Q3, max (whiskers), and outliers (points beyond Q1 IQR or Q3 IQR).  Example: For grades with an outlier , , , , . Outliers: ≤ is above . See .   Boxplot of grades with annotated quartiles and outlier.   Boxplot with one outlier.    Updated boxplot with annotations import matplotlib.pyplot as plt import numpy as np data = [70, 75, 80, 85, 90, 95, 100, 150] plt.figure(figsize=(8, 4)) bp = plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red')) q1, median, q3 = np.percentile(data, [25, 50, 75]) plt.text(q1 - 5, 1.1, 'Q1', ha='right') plt.text(median, 1.1, 'Median', ha='center') plt.text(q3 + 5, 1.1, 'Q3', ha='left') plt.text(150, 1.3, 'Outlier', ha='center') plt.title('Boxplot of Student Grades') plt.xlabel('Grade') plt.grid(True, alpha=0.3) plt.savefig('boxplot.png') plt.show()   Skewness : Skewness tells us about the shape of the distribution, specifically if it's \"tilted\" to one side. In a positively skewed distribution (right skew), the tail on the right side is longer, while in a negatively skewed distribution (left skew), the tail on the left side is longer.  Positive skew example: Imagine a dataset of household incomes: . The income of 1 million is much higher than the others, causing the data to be right-skewed. Most people earn a lower income, but a few very high incomes stretch the right side of the distribution, creating a longer right tail.  Negative skew example: Think of a set of exam scores . If most students score high but a few perform very poorly, the data is left-skewed. The low scores create a long tail on the left side of the distribution.  See for a visual representation.   Histograms comparing a normal distribution and a right-skewed distribution.   Comparison of normal vs. skewed distributions.    To help visualize the difference, here’s a Python code that generates two types of distributions: a normal (symmetrical) one and a right-skewed one. The plot will show how the shapes of these two distributions differ.  Skewness Visualization import matplotlib.pyplot as plt import numpy as np from scipy.stats import norm, skewnorm np.random.seed(42) normal_data = np.random.normal(50, 10, 1000) # Normal distribution skewed_data = np.random.exponential(20000, 1000) # Right-skewed distribution fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # normal distribution ax1.hist(normal_data, bins=30, edgecolor='black', alpha=0.7) ax1.set_title('Normal Distribution') ax1.set_xlabel('Value') ax1.set_ylabel('Frequency') #Plot right-skewed distribution ax2.hist(skewed_data, bins=30, edgecolor='black', alpha=0.7) ax2.set_title('Right-Skewed Distribution') ax2.set_xlabel('Value') ax2.set_ylabel('Frequency') plt.tight_layout() plt.savefig('skewness-comparison.png') plt.show()    Kurtosis: Checking Out the Tails   After digging into mean, median, mode, and skewness, let’s talk kurtosis . This stat zooms in on the \"tailedness\" of our data—how often we see wild outliers compared to a normal distribution. It’s like checking the edges of our data’s shape to see if they’re loaded with extreme values or totally chill.    Types of Kurtosis  Kurtosis comes in three flavors:   Mesokurtic : Matches a normal distribution. Not too many outliers, not too few. It’s the just-right vibe.  Leptokurtic : Sharp peak, heavy tails. Think lots of outliers, like stock prices during a market rollercoaster.  Platykurtic : Flatter peak, light tails. Fewer outliers, like steady daily temps.     How’s It Calculated?  Kurtosis measures how much data hangs out in the tails. The formula for excess kurtosis (comparing to a normal distribution) is:     Here, is the number of data points, is each data point, is the mean, and is the standard deviation. Don’t stress the math—Python will handle it in a bit!    Real-World Examples  Let's connect kurtosis to mean, median, mode, and skewness:   Leptokurtic Example : Take stock returns: {5, 7, 8, 8, 10, 12, 12, 12, 12, 50} . That 50 is a massive outlier, bulking up the tails (like skewness, but focused on extremes). This is leptokurtic—expect some crazy swings.  Platykurtic Example : Now daily temperatures: {30, 32, 33, 34, 35, 36, 38} . No wild outliers, just values chilling around the mean and median. This is platykurtic—nice and calm.   While skewness shows if our data’s lopsided, kurtosis tells us if outliers are stealing the show.    Seeing Kurtosis in Action  Picture kurtosis with histograms (like we used for mean and skewness):   Leptokurtic : Sharp peak, chunky tails (lots of outliers).  Platykurtic : Flatter top, skinny tails (few outliers).  Mesokurtic : Classic bell curve, balanced tails.   Check out this chart to see the difference:   Kurtosis Comparison: The top subplot shows the full distributions, with leptokurtic (sharp peak, heavy tails with more outliers, like stock returns), mesokurtic (normal distribution, balanced tails), and platykurtic (flat peak, light tails with fewer outliers, like temperatures). The bottom subplot zooms in on the right tail, showing how leptokurtic tails decay slower (higher density at large values) compared to mesokurtic and platykurtic tails, which drop off faster.    import numpy as np import matplotlib.pyplot as plt from scipy.stats import kurtosis, t, norm, uniform # Data for kurtosis calculation stock_data = [5, 7, 8, 8, 10, 12, 12, 12, 12, 50] # Leptokurtic temp_data = [30, 32, 33, 34, 35, 36, 38] # Platykurtic # Calculate kurtosis kurt_stock = kurtosis(stock_data, fisher=True) kurt_temp = kurtosis(temp_data, fisher=True) print(f\"Stock Returns Kurtosis: {kurt_stock:.2f} (Leptokurtic)\") print(f\"Temperature Kurtosis: {kurt_temp:.2f} (Platykurtic)\") # Generate data for plotting distributions x = np.linspace(-10, 10, 200) # Wider range to show tails # Leptokurtic: Student's t-distribution (df=3 for heavy tails) lepto = t.pdf(x, df=3) * 1.2 # Scale for visibility # Mesokurtic: Normal distribution meso = norm.pdf(x) # Platykurtic: Uniform-like distribution (approximated) platy = uniform.pdf(x, loc=-2, scale=4) * 0.8 # Flat, light tails # Create figure with two subplots fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # Full distribution plot (top subplot) ax1.plot(x, lepto, label='Leptokurtic (Heavy Tails)', color='#FF5733') ax1.plot(x, meso, label='Mesokurtic (Normal)', color='#33FF57') ax1.plot(x, platy, label='Platykurtic (Light Tails)', color='#3357FF') ax1.set_title('Kurtosis Comparison: Full Distribution') ax1.set_xlabel('Value') ax1.set_ylabel('Density') ax1.legend() ax1.grid(True) # Tail-focused plot (bottom subplot) ax2.plot(x, lepto, label='Leptokurtic (Heavy Tails)', color='#FF5733') ax2.plot(x, meso, label='Mesokurtic (Normal)', color='#33FF57') ax2.plot(x, platy, label='Platykurtic (Light Tails)', color='#3357FF') ax2.set_title('Kurtosis Comparison: Right Tail Focus') ax2.set_xlabel('Value') ax2.set_ylabel('Density') ax2.set_xlim(3, 10) # Focus on right tail ax2.set_ylim(0, 0.05) # Zoom in on low density ax2.legend() ax2.grid(True) # Adjust layout and save plt.tight_layout() plt.savefig('kurtosis.png', dpi=300, bbox_inches='tight') # Save both subplots plt.show()    Calculating Kurtosis with Python  Since we’re all about descriptive stats, let’s compute kurtosis with Python (more tools coming up later). Here’s a script for our stock returns example:  from scipy.stats import kurtosis import numpy as np # Stock returns data (leptokurtic) data = [5, 7, 8, 8, 10, 12, 12, 12, 12, 50] # Calculate excess kurtosis kurt = kurtosis(data, fisher=True) print(f\"Excess Kurtosis: {kurt:.2f}\") # Positive value means leptokurtic (heavy tails)!  Run this, and you’ll see a positive kurtosis, confirming big outliers in our stock returns. Try the temperature data {30, 32, 33, 34, 35, 36, 38} for a negative kurtosis and that platykurtic vibe.    Why Kurtosis Matters  Kurtosis wraps up our descriptive stats crew—mean, median, mode, and skewness. It’s like a heads-up about outliers. High kurtosis (leptokurtic) yells “watch out for big swings!”—think risky stocks. Low kurtosis (platykurtic) says “smooth sailing”—like predictable weather. It’s another piece of your data’s story.      Conclusion  In summary, descriptive statistics provide essential tools for understanding and summarizing datasets, helping to reveal underlying patterns and trends. By examining measures of central tendency (mean, median, mode), we can identify the \"typical\" values of a dataset, while measures of dispersion (variance, standard deviation, range, and quartiles) show how spread out the data is. Visualizations like histograms, boxplots, and density plots enhance this understanding by allowing us to visually inspect the distribution and shape of the data. As we move forward, the next section will explore the tools available for calculating these descriptive statistics, giving us the means to automate and refine these analyses in practice.   "
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
  "id": "subsec-Dispersion-3-1-3",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#subsec-Dispersion-3-1-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "standard deviation "
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
  "body": " Histogram of Grades    Bin  Range  Data  Count  Frequency    1    4  0.333    2    3  0.333    3    2  0.222    4    1  0.111    "
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
  "body": " Histograms comparing a normal distribution and a right-skewed distribution.   Comparison of normal vs. skewed distributions.   "
},
{
  "id": "kurtosis-chart",
  "level": "2",
  "url": "sec-Descriptive-Statistics.html#kurtosis-chart",
  "type": "Figure",
  "number": "1.1.7",
  "title": "",
  "body": " Kurtosis Comparison: The top subplot shows the full distributions, with leptokurtic (sharp peak, heavy tails with more outliers, like stock returns), mesokurtic (normal distribution, balanced tails), and platykurtic (flat peak, light tails with fewer outliers, like temperatures). The bottom subplot zooms in on the right tail, showing how leptokurtic tails decay slower (higher density at large values) compared to mesokurtic and platykurtic tails, which drop off faster.   "
},
{
  "id": "sec-useful-descriptive-statistics-tools",
  "level": "1",
  "url": "sec-useful-descriptive-statistics-tools.html",
  "type": "Section",
  "number": "1.2",
  "title": "Computation and Visualization Tools",
  "body": " Computation and Visualization Tools   Exploratory Data Analysis (EDA) is a critical step in understanding your data before applying advanced techniques like machine learning. It involves summarizing the main characteristics of a dataset, often using visual methods, to uncover patterns, spot anomalies, test hypotheses, and check assumptions.  In this section, we focus on Python-based tools that enable efficient and effective data analysis, tailored for machine learning workflows. While languages like R are powerful for statistics, we emphasize Python due to its widespread use in data science and machine learning communities. Key tools include NumPy for numerical computations, Pandas for data manipulation, and Matplotlib\/Seaborn for visualization. These libraries integrate seamlessly, allowing you to load, clean, analyze, and visualize data in a streamlined manner.  A typical EDA workflow includes: loading data, inspecting its structure, handling missing values, computing summary statistics, exploring distributions, and visualizing relationships. Using Jupyter notebooks ensures reproducibility and documentation of your analysis.    The Power of NumPy and SciPy  While Python lists and loops are flexible, they are slow for large-scale numerical work. The NumPy library provides fast, memory-efficient arrays and vectorized operations. These make Python competitive with lower-level languages for scientific and machine learning tasks. The SciPy library builds on NumPy, adding advanced tools for statistics, optimization, and more.  Let us begin with a simple example: computing descriptive statisticsand plotting a histogram ( ) of rolls of a six-sided die. Notice that NumPy computes the mean and standard deviation in a single line. This would require explicit loops in plain Python.  import numpy as np import matplotlib.pyplot as plt #Set Seed for reproducibility np.random.seed(seed=42) # Generate 10,000 simulated die rolls rolls = np.random.randint(1, 7, size=50) mean = np.mean(rolls) std = np.std(rolls) print(f\"mean = {mean}, std = {std}\") #(np.float64(3.4999), np.float64(1.7086251753968744)) fig, ax = plt.subplots() ax.hist(rolls, bins=6, color=\"b\", alpha=0.25) plt.xlabel('Face Value') plt.ylabel('Frequency') plt.title('Histogram of Die Rolls') plt.savefig(\"np-die-histogram.png\") plt.show()   Histogram of simulated die rolls using NumPy.   Histogram of simulated die rolls using NumPy.    The real strength of NumPy comes from vectorization , which eliminates explicit loops.  import time N = 10_000_000 x = np.random.rand(N) # Vectorized: compute sum of squares t0 = time.time() s1 = np.sum(x**2) t1 = time.time() # Loop version s2 = 0.0 for xi in x: s2 += xi**2 t2 = time.time() (t1 - t0, t2 - t1) # compare runtimes  The vectorized NumPy version runs in milliseconds, while the loop can take seconds. This difference is crucial in machine learning, where datasets often have millions of entries.   Vectorized NumPy operation vs. Python loop runtime.    NumPy also includes a linalg package for linear algebra. The following code snippet demonstrates how NumPy solves systems of equations and computes eigenvalues—core operations in data science, physics, and engineering.  A = np.array([[3, 1], [1, 2]]) b = np.array([9, 8]) # Solve Ax = b x = np.linalg.solve(A, b) # Eigenvalues and eigenvectors e_vals, e_vecs = np.linalg.eig(A) (x, e_vals)]  For advanced tasks, SciPy extends NumPy. For example, hypothesis testing or optimization. SciPy provides one-line solutions for statistical inference and numerical optimization.  from scipy import stats, optimize # Hypothesis test: is sample mean = 0? sample = np.random.normal(0, 1, size=100) t_stat, p_val = stats.ttest_1samp(sample, 0) print(f\"t-statistic = {t_stat}, p-value={p_val}\") # t-statistic = 0.8998073723146639, p-value=0.37040629150553495  from scipy import stats, optimize # Optimization: minimize f(x) = (x-3)^2 f = lambda x: (x-3)**2 res = optimize.minimize(f, x0=0) # [2.99999998]  Together, NumPy and SciPy form the numerical backbone of Python’s scientific ecosystem.    Pandas: Data Manipulation and Analysis   Pandas is a powerful, flexible library for data manipulation and analysis, built on NumPy. Its core data structures are:  Series : A one-dimensional labeled array for sequences of data.  DataFrame : A two-dimensional labeled table, similar to a spreadsheet or SQL table, ideal for tabular data.    Pandas is designed for cleaning, transforming, analyzing, and visualizing data. It supports multiple file formats (CSV, Excel, JSON, SQL) and integrates with NumPy, Matplotlib, Seaborn, and Scikit-learn, making it a cornerstone for EDA in machine learning.    Why Use Pandas?   Handles structured data efficiently (e.g., tabular data).  Supports data cleaning (missing values, duplicates, outliers).  Enables grouping, aggregation, and statistical summaries.  Scales to large datasets with optimized performance.     EDA Workflow with Pandas:   Load data ( pd.read_csv() , pd.read_excel() ).  Inspect structure ( head() , info() , describe() ).  Clean data (handle missing values, remove duplicates).  Compute statistics and explore distributions.  Visualize (integrate with Matplotlib\/Seaborn).     Example: Analyzing Student Data Let’s use a realistic dataset of student scores, including a missing value, loaded from a CSV file.  Creating and loading sample student data import pandas as pd # Create sample CSV data (in practice, load from disk) data = \"\"\"Name,Age,Score,Passed Alice,25,85.5,True Bob,30,90.0,True Carol,27,88.0,True Dave,22,76.5,False Eve,28,,True\"\"\" with open('students.csv', 'w') as f: f.write(data) # Load data df = pd.read_csv('students.csv') print(df)  Output as a table:   Student DataFrame    Name  Age  Score  Passed    Alice  25  85.5  True    Bob  30  90.0  True    Carol  27  88.0  True    Dave  22  76.5  False    Eve  28  NaN  True     Inspect the data using common Pandas methods:  Inspecting DataFrame # Inspect data print(\"First 3 rows:\") print(df.head(3)) print(\"\\nLast 2 rows:\") print(df.tail(2)) print(\"\\nShape:\", df.shape) # (5, 4) print(\"\\nColumns:\", df.columns.tolist()) print(\"\\nInfo:\") print(df.info()) print(\"\\nDescriptive Statistics:\") print(df.describe())  Output of df.describe() :   Descriptive Statistics from df.describe()     Age  Score    count  5.000000  4.000000    mean  26.400000  85.000000    std  3.209361  5.958188    min  22.000000  76.500000    25%  24.250000  83.250000    50%  26.000000  86.750000    75%  27.750000  88.500000    max  30.000000  90.000000     Clean and transform the data (e.g., handle missing values, filter, add columns, group, sort):  Data cleaning and transformation # Handle missing values print(\"Missing values:\\n\", df.isnull()) df['Score'] = df['Score'].fillna(df['Score'].mean()) # Fill NaN with mean # Filter rows high_scorers = df[df['Score'] > 85] print(\"\\nHigh scorers:\\n\", high_scorers) # Add new column df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C') print(\"\\nDataFrame with Grade:\\n\", df) # Group and aggregate grouped = df.groupby('Passed')['Score'].agg(['mean', 'count']) print(\"\\nGrouped by Passed:\\n\", grouped) # Sort by Score df_sorted = df.sort_values(by='Score', ascending=False) print(\"\\nSorted by Score:\\n\", df_sorted) # Chain operations result = df[df['Age'] > 25][['Name', 'Score']].sort_values(by='Score') print(\"\\nChained operations (Age > 25, select columns, sort):\\n\", result)  Visualize the score distribution:   Histogram of student scores using Pandas and Matplotlib.   Histogram from Pandas DataFrame.    Generating histogram from Pandas import matplotlib.pyplot as plt import pandas as pd # Assuming df from previous code df['Score'].hist(bins=5, edgecolor='black', alpha=0.7) plt.xlabel('Score') plt.ylabel('Frequency') plt.title('Distribution of Student Scores') plt.grid(True, alpha=0.3) plt.savefig('pandas-histogram.png', dpi=300) plt.show()  For further learning, explore Python for Data Analysis by Wes McKinney (free online) and Kaggle’s Pandas course .    Visualization with Matplotlib and Seaborn  Visualization is a cornerstone of EDA, making patterns and relationships in data intuitive. Matplotlib provides customizable, low-level plotting, while Seaborn, built on Matplotlib, offers high-level statistical visualizations with attractive defaults.   Matplotlib Key Features:   Flexible plots: histograms, boxplots, scatter plots, line plots.  Customizable axes, labels, and styles.  Integration with Pandas for direct plotting.     Seaborn Advantages:   Statistical plots: histplot with KDE, boxplot, pairplot for correlations.  Attractive themes and color palettes.  Simplified syntax for complex visualizations.     Example: Visualizing Student Data Using the student DataFrame, create a histogram and boxplot with Matplotlib, and a histplot with KDE and pairplot with Seaborn.   Matplotlib histogram and boxplot of student scores.   Matplotlib plots from Pandas.    Matplotlib histogram and boxplot import matplotlib.pyplot as plt import pandas as pd # Assuming df from previous code fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) df['Score'].hist(bins=5, ax=ax1, edgecolor='black', alpha=0.7) ax1.set_title('Histogram of Scores') ax1.set_xlabel('Score') ax1.set_ylabel('Frequency') ax1.grid(True, alpha=0.3) df.boxplot(column='Score', ax=ax2) ax2.set_title('Boxplot of Scores') ax2.set_ylabel('Score') ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('matplotlib-plots.png', dpi=300) plt.show()   Real-World Example: Load a larger dataset (e.g., from Kaggle) and visualize distributions and correlations.  EDA with a larger dataset import pandas as pd import seaborn as sns import matplotlib.pyplot as plt # Sample larger dataset (simulated for book) np.random.seed(42) n = 100 data = pd.DataFrame({ 'Age': np.random.normal(25, 5, n), 'Score': np.random.normal(85, 10, n), 'Hours_Studied': np.random.normal(20, 5, n) }) data['Score'] = data['Score'].clip(0, 100) # Ensure valid scores # Basic EDA print(data.describe()) print(\"\\nMissing values:\\n\", data.isnull().sum()) # Correlation matrix print(\"\\nCorrelation matrix:\\n\", data.corr()) # Visualization plt.figure(figsize=(10, 4)) plt.subplot(1, 2, 1) sns.histplot(data['Score'], kde=True) plt.title('Distribution of Scores') plt.subplot(1, 2, 2) sns.scatterplot(x='Hours_Studied', y='Score', data=data) plt.title('Score vs. Hours Studied') plt.tight_layout() plt.savefig('.\/images\/essential-probability-and-statistics\/eda-large-dataset.png', dpi=300) plt.show() # Pairplot sns.pairplot(data, diag_kind='kde') plt.savefig('.\/images\/essential-probability-and-statistics\/eda-pairplot.png', dpi=300) plt.show()   EDA on a larger dataset: histogram and scatter plot.   EDA visualizations for larger dataset.     Pairplot showing relationships in larger dataset.   Pairplot for larger dataset.      NumPy, Pandas, Matplotlib, and Seaborn form a powerful toolkit for EDA. Start with NumPy for numerical operations, use Pandas for data manipulation and cleaning, and leverage Matplotlib\/Seaborn for insightful visualizations. Practice with real datasets (e.g., from Kaggle) in Jupyter notebooks to build skills. For advanced machine learning pipelines, you can explore TensorFlow’s Data API later, but mastering these foundational tools is key for beginners. Resources like Python for Data Analysis and Kaggle’s Pandas course offer hands-on learning.   "
},
{
  "id": "fig-np-die-histogram",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-np-die-histogram",
  "type": "Figure",
  "number": "1.2.1",
  "title": "",
  "body": " Histogram of simulated die rolls using NumPy.   Histogram of simulated die rolls using NumPy.   "
},
{
  "id": "sec-useful-descriptive-statistics-tools-3-6",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#sec-useful-descriptive-statistics-tools-3-6",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "vectorization "
},
{
  "id": "fig-np-vectorization_timing",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-np-vectorization_timing",
  "type": "Figure",
  "number": "1.2.2",
  "title": "",
  "body": " Vectorized NumPy operation vs. Python loop runtime.   "
},
{
  "id": "tab-students-dataframe",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#tab-students-dataframe",
  "type": "Table",
  "number": "1.2.3",
  "title": "Student DataFrame",
  "body": " Student DataFrame    Name  Age  Score  Passed    Alice  25  85.5  True    Bob  30  90.0  True    Carol  27  88.0  True    Dave  22  76.5  False    Eve  28  NaN  True    "
},
{
  "id": "tab-students-describe",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#tab-students-describe",
  "type": "Table",
  "number": "1.2.4",
  "title": "Descriptive Statistics from df.describe()",
  "body": " Descriptive Statistics from df.describe()     Age  Score    count  5.000000  4.000000    mean  26.400000  85.000000    std  3.209361  5.958188    min  22.000000  76.500000    25%  24.250000  83.250000    50%  26.000000  86.750000    75%  27.750000  88.500000    max  30.000000  90.000000    "
},
{
  "id": "fig-pandas-histogram",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-pandas-histogram",
  "type": "Figure",
  "number": "1.2.5",
  "title": "",
  "body": " Histogram of student scores using Pandas and Matplotlib.   Histogram from Pandas DataFrame.   "
},
{
  "id": "fig-matplotlib-plots",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-matplotlib-plots",
  "type": "Figure",
  "number": "1.2.6",
  "title": "",
  "body": " Matplotlib histogram and boxplot of student scores.   Matplotlib plots from Pandas.   "
},
{
  "id": "fig-eda-large-dataset",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-eda-large-dataset",
  "type": "Figure",
  "number": "1.2.7",
  "title": "",
  "body": " EDA on a larger dataset: histogram and scatter plot.   EDA visualizations for larger dataset.   "
},
{
  "id": "fig-eda-pairplot",
  "level": "2",
  "url": "sec-useful-descriptive-statistics-tools.html#fig-eda-pairplot",
  "type": "Figure",
  "number": "1.2.8",
  "title": "",
  "body": " Pairplot showing relationships in larger dataset.   Pairplot for larger dataset.   "
},
{
  "id": "sec-Numerical-and-Categorical-Data",
  "level": "1",
  "url": "sec-Numerical-and-Categorical-Data.html",
  "type": "Section",
  "number": "1.3",
  "title": "Numerical and Categorical Data",
  "body": " Numerical and Categorical Data   When analyzing data, it is essential to understand the type of data you are working with and how to safely convert it into a numerical representation for processing by machine learning models. There are three fundamental types:  Categorical Data : Data that represents discrete groups or labels with no inherent order.  Ordinal Data : A type of categorical data with a defined, meaningful order.  Numerical Data : Data that represents quantities and can be measured and compared numerically.       Categorical Data and One-Hot Encoding   Categorical Data represents discrete groups or labels with no inherent order between the values. For example, a variable named \"Color\" can take values from {\"red\", \"blue\", \"green\"}. Another variable may be \"Animal\" with values from {\"cat\", \"dog\", \"parrot\"}. In Python's Pandas library, you can store this as an object type or, for efficiency and clarity, convert it to a category type.  import pandas as pd colors = pd.Series([\"red\", \"blue\", \"red\", \"green\"], dtype=\"category\") print(colors)  0 red 1 blue 2 red 3 green dtype: category Categories (3, object): ['blue', 'green', 'red']  Many machine learning algorithms require a numerical representation of these values. A common method for this is one-hot encoding .  Suppose we have 3 unique colors. We represent each value with a 3-dimensional vector with a \"1\" in one position and \"0\" in the others:    Example of One-Hot Encoding     Sample 1  Sample 2  Sample 3  Sample 4    Original Data  red  blue  red  green    Color_red  1  0  1  0    Color_blue  0  1  0  0    Color_green  0  0  0  1     The main limitation of one-hot encoding is that the number of dimensions equals the number of categories. If there are thousands of unique categories (like words in a language), this becomes inefficient. In such cases, embeddings are preferred.     Ordinal Data and Safe Encoding   Ordinal Data is a type of categorical data with a defined order. For example, a variable for clothing size may have the order small < medium < large, or a satisfaction rating may be low < medium < high. The order matters, but the numeric difference between categories is not meaningful.  sizes = pd.Series([\"medium\", \"small\", \"large\", \"small\"], dtype=pd.CategoricalDtype(categories=[\"small\", \"medium\", \"large\"], ordered=True)) print(sizes)  0 medium 1 small 2 large 3 small dtype: category Categories (3, object): ['small' < 'medium' < 'large']  If you map ordinal values to integers naively, you risk misleading the model. For instance, \"High School\" → 1, \"Bachelor's\" → 2, \"Master's\" → 3, \"PhD\" → 4 implies equal numeric gaps, which isn’t true in reality.  Safer strategies:   Integer Encoding : Map to integers, but use with models (like decision trees) that care about order, not differences.  Binning : Collapse many levels into broader, meaningful groups.  Embeddings : In deep learning, treat them like tokens and let the model learn relationships.  Avoid One-Hot : It removes ordering information completely.      Numerical Data: Discrete vs Continuous   Numerical Data represents measurable quantities. These can be:   Discrete : Countable items (e.g., number of rooms in a house: 3, 4, 5).  Continuous : Measurable values with potentially infinite precision (e.g., height, weight, price).      Data Type Summary  This table summarizes the properties and common encoding methods for categorical, ordinal, and numerical data.   Comparison of Data Types    Type  Meaningful Order  Meaningful Interval  Encoding Method  Examples    Categorical  No  No  One-Hot Encoding  Colors, Animals, Cities    Ordinal  Yes  No  Integer Encoding, Binning  Ratings, Clothing Sizes    Numerical  Yes  Yes  Scaling \/ Normalization  Height, Weight, Price      "
},
{
  "id": "sec-Numerical-and-Categorical-Data-2-1",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#sec-Numerical-and-Categorical-Data-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Categorical Data Ordinal Data Numerical Data "
},
{
  "id": "subsec-Categorical-Data-2",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Categorical-Data-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Categorical Data "
},
{
  "id": "subsec-Categorical-Data-5",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Categorical-Data-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "one-hot encoding "
},
{
  "id": "tab-one-hot-encoding-color",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#tab-one-hot-encoding-color",
  "type": "Table",
  "number": "1.3.1",
  "title": "Example of One-Hot Encoding",
  "body": " Example of One-Hot Encoding     Sample 1  Sample 2  Sample 3  Sample 4    Original Data  red  blue  red  green    Color_red  1  0  1  0    Color_blue  0  1  0  0    Color_green  0  0  0  1    "
},
{
  "id": "subsec-Categorical-Data-8",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Categorical-Data-8",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "embeddings "
},
{
  "id": "subsec-Ordinal-Data-2",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Ordinal-Data-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Ordinal Data "
},
{
  "id": "subsec-Numerical-Data-2",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Numerical-Data-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Numerical Data "
},
{
  "id": "subsec-Numerical-Data-3-1-1",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Numerical-Data-3-1-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Discrete "
},
{
  "id": "subsec-Numerical-Data-3-2-1",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#subsec-Numerical-Data-3-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Continuous "
},
{
  "id": "table-data-type-comparison",
  "level": "2",
  "url": "sec-Numerical-and-Categorical-Data.html#table-data-type-comparison",
  "type": "Table",
  "number": "1.3.2",
  "title": "Comparison of Data Types",
  "body": " Comparison of Data Types    Type  Meaningful Order  Meaningful Interval  Encoding Method  Examples    Categorical  No  No  One-Hot Encoding  Colors, Animals, Cities    Ordinal  Yes  No  Integer Encoding, Binning  Ratings, Clothing Sizes    Numerical  Yes  Yes  Scaling \/ Normalization  Height, Weight, Price    "
},
{
  "id": "sec-Basic-Probability",
  "level": "1",
  "url": "sec-Basic-Probability.html",
  "type": "Section",
  "number": "1.4",
  "title": "Basic Probability for Machine Learning",
  "body": " Basic Probability for Machine Learning   Probability is the backbone of machine learning, helping us model uncertainty in data, predictions, and outcomes. In machine learning, probability underpins tasks like classification (e.g., predicting labels), evaluating model confidence, and handling noisy data. This section introduces probability concepts such as sample spaces, events, and axioms—and connects them to practical machine learning applications using Python.  An event is a specific outcome or set of outcomes from an experiment, represented as a set. For a coin toss, \"heads\" is , \"tails\" is , and \"heads or tails\" is . Each trial answers whether an event occurred (yes\/no). For a die roll yielding , events like or occur if they include . Sets allow combining events via union ( ) or intersection ( ), such as .    Axiomatic View of Probability  In 1933, Andrey Kolmogorov formalized probability with three axioms, providing a mathematical framework. Think of these as rules that ensure probabilities make sense, like ensuring a weather forecast never predicts negative rain or more than chance.   Sample Space : The set of all possible outcomes. For a six-sided die, . For a student passing an exam, .   Event Space : All possible subsets of , including the empty set (impossible event) and (event certain to happen). For a coin toss ( ), . With outcomes, has events.  With every event we can identify its complementary event or complement . The complementary event includes all other possibilities that excluded the event . Thus, in a six-sided die, say . Its complement will be the event . Clearly, their intersection will be the null event.    Probability Measure : Assigns a number to each event , representing its likelihood. For example, for a fair die, . You only need for elementary events, which are the elements of , each taken as one event since you can use the Aditivity law below to get probability of any event in the entire event space .  The probability space is the triplet . Kolmogorov’s axioms are:  Non-negativity : Although, negative probabilities may be taunting science fiction scenarios, they do not make sense in the probailities we deal with everyday. We require that   Normalization : Since the event has all possible elementary events, except the null event, every possible event is inculded in . Therefor, is called event of certainty. Thus, , ensures total certainty for event . This is why although frequencies are proportional to probabilities, we need to divide them by total number of trials to convert them into normalized probabilities.  Additivity : If two events that are disjoint, i.e., there is no situation in which both events can occur together, i.e, ), the probability of either of them occuring, i.e., their union, must be sum of the probabilities of the events occuring separately. . Of course, if there was an overlap, between the two events, then, we would need to subtrat the overlap part since that would have been counted twice, once in and another time in .     Derived results:    illustrated the union formula in which two events (odd numbers), have an overlap. Theoretically, Let's see how it plays out in the rolls simulation shown in . The left region counts rolls of (in only), the right region counts rolls of (in only), the overlap counts rolls of (in ), and a total of counts for . From these counts we estimate probabilities. The counts , , and in total rolls . From these counts we get Now let's check    Venn diagram illustrating events (odd numbers) and for 1,000 simulated rolls of a fair six-sided die. The code of the simulation is given below.   Venn diagram of intersecting events.    Simulating die roll for union probability # --- DIE ROLL VENN DIAGRAM --- import numpy as np from matplotlib_venn import venn2 import matplotlib.pyplot as plt np.random.seed(42) n_trials = 1000 rolls = np.random.randint(1, 7, n_trials) # Events e1 = np.isin(rolls, [1, 3, 5]) # Odd numbers e2 = np.isin(rolls, [3, 5, 6]) # 3,5,6 e1_only = np.sum(e1 \\amp; ~e2) e2_only = np.sum(e2 \\amp; ~e1) both = np.sum(e1 \\amp; e2) # Venn diagram plt.figure(figsize=(6, 4)) venn2(subsets=(e1_only, e2_only, both), set_labels=('E1 (Odd)', 'E2 (3,5,6)')) plt.title('Venn Diagram of Die Roll Events') plt.savefig('venn-diagram-E1-E2.png', dpi=300) plt.show() # Probabilities p_e1 = np.mean(e1) p_e2 = np.mean(e2) p_inter = np.mean(e1 \\amp; e2) p_union = np.mean(e1 | e2) print(f\"P(E1): {p_e1:.3f}, P(E2): {p_e2:.3f}, P(E1 ∩ E2): {p_inter:.3f}, P(E1 ∪ E2): {p_union:.3f}\") # --- END CODE ---    Sum and Product Rules for Probability  The sum and product rules are two cornerstones of probability theory. They allow us to compute probabilities of unions and intersections of events, which are essential in many machine learning applications, from estimating marginal distributions to building classifiers.   Sum Rule : The probability of the union of two events and is given by: The subtraction of avoids double-counting the overlap between the two events. This can be clearly understood using a Venn diagram.   Illustration of the sum rule. The probability of is the sum of the shaded areas of and , minus the overlapping region that would otherwise be counted twice.   Venn diagram showing the sum rule for two overlapping events.     Product Rule : The joint probability of two events and is: where is the conditional probability of given . This factorization is the basis for probabilistic models such as Naive Bayes.    Conditional Probability and Independence   Conditional Probability : The probability of an event given that event has occurred is written as . Intuitively, this represents updating our belief about once we know that is true. For instance, the probability that a student passes an exam may be different depending on whether we already know they studied more than 20 hours.  Probability of King if the Card is known to be Spade  Imagine we have a standard deck of 52 playing cards. Event = \"the card is a King\". Event = \"the card is a Spade\".  The unconditional probability of drawing a King is . But suppose we are told that the card drawn is a Spade. Now, the sample space is only 13 Spade cards. Out of these, exactly one is a King (the King of Spades).  Thus the conditional probability is , which differs from . This illustrates how new information (that the card is a Spade) changes the probability of .    Independence : Two events and are independent if the occurrence of one does not affect the probability of the other. Formally, the eventa and are independent if . Equivalently, Independence means that knowing whether occurred does not provide any new information about .  Tosses of a fair coin are independent  Suppose we toss a fair coin twice. Let event be \"the first toss is Heads\" and event be \"the second toss is Heads.\" The sample space is .  We have , , and . Therefore, , which means the events are independent.  Now let event be \"at least one toss is Heads.\" Then , , and . But , so . Thus, and are not independent.    Example (Student Study Data):   Suppose we record whether students studied more than 20 hours (High Study) and whether they passed an exam (Pass). Using the dataset below, we will estimate probabilities empirically:    : overall fraction of students who passed,     : fraction who studied more than 20 hours,     : fraction who both passed and studied more than 20 hours.     : Just look at the students who studied more than 20 hours (High Study), what fraction Passed the exam (Pass).   The sum rule gives . The product rule verifies that .  Sum and product rules with student data import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns np.random.seed(42) data = pd.DataFrame({ \"Hours_Studied\": np.random.normal(20, 5, 200).clip(0, 40), \"Passed\": np.random.binomial(1, 0.7, 200) }) data[\"High_Study\"] = data[\"Hours_Studied\"] > 20 # Probabilities p_pass = data[\"Passed\"].mean() p_high = data[\"High_Study\"].mean() p_both = ((data[\"Passed\"] == 1) \\amp; (data[\"High_Study\"] == True)).mean() p_union = p_pass + p_high - p_both p_conditional = p_both \/ p_high print(f\"P(Pass) = {p_pass:.3f}\") print(f\"P(High Study) = {p_high:.3f}\") print(f\"P(Pass ∩ High Study) = {p_both:.3f}\") print(f\"P(Pass ∪ High Study) = {p_union:.3f}\") print(f\"P(Pass|High Study) * P(High Study) = {p_conditional:.3f} * {p_high:.3f} = {p_conditional * p_high:.3f}\") # Visualize joint distribution joint_table = pd.crosstab(data[\"Passed\"], data[\"High_Study\"], normalize=\"all\") sns.heatmap(joint_table, annot=True, cmap=\"Blues\", fmt=\".3f\") plt.xlabel(\"High Study Hours (>20)\") plt.ylabel(\"Pass (0=No, 1=Yes)\") plt.title(\"Joint Probability of Pass and High Study Hours\") plt.savefig(\"joint-probability-heatmap.png\", dpi=300) plt.show()   Joint probability table of passing and high study hours. Each cell represents . Row and column sums recover marginals ( and ), illustrating the sum rule. The product rule is verified by comparing the joint probability with .   Heatmap of joint probabilities for pass and study hours.    Let's see how we can read the heatmap in . I will read by the rows first. Let us do notation Then the joint probabilities in the heatmap are: First we will check that the total probaility is actually . Great! Now, let us the probability of Passing the exam, whether you studied or not. This is asking for probability of regardless of the values; So, we need to sum ovr the values. That's pretty high chance of passing. The chance of not passing will just be it's complement. Now, what is the chance that a random student has actullay studied hard regardless of whatever happened in the test. That will be , which we can get from the joint probabilities by summing over the values while keeping the values to . That would mean the probability of a random student having not studied excessively is Now, how about the conditional probabilities? From this heatmap, it is easy to get conditional probabilities. For instance, suppose we want to know Had you studied more than 20 hrs , wat would be your change of passing  Wow, this data produced by simulation showing us that had you studied excessively, your chance of passing the exam actually went down from to . Wild! Okay, now your turn of computing sme other conditional probabilities.    Probability Distributions  Probability distributions describe how probabilities are distributed over outcomes. In machine learning, distributions model data or predictions.   Bernoulli Distribution : Models a binary outcome (e.g., pass\/fail) with probability . For passing an exam, , .   Binomial Distribution : Counts successes in independent Bernoulli trials. For 10 students, the number who pass follows a binomial distribution.  Binomial distribution for student passes # --- BINOMIAL DISTRIBUTION --- import numpy as np import matplotlib.pyplot as plt from scipy.stats import binom n, p = 10, 0.7 # 10 students, P(Pass) = 0.7 k = np.arange(0, 11) pmf = binom.pmf(k, n, p) plt.bar(k, pmf) plt.xlabel('Number of Passes') plt.ylabel('Probability') plt.title('Binomial Distribution (n=10, p=0.7)') plt.grid(True, alpha=0.3) plt.savefig('.\/images\/essential-probability-and-statistics\/binomial-dist.png', dpi=300) plt.show() # --- END CODE ---   Bar plot of the binomial probability mass function (PMF) for the number of students passing an exam out of 10, with a pass probability . Each bar represents the probability of students passing, calculated as . The peak around 7 passes reflects the high likelihood of most students passing given . This distribution is critical in machine learning for modeling binary outcomes, such as predicting the number of successful predictions in a classification task.   Bar plot of binomial distribution.      Three Types of Probabilities  Probability can be approached theoretically, empirically (frequentist), or subjectively (Bayesian).     Theoretical Probability : Uses symmetry. For a fair die, . For even numbers, .     Frequentist Probability : Estimates probability from trial frequencies: .  Frequentist estimation for fair and biased dice # --- FREQUENTIST SIMULATION --- import numpy as np import matplotlib.pyplot as plt np.random.seed(42) n_trials = 1000 fair_rolls = np.random.randint(1, 7, n_trials) biased_rolls = np.random.choice([1, 2, 3, 4, 5, 6], n_trials, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]) # Cumulative probabilities cum_fair = np.cumsum(fair_rolls == 1) \/ np.arange(1, n_trials + 1) cum_biased = np.cumsum(biased_rolls == 1) \/ np.arange(1, n_trials + 1) plt.plot(cum_fair, label='Fair Die (P=1\/6)') plt.plot(cum_biased, label='Biased Die (P=0.2)') plt.axhline(1\/6, color='red', linestyle='--', label='Theoretical P=1\/6') plt.xlabel('Trials') plt.ylabel('Estimated P(1)') plt.title('Frequentist Estimates: Fair vs. Biased Die') plt.legend() plt.grid(True, alpha=0.3) plt.savefig('frequentist-convergence.png', dpi=300) plt.show() # --- END CODE ---   Plot showing the convergence of frequentist probability estimates for rolling a 1 on a fair die ( ) and a biased die ( ) over 1,000 trials. The fair die’s estimate (blue) fluctuates but approaches 1\/6 (red dashed line), while the biased die’s estimate (orange) converges to 0.2, reflecting the higher probability of rolling a 1. This visualization demonstrates how empirical frequencies approximate true probabilities in large samples, a technique used in machine learning to estimate probabilities from training data.   Convergence plot for frequentist estimates.       Bayesian Probability . This is also an empirical definition of probability. But, rather than give you one number for probability of an event, Bayesian gives you a probability distribution of the values of probability of the event. From that, you can work out the mean value, which you can use as one value for the probability of the event.  It is based on incorporating belief about the probability of an outcome BEFORE we even conduct the experiment and then updated this so-called prior assumption or bias with what we observe in the experiment. The updated belief is the posterior, and improved value of the probability.  Clearly, as we repeat the experiment infinitely many times, the effect of our initial belief would disappear and the answer will match the results of the frequentists' experiments. However, since we can never do infinite number of trials, the Bayesian gives an edge in cases where we have some information about the outcome even before we start the trials.   Example: This example is a little bit ahead of my presentation here as it requres a little bit of math to properly express how th Bayesian probability works. If you feel up to it, you can ahead and read on, but it's okay to skip it for now.  In the case of a six-sided die, suppose we want to estimate the probability for one-dot face up as we illustrated in the frequentist case above. First, we would need to choose a prior belief, i.e., a probability distribution for , i.e., how likely is any value of between its range of values, which will be from to , inclusive, . Since, we do not know which value is right, we might decide that it could be 1\/2 times it will be face up and 1\/2 of the time it will be not face up (I know a fair die will be 1\/6 times face up, but I want to show you how even a very off prior will eventually converge to the proper value). In such cases and our trial each time being either face up true or false ( which is a case of Bernoulli trials ), it is traditional to choose a beta distribution, which has two parameters and , with and . Using symbol for and , probability density of , we will write this as follows where . where is beta function. The mean value of beta distribution is an important result and can be easily found. Here, I have introduced physicists' notation for the mean of a quantity, . Thus, by choosing as the prior distribution, we are assuming that somehow we suspect that is close to . So, we are basically, starting way off in our belief.  Just a side math info: Beta function is usually written in terms of factorial or Gamma function. where, for integer arguments , and in general, To hide all the mathematical details in our work below, we will, as is normally done, just express the probability by a simpler notation and represent Eq.  where instead of lower case variable name , we use the notation of upper case .  Let's get back to our rolling experiment and see how our belief of the true value of evolves with each roll's result. Suppose we roll the die and observe that the up face is not one, then without showing you the calculations here, which will be done later in the chapter, we use Bayes rule, to be discussed later, to show that the probability distribution now shifts to . How did we go from distribution to ? I used Bayesian theorem. We will not show the calculation here but differ to a later section.  Toss second time, let's say the result is a one. Then our belief will be update with this new data to . Toss again, say no one. We keep updating the probability distribution of . At any point, we can take the expectation value of the the variable in the current distribution to give us the \"best current value\" for . Thus, after three trials above, we will say that .  Suppose you continued rolling and you had the following next 7 trials: After these 10 trials in total, the distribution will be This is still far away from that you would expect from a fair die, but you don't know if the die was fair. So, empirical results are all you have to go by.  For the same rolling results, frequentists' probability will give us the following estimate:   They look similar. But, had you expected the die was fair, you would start with a better prior, with say . Then the 10 trials would update to It would have revealed if the die was not a fair die. It's either not a fair die or we have rolled it too few times.      Conclusion  Probability provides the foundation for modeling uncertainty in machine learning. Axioms define the rules, while theoretical, frequentist, and Bayesian approaches offer different perspectives. Conditional probability and distributions are used in machine learning models to make predictions.   "
},
{
  "id": "fig-venn-diagram-E1-E2",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-venn-diagram-E1-E2",
  "type": "Figure",
  "number": "1.4.1",
  "title": "",
  "body": " Venn diagram illustrating events (odd numbers) and for 1,000 simulated rolls of a fair six-sided die. The code of the simulation is given below.   Venn diagram of intersecting events.   "
},
{
  "id": "fig-venn-sum-rule",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-venn-sum-rule",
  "type": "Figure",
  "number": "1.4.2",
  "title": "",
  "body": " Illustration of the sum rule. The probability of is the sum of the shaded areas of and , minus the overlapping region that would otherwise be counted twice.   Venn diagram showing the sum rule for two overlapping events.   "
},
{
  "id": "subsec-conditional-probability-independence-3",
  "level": "2",
  "url": "sec-Basic-Probability.html#subsec-conditional-probability-independence-3",
  "type": "Example",
  "number": "1.4.3",
  "title": "Probability of King if the Card is known to be Spade.",
  "body": "Probability of King if the Card is known to be Spade  Imagine we have a standard deck of 52 playing cards. Event = \"the card is a King\". Event = \"the card is a Spade\".  The unconditional probability of drawing a King is . But suppose we are told that the card drawn is a Spade. Now, the sample space is only 13 Spade cards. Out of these, exactly one is a King (the King of Spades).  Thus the conditional probability is , which differs from . This illustrates how new information (that the card is a Spade) changes the probability of .  "
},
{
  "id": "subsec-conditional-probability-independence-5",
  "level": "2",
  "url": "sec-Basic-Probability.html#subsec-conditional-probability-independence-5",
  "type": "Example",
  "number": "1.4.4",
  "title": "Tosses of a fair coin are independent.",
  "body": "Tosses of a fair coin are independent  Suppose we toss a fair coin twice. Let event be \"the first toss is Heads\" and event be \"the second toss is Heads.\" The sample space is .  We have , , and . Therefore, , which means the events are independent.  Now let event be \"at least one toss is Heads.\" Then , , and . But , so . Thus, and are not independent.  "
},
{
  "id": "fig-joint-probability-heatmap",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-joint-probability-heatmap",
  "type": "Figure",
  "number": "1.4.5",
  "title": "",
  "body": " Joint probability table of passing and high study hours. Each cell represents . Row and column sums recover marginals ( and ), illustrating the sum rule. The product rule is verified by comparing the joint probability with .   Heatmap of joint probabilities for pass and study hours.   "
},
{
  "id": "fig-binomial-dist",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-binomial-dist",
  "type": "Figure",
  "number": "1.4.6",
  "title": "",
  "body": " Bar plot of the binomial probability mass function (PMF) for the number of students passing an exam out of 10, with a pass probability . Each bar represents the probability of students passing, calculated as . The peak around 7 passes reflects the high likelihood of most students passing given . This distribution is critical in machine learning for modeling binary outcomes, such as predicting the number of successful predictions in a classification task.   Bar plot of binomial distribution.   "
},
{
  "id": "fig-frequentist-convergence",
  "level": "2",
  "url": "sec-Basic-Probability.html#fig-frequentist-convergence",
  "type": "Figure",
  "number": "1.4.7",
  "title": "",
  "body": " Plot showing the convergence of frequentist probability estimates for rolling a 1 on a fair die ( ) and a biased die ( ) over 1,000 trials. The fair die’s estimate (blue) fluctuates but approaches 1\/6 (red dashed line), while the biased die’s estimate (orange) converges to 0.2, reflecting the higher probability of rolling a 1. This visualization demonstrates how empirical frequencies approximate true probabilities in large samples, a technique used in machine learning to estimate probabilities from training data.   Convergence plot for frequentist estimates.   "
},
{
  "id": "sec-Random-Variables-and-Probabilities",
  "level": "1",
  "url": "sec-Random-Variables-and-Probabilities.html",
  "type": "Section",
  "number": "1.5",
  "title": "Random Variables and Probabilities",
  "body": " Random Variables and Probabilities   In this section we will present foundations of the calculational tools necessary for analytical work. First we will define random variables and then follow up with probability distributions of a single and then two random variables. The generalization of more than two random variables will be left for future sections.     Random Variables, Probabilities, and Expectations  We will think of variables as something the is observed or measured by experiments. The outcome in any experiment is a real value of the variable. A variable whose value is uncertain or unpredictable or varies from trial to trial, even though measurement conditions haven't changed, is called a random variable .  We tend to use capital letter for the name of the variable and small letters for its values. Thus, for a variable , the values will be denoted by etc., or a generic value by just . Sometimes we will use superscripts to denote values, .  An event will now refer to the outcome that in a particular trial, variable has some value or a set of the possible values. Thus, would be an event and so would be , etc. In case of continuous values for , and event may even be written as , etc.  A random variable can either continuous variable or a discrete variable . A continuous random variable takes values either in a finite segment of the real line or on an entire real line. For example, the price of a house (say, denoted by ) in the US dollars could be between and . We can say .  A discrete random variable takes values in a countable set e.g., die outcomes , colors . When a discrete random variable has abstract symbols or wrods as values, we seek numerical embeddings of them using real-valued vectors so that they can be fed into machine learning algorithms for processing.    Probability Mass Function  For a discrete random variable with values ( ), the probability mass function (PMF) assigns appropriate probabilities to events of all the unique values that the random variable can take. Thus, PMF of our example variable will give all the values in: where the last condition is important since it makes sure that probability values are properly normalized and all the unique events are included.   Probability mass function (PMF) of a fair six-sided die, showing equal probabilities ( ) for outcomes . This bar chart visualizes the discrete distribution, useful for understanding expected values in games of chance.   Bar chart of die PMF.    PMF of a fair die # === CODE: PMF of a fair die === import matplotlib.pyplot as plt x = [1, 2, 3, 4, 5, 6] p = [1\/6] * 6 plt.figure() plt.bar(x, p) plt.xlabel(\"Die Outcome\") plt.ylabel(\"Probability\") plt.title(\"PMF of a Fair Die\") plt.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"die_pmf.png\", dpi=300) plt.show()    Expectation Values  The expectation value of a function , which is also simply called expectation of , is the weighted value of the function according to the probailities of each value of . The mean is just when the function is the variable itself. In Physics, the expectation is commonly written as , but in probability and statistics, the notation is , with in the bracket.  The variance measures spread of distribution about the mean value. It is defined by   Proof of the last step  Here, the second equality is easy to see by an explicit work, noting that is just a number. Let's denote it by letter .    The standard deviation is just square root of variance. .   Die roll expectation and variance  For a fair six-sided die, for . The mean is: The second moment is: The variance is:      Probability Density of Continuous Variables  Probability distribution of a continuous variable cannot be given by a PMF since there are uncountably infinitely many unique values in the sample space , which is taken to be the entire real line. If we have an that is non-zero only in an interval, we can extend this to the entire real line by just stating that the probability for all other values be zero.  Instead of PMF, it turns out that the best we can do in the case of continuous variables is to define probability within an infinitesimal interval around some value  when we want to know about the distribution of probabilities at  . Thus, we nead probability per unit interval, i.e., probability density function ( PDF ) at , which we will denote by . Events are then defined by an interval in which you will find the value of the random variable . Thus, the probability of the event that will be Normalization requires that this integral over all values of be . A very useful probability density is a uniform value between, say and . Due to normalization, the constant value of .       (for accessibility)    You can easily check that it is properly normalized.    Another probability density of extreme importance is the Gaussian density function centered about zero and having unit variance. This Gaussian is called normal density.    (for accessibility)    Although, it is more difficult to show but the integral of a Gaussian with the factor as shown does give you the correct normaliztion.    Proof: Let denote the following integral. We need to show that . We look at and write the second using another dummy symbol . Now, combine the two integrals and think of them as integral in the -plane. Now, change variable to polar coordinates with With , the integral over separates out and gives leaving only integral over , which goes from to . The integral over is easily done by substitution . Show that the value of this integral is just . Since , only positive root is applicable. Hence     Expectation values of functions continuous random variables are also weighted values of the functions. But, here you need to do the integral. Accordingly, mean, variance, and standard deviations will be We will study examples of various distribution functions in the next section.    Cumulative Distribution Function (CDF)  For continuous distribution functions, we can find the probability of an event that the outcome of an observation on the random variable will be less than or equal to some real number by the following integral. It's important to note that on the right side the is some definite fixed value and that is why I am using as the dummy variable for the integral. You could use or or any other symbol for the dummy variable. Try not to confuse the dummy variable with the value variable. The value can be any value on the real line.  Probability in Eq. is called cumulative distribution function ( CDF ) corresponding to that probability density . We typically use symbol for the CDF. It is clearly a function of the value variable . The integral-derivative relation in the fundamental theorem of Calculus immediately gives us the following relation between and . From the definition in Eq. it is clear that when , then we would be integrating the PDF over the entire real line, which by normalization should be . Similarly, when , the integral should be zero. Thus over the real line, CDF goes from at to at . shows this behavior of CDF of two distributions. Notice that the shape of the CDF's are different for different distribution functions and carry the information of the probabilities of events in different intervals.   Comparing cumulative distribution functions (CDF) of a standard normal distribution and a uniform distribution which is non-zero between and .   Comparing cumulative distribution functions (CDF) of a standard normal distribution and a uniform distribution.     "
},
{
  "id": "subsec-Random-Variables-2",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Random-Variables-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "random variable "
},
{
  "id": "subsec-Random-Variables-5",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Random-Variables-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "continuous variable discrete variable "
},
{
  "id": "subsec-Random-Variables-6",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Random-Variables-6",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "discrete random variable "
},
{
  "id": "fig-die-pmf",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#fig-die-pmf",
  "type": "Figure",
  "number": "1.5.1",
  "title": "",
  "body": " Probability mass function (PMF) of a fair six-sided die, showing equal probabilities ( ) for outcomes . This bar chart visualizes the discrete distribution, useful for understanding expected values in games of chance.   Bar chart of die PMF.   "
},
{
  "id": "subsec-Expectation-Values-2",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Expectation-Values-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "mean "
},
{
  "id": "subsec-Expectation-Values-3",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Expectation-Values-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "variance "
},
{
  "id": "subsec-Expectation-Values-4",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Expectation-Values-4",
  "type": "Proof",
  "number": "1.5.3.1",
  "title": "Proof of the last step.",
  "body": "Proof of the last step  Here, the second equality is easy to see by an explicit work, noting that is just a number. Let's denote it by letter .   "
},
{
  "id": "subsec-Expectation-Values-5",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Expectation-Values-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "standard deviation "
},
{
  "id": "subsec-Expectation-Values-6",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Expectation-Values-6",
  "type": "Example",
  "number": "1.5.2",
  "title": "Die roll expectation and variance.",
  "body": " Die roll expectation and variance  For a fair six-sided die, for . The mean is: The second moment is: The variance is:   "
},
{
  "id": "subsec-Probability-Distribution-of-Continuous-Variable-3",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Probability-Distribution-of-Continuous-Variable-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "probability density function PDF "
},
{
  "id": "subsec-Probability-Distribution-of-Continuous-Variable-8",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Probability-Distribution-of-Continuous-Variable-8",
  "type": "Proof",
  "number": "1.5.4.1",
  "title": "",
  "body": " Proof: Let denote the following integral. We need to show that . We look at and write the second using another dummy symbol . Now, combine the two integrals and think of them as integral in the -plane. Now, change variable to polar coordinates with With , the integral over separates out and gives leaving only integral over , which goes from to . The integral over is easily done by substitution . Show that the value of this integral is just . Since , only positive root is applicable. Hence   "
},
{
  "id": "subsec-Cumulative-Distribution-Function-3",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#subsec-Cumulative-Distribution-Function-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "cumulative distribution function CDF "
},
{
  "id": "fig-cdf-compared-normal-uniform",
  "level": "2",
  "url": "sec-Random-Variables-and-Probabilities.html#fig-cdf-compared-normal-uniform",
  "type": "Figure",
  "number": "1.5.3",
  "title": "",
  "body": " Comparing cumulative distribution functions (CDF) of a standard normal distribution and a uniform distribution which is non-zero between and .   Comparing cumulative distribution functions (CDF) of a standard normal distribution and a uniform distribution.   "
},
{
  "id": "sec-Two-or-More-Random-Variables",
  "level": "1",
  "url": "sec-Two-or-More-Random-Variables.html",
  "type": "Section",
  "number": "1.6",
  "title": "Two or More Random Variables",
  "body": " Two or More Random Variables   Suppose we have two or more random variables that characterize our data. For example, we may be interested in studying heights and weights and weights of all children in age to . Our probabilities of interest will look like This kind of probability is called a joint probability - it is probability of two events together. You might think of the expressions on either side of as events and and replace by the set symbol , and write it more abstractly as Or, more generally, you might think of this as a statement about random variables and and write it as Sometimes, a more verbose notation is used:   We may also be interested in just to probability of some range of Height itself, regardless of the weights. Since we have collected data on two variables, but we are looking at only one of the variables, we say that we have margined out the weight variable. This type of probability is called marginal probability , here of Height.  We may instead be interested in a more complicated type of probaility: what if we look at children of Weight between and , what will be the probability of a Height range? We write these using a different symbol to indicate the constraining part of this sentence, i.e, the given part. Or, we may be interested in a question that is other way around, The answers to the two questions will be different since you would be selcting different distibutions based on the given condition; is probability in the -space and is probability in the -space. These types of probability are called Conditional Probability . Conditional probability plays an important role in Machine Learning Algorithms.  Below, we will look at these probabilities in more detail.    Joint Probability  Now, let's dig in a little deeper into joint probability and work out an example.  Suppose we have two discrete random variables and . Suppose the unique values that can take are and can take are . Then, the pair can take value pairs with and . Joint porbability , i.e., the Joint PMF, assigns probabilities to every unique pairs so that they are normalized to sum to . If you are dealing with continuous variables and , then we can only talk about probabilities in ranges since both and take values on the real line . Here, the space of all possibilities will be the -plane, which is . Thus joint probability in an infinitesimal rectangle will be where the Joint PDF wil be normalized by the following integral: In a weird way the Joint PMF of the discrete case corresponds to a the Joint PDF except that you need to take integral of the later in place of sum for the PMF.   Patient disease and test results  Consider a dataset of 1,000 patients, with two discrete random variables indicating disease status and indicating test result on a partcular test for the said disease. For each patient, the disease status variable has on of the two values: Similarly, the test result variable for each patient has one of the two values Suppose, in people in the dataset you found people had tested positive and also had the disease. This will be the joint probability Suppose, you found that patients who had the disease but somehow tested negative. That would be Now, it turned out that people who didn't have the disease but their tests came out positive. Finally, people that didn't have the disease were also found to be negative on the test. These joint probabilities are usually organized in a table form as shown in    Disease and Test Results       Test: Test:    Disease (D)     No Disease (N)          Marginal Probability   Marginal probability is an unconditional probability of a single event. for instance marginal probability of event is the same old regular probability of , viz. we are aware of. In the context of joint probabilities, we use the phrase marginal probability. Outside of this context, it is just probability.  So, what is it's relation to the joint probability? To present the answer, let's look at joint probability in the context of two discrete random variables and , which take values in finite sets and , respectively. Let be their joint probability, meaning collections of pairs. Marginalized probabilities will be probabilities of events like , , etc. without any consideration of the variable . Or, alternatively, , , etc. without any consideration of the variable .  We can get from . That means, we need to sum over all the values of in . Rather than clutter the formula we are going to use the following notation in equations. Therefore, marginal probability : For arbitrary : Of course, we can write this in a compact notation and drop the subscripts. If and were continuois variables, we will work with the PDF and replace sum by an integral, in this case integral over . Note that we are using same symbol for TWO DIFFERENT PDFs. To remove the confusion, subscripts are used to remind the random variable names.    Marginal Probabilites from Joint Probability Table  Let's work out Marginal probabilities , , , anbd from the joint probabilites given in . These make up the PMF of the random variable Disease Status , regardless of what ever happened in the tests. The PMF of the Test Result variable regarless of the Disease Status are   Often, we display joint and marginal probabilities in a heatmap as shown in with the marginals shown in the margins outside the Table.   Heatmap of joint probabilities for disease status ( ) and test result ( ) from patients ( ). Darker shades indicate higher probabilities. Marginal probabilities ( , ) are shown in the margins.   Heatmap of joint probabilities with marginals.   The following program was used to create the heatmap. Joint probability heatmap import numpy as np import matplotlib.pyplot as plt import seaborn as sns # Joint probabilities from patient table joint = np.array([[0.15, 0.05], [0.30, 0.50]]) # Rows: X=D, X=N; Cols: Y=+, Y=- marginal_x = np.sum(joint, axis=1) # P(D), P(N) marginal_y = np.sum(joint, axis=0) # P(+), P(-) # Heatmap with marginals fig, ax = plt.subplots() sns.heatmap(joint, annot=True, fmt=\".3f\", cmap=\"Blues\", cbar=False, xticklabels=[\"$Y=+$\", \"$Y=-$\"], yticklabels=[\"$X=D$\", \"$X=N$\"], annot_kws={\"size\": 16}, ax=ax) for i, m in enumerate(marginal_x): ax.text(2.1, i + 0.5, f\"{m:.3f}\", va=\"center\") for j, m in enumerate(marginal_y): ax.text(j + 0.33, 2.2, f\"{m:.3f}\", ha=\"center\") ax.text(2.2, 2.2, \"1.000\", va=\"center\", ha=\"center\") plt.title(\"Joint Probability Heatmap with Marginals\") plt.tight_layout() plt.savefig(\"joint_heatmap.png\", dpi=300) plt.show()      Conditional Probability   Conditional probability is probability of one event given the knowledge that another event has occured. That is, you are allowed to look into the world in which should occur and then in that world, ask what fraction of that world is where would also occur, i.e., the event . Thus, conditional probability, denoted by will be the ratio of to . Of course, from the setup itself, since we are assuming has occured, i.e., its unconditional probability must have been zero.  So, how is it related to joint and marginal probabilities? Let's look at our ongoing example of two discrete random variables and with joint probabilities and marginal probabilities and , where I used in so that we don't confused by for and another from in the same context. The conditional probability of event for given event for will be, using the simplified notation of dropping and , Now, notice that is a distribution in a world where has a particula value and we are uncertain about , which could be in any of its values, and we are asking what will be the probability that the value will be . We can denote the PMF resulting from the values of when is kept at as .  Thus, we will have DIFFERENT CONDITIONAL PROBABILITY MASS FUNCTIONS in space, one for each value of : That is the complication that conditional probability brings and also richness of the questions you can ask of the data!  Now, let's look at Conditonal PMF . It is a bonafide probability mass function with values: They must add to .    Patient disease and test results  Using the patient dataset ( ), compute all conditional probabilities:   Patient Counts (Disease vs. Test)   Marginal  150 50  300 500  Marginal    Joint probabilities:   Marginals: , , , .  Conditional probabilities: Let us do one of the conditional probabilities in detail and just write the answers for the rest:  Suppose we want to know that if test came out positive, what is the proability that the individual has disease? Since, we have the full joint probability table, we can answer this question by computing . That is, based on the table, it is only chance that the individual has disease. If we use for the Disease\/No Disease variable, We will have the PDF by completing the calculation on , by OR, simply by using the normalization condition on the PDF. Now, you should compute the PDF's , , and P(Y \\mid N), where is the variable for Test and check your answer for the following table.   For two variables, each with two possible values, i.e., and there are possible conditional PMFs, same as the number of marginal PMFs, but only one joint PMF.  Another useful way to display the information in a conditional probability is to plot the PMFs in a bar chart as shown in , where each PMF is called a group.   Grouped bar chart of conditional probabilities for disease status ( ) given test result ( ) and test result given disease status. Each group shows a probability distribution (summing to 1), illustrating how conditional probabilities slice the joint distribution.   Grouped bar chart of conditional probabilities.    # === Grouped bar chart for all conditional probabilities === import matplotlib.pyplot as plt import numpy as np # Joint counts and probabilities counts = {\"D+\": 150, \"D-\": 50, \"N+\": 300, \"N-\": 500} total = 1000 P_D_plus = counts[\"D+\"]\/total P_D_minus = counts[\"D-\"]\/total P_N_plus = counts[\"N+\"]\/total P_N_minus = counts[\"N-\"]\/total P_plus = P_D_plus + P_N_plus P_minus = P_D_minus + P_N_minus P_D = P_D_plus + P_D_minus P_N = P_N_plus + P_N_minus # Conditional probabilities probs = { \"P(D|+)\": P_D_plus \/ P_plus, \"P(N|+)\": P_N_plus \/ P_plus, \"P(D|-)\": P_D_minus \/ P_minus, \"P(N|-)\": P_N_minus \/ P_minus, \"P(+|D)\": P_D_plus \/ P_D, \"P(-|D)\": P_D_minus \/ P_D, \"P(+|N)\": P_N_plus \/ P_N, \"P(-|N)\": P_N_minus \/ P_N } # Grouped bar plot labels = [\"Given $Y=+$\", \"Given $Y=-$\", \"Given $X=D$\", \"Given $X=N$\"] values = [[probs[\"P(D|+)\"], probs[\"P(N|+)\"]], [probs[\"P(D|-)\"], probs[\"P(N|-)\"]], [probs[\"P(+|D)\"], probs[\"P(-|D)\"]], [probs[\"P(+|N)\"], probs[\"P(-|N)\"]]] x = np.arange(len(labels)) width = 0.2 fig, ax = plt.subplots(figsize=(10, 6)) ax.bar(x - width\/2, [v[0] for v in values], width, label=\"First Outcome\", color=\"blue\") ax.bar(x + width\/2, [v[1] for v in values], width, label=\"Second Outcome\", color=\"red\") ax.set_xticks(x) ax.set_xticklabels(labels) ax.set_ylabel(\"Conditional Probability\") ax.set_title(\"Conditional Probabilities from Patient Table\") ax.legend([\"$P(D|·)$, $P(+|·)$\", \"$P(N|·)$, $P(-|·)$\"]) ax.set_ylim(0, 1) ax.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"patient_conditional_bars.png\", dpi=300) plt.show()      Bayes’ Rule  Bayes's rule (also called Bayes's theorem) is a fundamental principle in probability theory. It describes how to update the probability of a hypothesis, which is expressed as a conditional probability when new evidence is observed. This looks and feels like too abstract. So, to get a better feel for it, let's look at the mechanism of this process.  From the conditional probability definition and\/or product rule of probabilities, we have the following relations between the Joint, Marginal, and Condtional probabilites concerning two events and : Switching and is equally valid, giving us But, the order of listing of events in a joint probability is immaterial since the list is just another way of indicating intersection of the events, Therefore, equating the right sides of Eqs. and gives Suppose we observe , then we would conclude that . Then, we can predict probability of event using the conditional probability. This is Bayes's Rule . We are using the information that has occured to make prediction on another event . Of course, we also need and .  Let us now look at Bayes's rule in the context of two discrete random variables, and , which take values and , respectively, as we have presented above in earlier subsections. For concreteness, let us focus on particular events Then, Bayes's rule will be Now, let us write these in our short notation by dropping and , just remembering this detail in out mind only. Now, recall that marginal probabilities can be obtaine by margining out other variables. Say, we want the denominator part on the right side, . That is the denominator is just a normalizing factor of the the numerator! If you didn't have this factor, the PMF on the left side, will not be normalized to . In this equation, I introduced another index , which is a dummy index for summation, so that we do not get confused by particular events connected with our original event . Using this normalization, we get a form of Bayes's Rule that can be interpreted as it being an update rule on probabilities based on evidence. In this form, everything on the right side is either probability of or is conditioned on the prossibility of for all . That is =, on the right side we have some prior knowledge about and on the right side we have an updated predictions on . It's important to remember that all of these occur event by event although sometimes you will see Bayes's rule written in a grand notation using the random variables themselves.    Use of Bayes's Rule in Medical Diagnosis  Suppose, in a population of women in the age group 30 to 40, develop breast cancer. It's also known that mammogram test identifies of cancers accurately and misses of them. This means that if a woman has breast cancer, the test will be positive only of the time. Furthermore, the test also give negative values of the time correctly, i.e., if a woman doesn't have cancer, the test would came out negative of the times.  Now, a new woman in that age group comes in the lab and unfortunately, she tests positive. So, what would you say about here chances of actually having the cancer? Will it be or something else?  Let us denote for cancer for no cancer, for mamomgram being positive and if negative. What we want is Let's write the Bayes's rule to identify what we need to get this. From the description, we know quite a bit. For instante, probability of cancer in that population is . We are also told that if someone had Cancer, test will be positive of the time. This is a conditional probability data on condition . Now, only thing left in Eq. is . But we are also told that if someone did not have cancer, the test will show negative of the time. Hence, using Eq. we will answer That is the probability of the individual having cancer will be .      Independent Random Variables  Two random variables are said to be independent if their joint probability factorizes in the product of their marginal probabilities. Keep in mind that behind the scene, probabilities in this equation are over events, i.e., it's for having some particular value and having its own particular value. If and are independent variables, we expect We wouldn't write our equations in the verbose manner, prefering to keep it simple. But, beware that probabilities are probabilities of events! Now, we write the left side of Eq. using marginal and conditional probabilities. Canceling from both sides we get That is knowing something about does not tell you anything about probability of . That is, conditioning on is useless when and are independent.  For continuous variables, the equivalent statement is about the joint PDF factorizing into component marginal PDFs.   Independence also implies that their covariance is zero: But zero covariance does not imply independence, except for jointly normal variables.      Covariance and Correlation  A very common aspect of dealing with more than one random variable, say (e.g., height of a man) and (e.g., the weight of the man) is to find out to what extent they tend to vary together. Covariance is a measure of their varying together either in the same direction or in the opposite direction. The normalized version of covariance so that result lies between and is called correlation .  Let denote the joint probability of and . Then, covariance is the following expectation value computed in this probability distribution. where to keep the formula simpler, the mean values of and are dented by and respectively. By opening the braces inside the angle brackets in Eq. , we can rewrite the Covariance formula in another way. where is to be computed using . For and discrete variable and a PMF, the calculation will be Themean values and in this case would relate to the marginals and  Covariance can take any real value, . By dividing the covariance by the standard deviations of and we get correlations, whose values are in the range . The standard deviations in the joint distribution are the same as the standard deviation in the marginal as was the case for the means and . For instance, in the case of discrete random variables, we will get the following for the variances, which is the square of standard deviations. The positive correlation (or covariance) means vary in the same direction, i.e., increasing and increasing occur together. The opposite is the case for negative correlation (or covariance). See .   Positive correlation shows upward trend, nagative shows downward trend and no correlation shows no discenable trend.   Positive correlation shows upward trend, nagative shows downward trend and no correlation shows no discenable trend.     Beware, Correlation is not the same thing as Causation! Just because two things move together doesn't mean one causes the other. For example, suppose you find that ice cream sales and shark attacks may rise together (both happen in hot weather when people flock to the beaches), but it would be ludicurous to suggest that it means ice cream sales cause the shark attack or vice versa! Machine learning models may find patterns, but not always causal ones.    Joint, marginal, and conditional probabilities form the foundation for modeling relationships between random variables. Joint probabilities capture co-occurrence, marginals summarize individual variables, and conditionals refine probabilities based on evidence. Bayes’ rule updates beliefs, while independence simplifies joint distributions. Visualizations like heatmaps, scatter plots, and tree diagrams clarify these concepts. Apply these tools in fields like medical diagnostics, as shown, or explore further with resources like Probability Course .   "
},
{
  "id": "sec-Two-or-More-Random-Variables-2-1",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#sec-Two-or-More-Random-Variables-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "joint probability "
},
{
  "id": "sec-Two-or-More-Random-Variables-2-2",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#sec-Two-or-More-Random-Variables-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "marginal probability "
},
{
  "id": "sec-Two-or-More-Random-Variables-2-3",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#sec-Two-or-More-Random-Variables-2-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Conditional Probability "
},
{
  "id": "ex-Patient-disease-and-test-results",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#ex-Patient-disease-and-test-results",
  "type": "Example",
  "number": "1.6.1",
  "title": "Patient disease and test results.",
  "body": " Patient disease and test results  Consider a dataset of 1,000 patients, with two discrete random variables indicating disease status and indicating test result on a partcular test for the said disease. For each patient, the disease status variable has on of the two values: Similarly, the test result variable for each patient has one of the two values Suppose, in people in the dataset you found people had tested positive and also had the disease. This will be the joint probability Suppose, you found that patients who had the disease but somehow tested negative. That would be Now, it turned out that people who didn't have the disease but their tests came out positive. Finally, people that didn't have the disease were also found to be negative on the test. These joint probabilities are usually organized in a table form as shown in    Disease and Test Results       Test: Test:    Disease (D)     No Disease (N)      "
},
{
  "id": "subsec-Marginal-Probability-2",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#subsec-Marginal-Probability-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Marginal probability "
},
{
  "id": "subsec-Marginal-Probability-5",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#subsec-Marginal-Probability-5",
  "type": "Example",
  "number": "1.6.3",
  "title": "Marginal Probabilites from Joint Probability Table.",
  "body": " Marginal Probabilites from Joint Probability Table  Let's work out Marginal probabilities , , , anbd from the joint probabilites given in . These make up the PMF of the random variable Disease Status , regardless of what ever happened in the tests. The PMF of the Test Result variable regarless of the Disease Status are   Often, we display joint and marginal probabilities in a heatmap as shown in with the marginals shown in the margins outside the Table.   Heatmap of joint probabilities for disease status ( ) and test result ( ) from patients ( ). Darker shades indicate higher probabilities. Marginal probabilities ( , ) are shown in the margins.   Heatmap of joint probabilities with marginals.   The following program was used to create the heatmap. Joint probability heatmap import numpy as np import matplotlib.pyplot as plt import seaborn as sns # Joint probabilities from patient table joint = np.array([[0.15, 0.05], [0.30, 0.50]]) # Rows: X=D, X=N; Cols: Y=+, Y=- marginal_x = np.sum(joint, axis=1) # P(D), P(N) marginal_y = np.sum(joint, axis=0) # P(+), P(-) # Heatmap with marginals fig, ax = plt.subplots() sns.heatmap(joint, annot=True, fmt=\".3f\", cmap=\"Blues\", cbar=False, xticklabels=[\"$Y=+$\", \"$Y=-$\"], yticklabels=[\"$X=D$\", \"$X=N$\"], annot_kws={\"size\": 16}, ax=ax) for i, m in enumerate(marginal_x): ax.text(2.1, i + 0.5, f\"{m:.3f}\", va=\"center\") for j, m in enumerate(marginal_y): ax.text(j + 0.33, 2.2, f\"{m:.3f}\", ha=\"center\") ax.text(2.2, 2.2, \"1.000\", va=\"center\", ha=\"center\") plt.title(\"Joint Probability Heatmap with Marginals\") plt.tight_layout() plt.savefig(\"joint_heatmap.png\", dpi=300) plt.show()  "
},
{
  "id": "subsec-Conditional-Probability-2",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#subsec-Conditional-Probability-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Conditional probability "
},
{
  "id": "exp-Patient-disease-and-test-results-conditional-probabilities",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#exp-Patient-disease-and-test-results-conditional-probabilities",
  "type": "Example",
  "number": "1.6.5",
  "title": "Patient disease and test results.",
  "body": " Patient disease and test results  Using the patient dataset ( ), compute all conditional probabilities:   Patient Counts (Disease vs. Test)   Marginal  150 50  300 500  Marginal    Joint probabilities:   Marginals: , , , .  Conditional probabilities: Let us do one of the conditional probabilities in detail and just write the answers for the rest:  Suppose we want to know that if test came out positive, what is the proability that the individual has disease? Since, we have the full joint probability table, we can answer this question by computing . That is, based on the table, it is only chance that the individual has disease. If we use for the Disease\/No Disease variable, We will have the PDF by completing the calculation on , by OR, simply by using the normalization condition on the PDF. Now, you should compute the PDF's , , and P(Y \\mid N), where is the variable for Test and check your answer for the following table.   For two variables, each with two possible values, i.e., and there are possible conditional PMFs, same as the number of marginal PMFs, but only one joint PMF.  Another useful way to display the information in a conditional probability is to plot the PMFs in a bar chart as shown in , where each PMF is called a group.   Grouped bar chart of conditional probabilities for disease status ( ) given test result ( ) and test result given disease status. Each group shows a probability distribution (summing to 1), illustrating how conditional probabilities slice the joint distribution.   Grouped bar chart of conditional probabilities.    # === Grouped bar chart for all conditional probabilities === import matplotlib.pyplot as plt import numpy as np # Joint counts and probabilities counts = {\"D+\": 150, \"D-\": 50, \"N+\": 300, \"N-\": 500} total = 1000 P_D_plus = counts[\"D+\"]\/total P_D_minus = counts[\"D-\"]\/total P_N_plus = counts[\"N+\"]\/total P_N_minus = counts[\"N-\"]\/total P_plus = P_D_plus + P_N_plus P_minus = P_D_minus + P_N_minus P_D = P_D_plus + P_D_minus P_N = P_N_plus + P_N_minus # Conditional probabilities probs = { \"P(D|+)\": P_D_plus \/ P_plus, \"P(N|+)\": P_N_plus \/ P_plus, \"P(D|-)\": P_D_minus \/ P_minus, \"P(N|-)\": P_N_minus \/ P_minus, \"P(+|D)\": P_D_plus \/ P_D, \"P(-|D)\": P_D_minus \/ P_D, \"P(+|N)\": P_N_plus \/ P_N, \"P(-|N)\": P_N_minus \/ P_N } # Grouped bar plot labels = [\"Given $Y=+$\", \"Given $Y=-$\", \"Given $X=D$\", \"Given $X=N$\"] values = [[probs[\"P(D|+)\"], probs[\"P(N|+)\"]], [probs[\"P(D|-)\"], probs[\"P(N|-)\"]], [probs[\"P(+|D)\"], probs[\"P(-|D)\"]], [probs[\"P(+|N)\"], probs[\"P(-|N)\"]]] x = np.arange(len(labels)) width = 0.2 fig, ax = plt.subplots(figsize=(10, 6)) ax.bar(x - width\/2, [v[0] for v in values], width, label=\"First Outcome\", color=\"blue\") ax.bar(x + width\/2, [v[1] for v in values], width, label=\"Second Outcome\", color=\"red\") ax.set_xticks(x) ax.set_xticklabels(labels) ax.set_ylabel(\"Conditional Probability\") ax.set_title(\"Conditional Probabilities from Patient Table\") ax.legend([\"$P(D|·)$, $P(+|·)$\", \"$P(N|·)$, $P(-|·)$\"]) ax.set_ylim(0, 1) ax.grid(True, alpha=0.3) plt.tight_layout() plt.savefig(\"patient_conditional_bars.png\", dpi=300) plt.show()  "
},
{
  "id": "subsec-conditional-probability-from-joint-probability-3",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#subsec-conditional-probability-from-joint-probability-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Bayes's Rule "
},
{
  "id": "subsec-conditional-probability-from-joint-probability-5",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#subsec-conditional-probability-from-joint-probability-5",
  "type": "Example",
  "number": "1.6.8",
  "title": "Use of Bayes’s Rule in Medical Diagnosis.",
  "body": " Use of Bayes's Rule in Medical Diagnosis  Suppose, in a population of women in the age group 30 to 40, develop breast cancer. It's also known that mammogram test identifies of cancers accurately and misses of them. This means that if a woman has breast cancer, the test will be positive only of the time. Furthermore, the test also give negative values of the time correctly, i.e., if a woman doesn't have cancer, the test would came out negative of the times.  Now, a new woman in that age group comes in the lab and unfortunately, she tests positive. So, what would you say about here chances of actually having the cancer? Will it be or something else?  Let us denote for cancer for no cancer, for mamomgram being positive and if negative. What we want is Let's write the Bayes's rule to identify what we need to get this. From the description, we know quite a bit. For instante, probability of cancer in that population is . We are also told that if someone had Cancer, test will be positive of the time. This is a conditional probability data on condition . Now, only thing left in Eq. is . But we are also told that if someone did not have cancer, the test will show negative of the time. Hence, using Eq. we will answer That is the probability of the individual having cancer will be .  "
},
{
  "id": "fig-pos-neg-no-correlation-examples",
  "level": "2",
  "url": "sec-Two-or-More-Random-Variables.html#fig-pos-neg-no-correlation-examples",
  "type": "Figure",
  "number": "1.6.9",
  "title": "",
  "body": " Positive correlation shows upward trend, nagative shows downward trend and no correlation shows no discenable trend.   Positive correlation shows upward trend, nagative shows downward trend and no correlation shows no discenable trend.   "
},
{
  "id": "sec-Example-Discrete-Probability-Distributions",
  "level": "1",
  "url": "sec-Example-Discrete-Probability-Distributions.html",
  "type": "Section",
  "number": "1.7",
  "title": "Discrete Probability Distributions",
  "body": " Discrete Probability Distributions   Probability distributions describe how probabilities are spread across values of a random variable. Recall that when the random variable is discrete, we have a finite set of choices for the variable values. One example is a random variable that records the outcome of rolls of a six-sided die. This will take one of the six values in its sample space . The probability for each outcome gives us Probability Mass Function (PMF) of the random variable. Thus, PMF of the die rolls, which may be an unfair die prefering one outcome over another, could be The sum of probabilities of every unique outcome in the sample space must equal 1. In this section, we will look at commonly occuring discrete random variables and their PMFs. We defer discussion of examples of continuous random variables and their probability density function (PDF) and cumulative distribution functions (CDF) to the next section.    Bernoulli Distribution  Tossing of a two-sided cin is a prototypical example of a Bernouli experiment - you get either heads ( ) or tails ( ). Every two-state problem where outcome is uncertain can be mapped to this problem. We can represent the uncertain outcome of the experiment by a random variable with two real values for \"tails\" and for \"tails\". That is, the sample space will be just We call such random variables Bernoulli variables . We specify the PMF of Bernouli disrtibution by stating the probability of the two unique events with letter for . From normalization of PMF, it's immediately known that   The separate listing of the probability of the two values of in Eqs. and can actually be written more conveniently in one formula. From this formula, you will get and by substitting and , respectively.   Graphically, Bernoulli distribution are usually plotted as a bar graph. To the right is a plot as bars with a dot at the top of the bar so that you can read the and values more easily. Figure to the side shows an illustration with for , and of course, for .     Bernoulli distribution with p=0.6.    The mean of variable is nothing but weighed average of each value of weighted with its probability. This is also the expectation value . We also represent mean of distributions by symbol . Similarly we can find the expectation value of any power of . For instance, the expectation value of the power of will be Thus, variance, which is represented by , where is the standard deviation, will be Therefore, standard deviation, , of Bernoulli variable is     Binomial Distribution   Imagine a collection of identical coins. A Binomial experiment on this system consists of tossing each coin once and counting the total number of Heads or Tails you got. For instance, for a -coin experiment, you might get no Heads at all or 1 Heads and 9 Tails, or 2 Heads and 8 Tails, etc. Record how many Heads you got in this trial, say you got 3 Heads. Now, let's repeat the experiment. In the second trial, you might get a different number of Heads, say this time you got 8 Heads. That is, total number of heads in this experiment is an example of a Binomial random variable.  If you repeat this experiment many times, you can see a pattern emerge among the values of the random variable. Table shows results 2000 trials of this experiment, organized un a table, called a frequency table . You have more chances of the results to be somewhere in the middle of the range of values than on either exremes.   Binomial Experiment: Frequency and Approximate Probability    Total Number of Heads  Frequency  Approximate Probability    0  8     1  45       2  120     3  220     4  300     5  350     6  300     7  250     8  200     9  150     10  57     Total Number of Trials:  2000     In general, a Binomial random variable counts the total number of success (or failure) in a fixed number of Bernoulli trials. The resulting probabilities make up the Probability Mass Function (PMF) of Binomial disrtibution . Since a Binomial experiment is a collection of identical and independent Bernoulli trials, it is possible to deduce a mathematical formula of the Binomial PMF based on the PMF of the Bernoulli.  Let be the probability in each Bernoulli trial and let there be Bernoulli trials in each Binomial experiment. In each trial of the Binomial experiment you will have a sequence of Bernoulli outcomes (for success) or (for failure). Suppose this sequence has Heads ( ) and Tails ( ). The probability of this sequence of independent Bernoulli results will be just a product of factors of for each and factors for for each , which will give us But the and could have occurred in any order and we would still have the same Heads.To get the probability of getting a total of Heads in any order, we must add up probabilities from all those other permutations as well. That should just mean we need to multiply the above one-order probability by the number of permutations of  's in slots and the remainder by 's. That turns out to be the Binomial coefficient, and hence the name Binomial distribution. Note that this distribution has two fixed parameters , the number of independent Bernoulli trials in each Binomial trial and , the probability of success in each Bernoulli trial. I will like to simplify this notation, by writing with subscripts and keep the when necessary. It emphasizes the fact that even for the same , you have different Binomial experiments for different values of . For instance, in Table , if you had conducted the experiment with Bernoulli trials in each Binomial experiment, instead of , you would have gotten much different probabilities for . This is illustrated in the following figure, .   Illustrating that different values in Binomial distribution correspond to different distributions. Here, with and . See that the probabilities for same value are different for the two distributions.   llustrating that different N values in Binomial distribution correspond to different distributions. Here, with N=10 and N=20. See that the probabilities for same X value are different for the two distributions.    Following code was used to create the plot above.  import numpy as np import matplotlib.pyplot as plt from scipy.stats import binom # Parameters p = 0.5 # probability of success N1 = 20 # number of trials for first distribution N2 = 10 # number of trials for second distribution # Support for each distribution x1 = np.arange(0, N1+1) x2 = np.arange(0, N2+1) # PMFs pmf1 = binom.pmf(x1, N1, p) pmf2 = binom.pmf(x2, N2, p) # Plot fig, ax = plt.subplots(figsize=(8,5)) # Binomial N1 ax.vlines(x1, 0, pmf1, colors='blue', lw=2, label=f'Bin{n1}') ax.plot(x1, pmf1, 'o', color='blue') # Binomial N2 ax.vlines(x2, 0, pmf2, colors='orange', lw=2, label=f'Bin{n2}') ax.plot(x2, pmf2, 'o', color='orange') # Labels and grid ax.set_title(f'Binomial Distribution PMFs (p={p})') ax.set_xlabel('Number of Successes') ax.set_ylabel('Probability') ax.grid(axis='y', linestyle='--', alpha=0.6) ax.legend() plt.show()  Another way to improve your intuition about the Binomial distribution is to look at the impact of changing value for the Bernoulli trials themselves - what impact do they have on a -Binomial? It is shown in . These plots show that low skews the PMF toward fewer successes; produces a symmetric distribution centered at ; high skews toward more successes.   Illustrating that different values in Binomial distribution correspond to different distributions but with .   Illustrating that different p values in Binomial distribution correspond to different distributions but with N=10.    For doing analytical calculations with the Binomial distribution, it is important to recall the following algebraic identity, called Binomial expansion. Using this it is straightforward to show that Binomial distribution is normalized properly since The mean of the Binomial random variable can be obtained by weighing each value of by the corresponding probability. This is just the expectation value of the variable itself. A simple method of showing the result involves taking an appropriate derivative appropriately. The variance is similarly shown to be And, the standard deviation is just the square root.     Average of Random Variables  If we divide the Binomial random variable by , the number of independent Bernoulli trials in the Binomial experiment, we will get the average of those Bernoulli trials. Let denote the Bernoulli random variables that make up the Binomial experiment. The average, denoted by will just be  This random variable is called sample mean . From the values the Binomial random value in the numerator, it is clear that the samle mean will be one of the following values in each Binomial trial. A good way to think about the sample mean is that you have some elementary experiment whose outcome is or (the component Bernoulli trials) and you would like to find out . So, you set up trials and average them to get a better estimation.  With being probability of any of the individual Bernoulli variables to produce a success, i.e., for every one of the , it is immediately follows that expectation value of the sample mean variable is just , independent of . Due to the scaling of the Binomial variable to get the sample mean variable, the variance of the sample mean variable is times the variance of the Binomial variable. which translates to the standard deviation that goes as  As you include increasing number of Bernoulli trials in each Binomial experiment, you estimate of the mean of the sample mean shrink around the true value . Where does this result matter?       Estimation: In statistics, we often estimate by from data.     Interpretation: If you run multiple experiments, your average success rate will be centered at and become more concentrated as grows since the standard deviation drops as . This is illustrated in .     Connection to the Central Limit Theorem: For large , it can be shown that the probability distribution of the random variable tends to become Gaussian with the mean and variance . We write this as even though each is a discrete random variable. This goes by the name Central Limit Theorem , which we will discuss in another section.      Illustrating that for large the distribution of the average of Bernoulli variables of the same tends towards a Gaussian distribution.   Illustrating that for large N the distribution of the average of N Bernoulli variables of the same p tends towards a Gaussian distribution.       Multinomial\/Categorical Distribution  Multinomial distribution is like the binomial's cooler cousin who handles multiple categories at once. Imagine rolling a die, sorting emails into spam, promo, or important, or predicting labels in a machine learning classifier. This distribution is your go-to for modeling counts across several outcomes. Why's it a big deal in machine learning? It's everywhere! In natural language processing, it powers things like bag-of-words models for text. In classification, it's behind softmax outputs, where your model predicts probabilities over multiple classes. Plus, it's key for understanding likelihoods in algorithms like Naive Bayes or evaluating models with metrics like cross-entropy loss. If you're working with categorical data in ML, the multinomial is your best friend.   Definition of the Multinomial Distribution   Picture this: you run independent trials, each with possible outcomes, where outcome has probability (and ). Let us use a compact notation to represent variables for the outcomes as a vector: count how many times each outcome happens. Then , with probability: where are non-negative integers summing to . Each is like a binomial, but they're tied together since .     Visualizing with a Bar Plot  Let's make this fun with a Python plot. Let's simulate 1000 rolls of a fair six-sided die ( , ) and show the counts in a bar chart, with each bar labeled with its frequency. Run this code to see how the counts wiggle around the expected .  import numpy as np import matplotlib.pyplot as plt # Parameters n = 1000 # number of trials p = np.array([1\/6] * 6) # fair die probabilities labels = ['1', '2', '3', '4', '5', '6'] # Generate one sample sample = np.random.multinomial(n, p) print(sample) # Plot bars = plt.bar(labels, sample, color='skyblue') plt.xlabel('Die Faces') plt.ylabel('Counts') plt.title('Multinomial Sample: 1000 Rolls of a Fair Die') # Add frequency labels on top of each bar for bar, count in zip(bars, sample): plt.text(bar.get_x() + bar.get_width() \/ 2, bar.get_height(), str(count), ha='center', va='bottom') plt.show()   shows the results of runninng this program once. You would have to run it a few times to see the randomness, or bump up to watch the counts stabilize.   Multinomial distribution resulting from simulation of a six-sided die shown as a bar graph. Frequency of each outcome is shown with each bar showing its count on top with the constraint that the sum of all frequencies is equal to the total number of trials.   Multinomial distribution resulting from simulation of a six-sided die shown as a bar graph.     The mean, denoted by and variance, denoted by (the use of two subscripts will become clear below when we discuss Covariance), of a multinomial variable, i.e., one of the in its definition, are The covariance among the different outcome variables is important here since different 's are not independent. In fact, covariance is given by These are off-diagonal elements of the covariance matrix, usually denoted by a bold Greek letter, with The off-diagonals are negative because of th constraint among the variables.   Derivation of the Formula for the Mean  Derivation of these formulas is instructive. So, I will include them here. Let's take the class labeled for specificity. Now, let us lump together the outcomes as . Let the probability for be . Then, the sum with constraint already built in will be just the Binomial distribution between the outcomes and . It's easy to do this sum with the trick of taking derivative to bring down a factor of . The sum is just a binomial expression with , which we use at the end.    Derivation of the Formula for the Variance  For variance, we need to compute the second moment of the distribution. We use the same trick as for the first moment. Again, just lookig at will be enough. Hence,    Using Indicator Variables to Prove Variance Formual  There is a very useful method for doing calculations in multinomial variables (and other places as well). Here, we have experiments, each gives outcome in possibilities. So, we can intriduce Bernouli random variables , where and by the following. Note that since is a Bernoulli trial with probability , we will have easy answer: Since is the total number of trials that had outcome , we will have Now, it is almost trivial to get the mean, variance, and the off-diagonal covariances.  For we have a do a little extra work, but still trivial. Now, notice the first sum. Since in the same trial, you will get a unique outcome, either or . Hence, one of these values will be zero in every term of the first term. That will make the first term identically zero. In the second sum, the and are from different trials, so they must be independent. This sums over trials , each of which run from to , so we have terms in total. Hence, the off-diagonal covariance elements are    The aymptotic behavior of multinomial distribution when is huge is of considerable importance for powering tests like chi-squared for goodness of fit test, among other uses. So, I will just state here the results and use them later. By the multivariate central limit theorem, the proportions get close to a multivariate normal in distribution sense: where has diagonals and off-diagonals for .    Poisson Distribution   The Poisson distribution models the number of times an event occurs in a fixed interval of time or space, given that:   Events occur independently.    Events happen at a constant average rate, usually denoted by Greek letter lambda .    Two events cannot occur at the exact same instant. That means we are usually interested in events that are rare within the interval we choose to work with so that it can be safely assumed that two events do not coincide.   A Poisson random variable can take any non-negative integer values since it's just a count. The probability mass function for a Poisson random variable will give probabilities for for each non-negative value for a constant average rate is given by It is obviously normalized since Therefore The mean of Poisson distribution is the average count, \\lambda. as we can show by the following calculations. The variance of Poisson distribution is similarly shown to be also . where the missing steps are left for the student to practice, using the same type of argument as introducing operators appropriately.  A mathematically interesting result is that in an appropriate limit, a Binomial distribution can be shown to become same as Poisson distribution. I will just state the result without giving you the detailed calculations. (hint: You can get factors of from ).    Example Radioactivity is one of the classic and most intuitive real-life examples of the Poisson distribution. Let's look at it a little closely. Radioactive decay is a random process. Each atom has a constant probability of decaying in a fixed time interval. The decays are:    Independent (one decay does not affect another).     Rare events relative to the huge number of atoms.    Occurring with a constant average rate .   These aspects make the Poisson distribution a perfect model for studying the statistics of radioactivity.  Suppose we measure the number of particles emitted from a radioactive source in 10-second intervals. From past experiments, we know that the detector records on average 3 decays per 10 seconds. So, per second, we expect on-average decays. That completely specifies the Poisson distribution. Therefore, we can immediately calculate all sorts of things for the phenomenon. For instance, probability of seeing exactly decays in seconds will be Probability of exactly 3 decays in a second will be Now, for a trick question. What will be the probability of 10 decays in one minute? Well, we will convert our lambda per second to a new lambda per minute. Let's label lambda's by the intervals they refer to. Then   A visual representation of the PMF often helps to build intuition. A simple program in Python can be used to to do that. The plot with is shown in .  import numpy as np import matplotlib.pyplot as plt from scipy.stats import poisson # Average decay rate lam = 3 # Range of possible counts x = np.arange(0, 15) pmf = poisson.pmf(x, lam) plt.figure(figsize=(8,5)) plt.vlines(x, 0, pmf, colors='darkred', lw=2, alpha=0.7, label=f'λ={lam}') plt.plot(x, pmf, 'o', color='black', markersize=5) plt.xlabel('Number of Decays in One Interval') plt.ylabel('Probability P(X=k)') plt.title('Poisson Distribution of Radioactive Decays (λ=3)') plt.grid(axis='y', linestyle='--', alpha=0.6) plt.legend() plt.show()   Poisson Distribution for . The most likely outcome is 3 decays per 10 seconds (the mean). Note that 0 or 1 decay is possible, but much less likely. Seeing 6 or more decays is rare, but not impossible.   Poisson Distribution for lambda=3.      Poisson Process  A Poisson process is a stochastic process used to model the occurrence of events that happen independently and at a constant average rate over time or space.  Formally, a Poisson process is a counting process , where represents the number of events that have occurred up to time  , and it satisfies the following properties:    Initial Condition : , meaning the process starts with no events at time zero.     Independent Increments : The number of events in non-overlapping time intervals is independent. For example, the number of events in is independent of the number in if the intervals do not overlap.     Stationary Increments : The number of events in a time interval of length , i.e., , depends only on the length and not on the starting point .     Poisson Distribution : The number of events in any interval of length follows a Poisson distribution with mean , where is the rate parameter (average number of events per unit time). The probability of events in an interval of length is:      no Simultaneous Events : The probability of two or more events occurring at exactly the same time is negligible (technically, the probability of multiple events in an infinitesimally small interval is zero).       Examples :    Queueing Systems : Customers arriving at a store at an average rate of customers per hour.     Telecommunications : Phone calls arriving at a call center with a constant average rate.     Reliability : Failures of a machine occurring randomly at an average rate of failures per hour.     Traffic : Cars passing a checkpoint on a highway at a constant average rate.       "
},
{
  "id": "sec-Example-Discrete-Probability-Distributions-2-1",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#sec-Example-Discrete-Probability-Distributions-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Probability Mass Function (PMF) "
},
{
  "id": "subsec-Binomial-Distribution-2-1",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsec-Binomial-Distribution-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Binomial experiment "
},
{
  "id": "subsec-Binomial-Distribution-2-2",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsec-Binomial-Distribution-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "frequency table "
},
{
  "id": "tab-Binomial-frequency-and-prob",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#tab-Binomial-frequency-and-prob",
  "type": "Table",
  "number": "1.7.1",
  "title": "Binomial Experiment: Frequency and Approximate Probability",
  "body": " Binomial Experiment: Frequency and Approximate Probability    Total Number of Heads  Frequency  Approximate Probability    0  8     1  45       2  120     3  220     4  300     5  350     6  300     7  250     8  200     9  150     10  57     Total Number of Trials:  2000    "
},
{
  "id": "subsec-Binomial-Distribution-2-4",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsec-Binomial-Distribution-2-4",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Binomial random variable Binomial disrtibution "
},
{
  "id": "fig-binomial-10N20",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-binomial-10N20",
  "type": "Figure",
  "number": "1.7.2",
  "title": "",
  "body": " Illustrating that different values in Binomial distribution correspond to different distributions. Here, with and . See that the probabilities for same value are different for the two distributions.   llustrating that different N values in Binomial distribution correspond to different distributions. Here, with N=10 and N=20. See that the probabilities for same X value are different for the two distributions.   "
},
{
  "id": "fig-binomial-N10p2p5pp8",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-binomial-N10p2p5pp8",
  "type": "Figure",
  "number": "1.7.3",
  "title": "",
  "body": " Illustrating that different values in Binomial distribution correspond to different distributions but with .   Illustrating that different p values in Binomial distribution correspond to different distributions but with N=10.   "
},
{
  "id": "subsubsec-Average-of-Random-Variables-2",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsubsec-Average-of-Random-Variables-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "sample mean "
},
{
  "id": "fig-bernoulli-to-CLT",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-bernoulli-to-CLT",
  "type": "Figure",
  "number": "1.7.4",
  "title": "",
  "body": " Illustrating that for large the distribution of the average of Bernoulli variables of the same tends towards a Gaussian distribution.   Illustrating that for large N the distribution of the average of N Bernoulli variables of the same p tends towards a Gaussian distribution.   "
},
{
  "id": "def-multinomial",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#def-multinomial",
  "type": "Definition",
  "number": "1.7.5",
  "title": "Definition of the Multinomial Distribution.",
  "body": " Definition of the Multinomial Distribution   Picture this: you run independent trials, each with possible outcomes, where outcome has probability (and ). Let us use a compact notation to represent variables for the outcomes as a vector: count how many times each outcome happens. Then , with probability: where are non-negative integers summing to . Each is like a binomial, but they're tied together since .   "
},
{
  "id": "ex-multinomial-figure",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#ex-multinomial-figure",
  "type": "Example",
  "number": "1.7.6",
  "title": "Visualizing with a Bar Plot.",
  "body": " Visualizing with a Bar Plot  Let's make this fun with a Python plot. Let's simulate 1000 rolls of a fair six-sided die ( , ) and show the counts in a bar chart, with each bar labeled with its frequency. Run this code to see how the counts wiggle around the expected .  import numpy as np import matplotlib.pyplot as plt # Parameters n = 1000 # number of trials p = np.array([1\/6] * 6) # fair die probabilities labels = ['1', '2', '3', '4', '5', '6'] # Generate one sample sample = np.random.multinomial(n, p) print(sample) # Plot bars = plt.bar(labels, sample, color='skyblue') plt.xlabel('Die Faces') plt.ylabel('Counts') plt.title('Multinomial Sample: 1000 Rolls of a Fair Die') # Add frequency labels on top of each bar for bar, count in zip(bars, sample): plt.text(bar.get_x() + bar.get_width() \/ 2, bar.get_height(), str(count), ha='center', va='bottom') plt.show()   shows the results of runninng this program once. You would have to run it a few times to see the randomness, or bump up to watch the counts stabilize.   Multinomial distribution resulting from simulation of a six-sided die shown as a bar graph. Frequency of each outcome is shown with each bar showing its count on top with the constraint that the sum of all frequencies is equal to the total number of trials.   Multinomial distribution resulting from simulation of a six-sided die shown as a bar graph.    "
},
{
  "id": "subsec-Multinomial-Distribution-6",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsec-Multinomial-Distribution-6",
  "type": "Proof",
  "number": "1.7.3.1",
  "title": "Derivation of the Formula for the Mean.",
  "body": "Derivation of the Formula for the Mean  Derivation of these formulas is instructive. So, I will include them here. Let's take the class labeled for specificity. Now, let us lump together the outcomes as . Let the probability for be . Then, the sum with constraint already built in will be just the Binomial distribution between the outcomes and . It's easy to do this sum with the trick of taking derivative to bring down a factor of . The sum is just a binomial expression with , which we use at the end.   "
},
{
  "id": "subsec-Multinomial-Distribution-7",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsec-Multinomial-Distribution-7",
  "type": "Proof",
  "number": "1.7.3.2",
  "title": "Derivation of the Formula for the Variance.",
  "body": "Derivation of the Formula for the Variance  For variance, we need to compute the second moment of the distribution. We use the same trick as for the first moment. Again, just lookig at will be enough. Hence,   "
},
{
  "id": "subsec-Multinomial-Distribution-8",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#subsec-Multinomial-Distribution-8",
  "type": "Proof",
  "number": "1.7.3.3",
  "title": "Using Indicator Variables to Prove Variance Formual.",
  "body": "Using Indicator Variables to Prove Variance Formual  There is a very useful method for doing calculations in multinomial variables (and other places as well). Here, we have experiments, each gives outcome in possibilities. So, we can intriduce Bernouli random variables , where and by the following. Note that since is a Bernoulli trial with probability , we will have easy answer: Since is the total number of trials that had outcome , we will have Now, it is almost trivial to get the mean, variance, and the off-diagonal covariances.  For we have a do a little extra work, but still trivial. Now, notice the first sum. Since in the same trial, you will get a unique outcome, either or . Hence, one of these values will be zero in every term of the first term. That will make the first term identically zero. In the second sum, the and are from different trials, so they must be independent. This sums over trials , each of which run from to , so we have terms in total. Hence, the off-diagonal covariance elements are   "
},
{
  "id": "fig-poisson-distribution",
  "level": "2",
  "url": "sec-Example-Discrete-Probability-Distributions.html#fig-poisson-distribution",
  "type": "Figure",
  "number": "1.7.8",
  "title": "",
  "body": " Poisson Distribution for . The most likely outcome is 3 decays per 10 seconds (the mean). Note that 0 or 1 decay is possible, but much less likely. Seeing 6 or more decays is rare, but not impossible.   Poisson Distribution for lambda=3.   "
},
{
  "id": "sec-Continuous-Probability-Distributions",
  "level": "1",
  "url": "sec-Continuous-Probability-Distributions.html",
  "type": "Section",
  "number": "1.8",
  "title": "Continuous Probability Distributions",
  "body": " Continuous Probability Distributions   Continuous random variables take values on an interval of the real line. Unlike discrete random variables, we do not assign probability to single points (which would be zero). Instead, we assign probability to intervals, using probability density functions ( PDFs ), denoted by . For a random variable : Note that the PDF must be a positive semi-definite for probability to be positive semidefinite and sum (integral here) of the probabilities must add to over the entire real line. To faciliate analytical work, we introduce a cumulative distribution function ( CDF ), denoted by , that gives probability of the event that for some particular . Clearly, normalization of probability requires that From the definition of and the Fundamental Theorem of Calculus, we have the following relation also. Although, we have not attached the random variable to the symbols , , , and , you will find the following notations in other places. If your distribution is over a conditonal random variable, say the variable , conditioned over some specific value of another random variable , then notation may be   In this section, we will discuss three distributions that are important for ML use.    Uniform: All values equally likely     Normal (Gaussian): Bell-shaped curve     Exponential: Time until an event occurs   There are, of course, other distributions such as beta and gamma distributions that are also in common use. By presenting just these three here, I hope to give the reader enough feel for what to look for when studying other distributions.   Inverse CDF:   Another function of interest when studying continuous random variables is the inverse cumulative distribution function . It maps a probability to the value , the real line.  We will denote the inverse CDF by since we are denoting CDF by except for the standard Normal distribution, . For , the CDF is tranditionally denoted by the Greek letter phi and inverse by . The symbol (or ) does not mean (or ); it's just a symbol for inverse operation of CDF. Note that inverse of a CDF is a function of probability since it maps a probability to the value of random variable on the real line. As a result, it is often used for generating samples according to a distribution of interest for particular application. Say we want samples of a random variable that is distributed according a Gaussian distribution, (\\mu,\\sigma^2), i.e., Then, we just find the inverse CDF over values of probability distributed uniformly and randomly in the range : Then, we get the corresponding -samples by using the inverse CDF of the distribution,     Uniform Distribution   When a random variable is equally likely to take any value between two real numbers and , we say that the distribution is uniform between these values. The distribution is usually designated by with , andwe say that which is read as  is distributed as  . Since is uniform between and , it's probability interpretation will be where the part without the is the PDF of the distribution, rho, .  The mean value of the uniform distribution will clearly be half way between and as can easily be verified by performing the simple integral for the expecation value of . The variance, which is square of the standard deviation , is similarly calculated to yield where Thus, standard deviation is just the square root of the variance, as usual.   The cumulative distribution function, , which gives the probability that is easily calculated for uniform distribution. where I have used for the dummy variable since now is a particular value. This will be a step function since the formula resulting from the integration depends on where the point happens to lie. The last line says that probability that to have less than any value in is 1.0 since obviously the entire non-zero value range, viz., , is included in this case. The second line says that the probability increases linearly between and since the PDF is constant in this range.   shows plots of PDF and CDF of uniform distribution . You can see that as we scan through the interval of the uniform PDF, the probability accumulates in the CDF and eventually, CDF becomes , which represents probability of any of the values in the interval.   PDF and CDF of Uniform distribution . Note the value of PDF is uniformly while that of CDF increases linear in the interval.   PDF and CDF of Uniform distribution U(0,10). Note the value of PDF is uniformly 0.1 while that of CDF increases linear in the interval.    To generate these plots import: from scipy.stats import uniform and then use the methods uniform.pdf() and uniform.cdf().    Inverse Uniform CDF  Since the CDF of uniform random variable is linear in the range of interest, the inverse is also linear and turns out to be trivial. Recall the CDF of in the range is which is linear in its argument . The inverse will be which is also linear in its argument . Let's check if that's true. So, if you wanted to generate samples in the range that act like they are sampled from the uniform distribution , you would first generate pseudo-random numbers in unit interval using algorithms lke Mersenne Twister. Suppose, you have such sample . Then, you will plug them in the inverse CDF to generate samples from . This is trivial here since we have an analytic expression for the inverse CDF. In other distributions, such as the Gaussian\/Normal distribution, the inverse can be computed only numerically. The stats packages usually have functions that do it for you. For instance, ppf() method in Python\/scipy.stats is used for that purpose. In case of uniform distribution, the command is   from scipy.stats import uniform samples = uniform.ppf(u, loc=a, scale=b-a).      Normal (Gaussian) Distribution   The PDF of a Gaussian or Normal distribution is a bell-shaped curve with only two parameters, a mean and a standard deviation . The name Gaussian is preferred in Physics and Engineering, and Normal is preferred in statistics and data science. In these notes I will use both of them, just for fun.  The PDF of the Gaussian distribution of mean and standard deviation for a scalar variable is defined by where The reason I am writing rather than is that exponent in the latter expression usually prints too small on the screen. While doing calculations by hand, you should stick to notation.  When the distribution of a random variable is Gaussian of mean and standard deviation , i.e., variance we denote this as a short hand notation by The special case of and is called standard normal distribution . A standard normal random variable will obey   As always, the PDF in Eq. has the probability interpretation in an infinitesimal interval around . Thus, if you want the probability of , you will just integrate it. The CDF is just such an integral for the probability of . The integral is unwieldy and only done numerically except when . If , then entire real line is included. That would make While the CDF of a general Gaussian is denoted by , the CDF of Standard Normal, i.e., is given it's own symbol, and it's inverse by .   shows plots of PDF and CDF of Gaussian distribution . Note the bell-shape of the PDF and the soft step of the CDF which goes from to monotonically. To generate these plots import: from scipy.stats import uniform and then use the methods norm.pdf() and norm.cdf().   PDF and CDF of Gaussian distribution .   PDF and CDF of Gaussian distribution N(0,1). Note the bell-shape of the PDF and the soft step of the CDF which goes from 0 to 1.    The mean and variance of a Gaussian is in the definition itself and can be readily checked if you know how do Gaussian integrals.   Doing Gaussian Integrals  Here are couple of tricks of doing Gaussian integrals.      Inverse CDF of Gaussian and Sampling  Sampling using inverse Gaussian CDF is a very common operation in computer simulation and statitical analysis. Just to be sure, recall that CDF is a mapping from to and the inverse will do the opposite mapping. In the case of the uniform didtribution since CDF was linear, the inverse CDF was also linear.  However, the CDF of Gaussian is known only as an intractable integral, the inverse is computed only numerically. Fortunately, all standard stats packages already have them and it is just a matte of calling a function. In SciPy pacakge, scipy.stats.norm.ppf(p) will produce the value corresponding to the probability level .  Although, general Gaussian distributions are available in the computer libraries, most commonly, it is the standard normal distibution that we use. As has been stated earlier, the CDF for the standard normal is given a special symbol, viz., and the inverse by     The process of drawing random samples for either or of Gaussian begins as for any other distribution by first getting hold of uniformly disrtibuted (pseudo)random values , which are then feed into the ppf() function of . The steps are as follows.     Generate uniform random numbers .    Apply , if using inverse of standard normal. If using scipy.stats pacakge, the code for will be:  scipy.stats.norm.ppf( , loc = , scale = ).  In scipy.stats, you can include the actual loc and scale in the argument itself as shown in the program listing below. In that case: Apply .    Use as your Gaussian samples.      displays an an example of drawing from a Gaussian distribution and how the samples match up with the theoretical distribution. Clearly, the histogram based on the samples is very representative of the theoretical curve. It was produced by the code below.   import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Parameters mu, sigma = 0, 1 # Gaussian if your choice n_samples = 1000 # Step 1: uniform samples u = np.random.rand(n_samples) # Step 2: transform with inverse CDF (ppf) samples = norm.ppf(u, loc=mu, scale=sigma) # Plot histogram vs theoretical PDF x = np.linspace(-4, 4, 200) pdf = norm.pdf(x, mu, sigma) plt.figure(figsize=(7,5)) plt.hist(samples, bins=30, density=True, alpha=0.6, label=\"Sampled (inverse CDF)\") plt.plot(x, pdf, 'r-', lw=2, label=\"Theoretical PDF\") plt.xlabel(\"x\") plt.ylabel(\"Density\") plt.title(\"Sampling from Normal(0,1) using Inverse CDF\") plt.legend() plt.show()    Samples from a Gaussian distribution and the theoretical curve. The histogram is based on 1000 sample points.   Samples from a Gaussian distribution N(0,1) and the theoretical curve. The histogram is based on 1000 sample points.       Exponential Distribution  Exponential distribution is commonly used to model the time till first success in a Possion process. Recall that in a Poisson process events occur independently and at a constant average rate, usually denoted by letter . The exponential random variable can take any value on zero and positive real line, with the PDF given by As usual, it has the following probability interpretation. You can verify that the PDF in Eq. is properly normalized to give probability over the entire range f values, i.e, is . The Cumulative Distribution Function, (CDF), is the probability for . Therefore,  shows the PDF and CDF of the exponential distribution for .   PDF and CDF of Exponential distribution . Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from to .   PDF and CDF of Exponential distribution E(1.0). Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from 0 to 1.    The complement of the CDF, i.e., the probability that is called survival function (SF).   From the survival function, it is possible to prove an important property of exponential distribution: that it is memoryless , meaning that the probability of an event occurring in the next time interval does not depend on how much time has already elapsed. In formulas, this will be a condition on the conditional probability. That is, whether you wait upto and then look at the next units of time or you don't wait and look at the next interval of time , the two events will give the same probability. That is, the probability for intervals of time will be independent of the starting instant.  Proof of Memoryless Property  Let's write the left side of Eq. in terms of joint and prior, based on the definition of conditional probabilities given in an earlier section. Since the joint probability, you see that joint probability will simply equal . Therefore, Now, we use the survival function given in Eq. to write the right hand side quantities and then simplify.    Finally, let's go over the mean and variance of exponential distribution. Mean as usual is the expectation value of the random variable . The variance will be from which we get the standard deviation , which is the positive square root of variance.    Real-World Examples :   Radioactive Decay:  Time until the next decay event for a particle with decay rate .    Queueing Systems:  Time until the next customer arrives at a store, assuming arrivals follow a Poisson process with rate customers per hour. Mean waiting time = hours.    Reliability:  Lifetime of a lightbulb that fails at a constant rate failures per hour. The probability it lasts more than hours is .       Gamma Distribution  The exponential distribution gave us distribution of interval till first success in a Poisson process with an average rate of event occuring is given by a parameter . What will be the distribution of intervals till the success? Let us call this event by name and use letter for its values. Then the PDF of the disrtribution of is given by a Gamma distribution given by where I have chosen to include the information that it is the we are looking at. Often, in place of rate parameter , we use it's inverse, called shape parameter : Let's see how we can derive the expression for the PDF.  Derivation  It turns out that it is far easier to just derive the CDF of this distribution which will give the probability of event occuring in less time than . But, this probability os complement of event that , i.e., in which less than events occur in the interval . That is, Now, less than means, we could have events during , which is easy to work out by using Poisson's distribution of counting events. Thus, Use this in Eq. and take derivative to get the PDF. Now, we change the summation dummy index in the first sum. Let , then goes from and . Then, this sum cancels out all the terms of the second term except the last one.    To get an intuitive feel for the Gamma distribution function, let us plot the PDF for different value of the average parameter as shown in . Since is average waiting time before the event, we expect that with larger , the peak will shift to the right as is clearly evident in the figure.   Gamma distribution with for various values of the average rate parameter .   Gamma distribution with k=2 for various values of the average rate parameter.    Now, let's plot for various values of , i.e., for the same average rate parameter, what happens if we decide to wait for first occurence, or second occurence or 4th occurence of the event. This is shown in . Clearly, the PDFs for higher and higher occurence   Gamma distribution with for various values of .   Gamma distribution with rate parameter fixed and for various values of k.    Why is it called Gamma Distribution?  The name has to do with the factorial in the denominator of the PDF. I will now show you how. In mathematics, there is a special function called, the Gamma function , which is defined by the following integral. By doing this integral once using integration by parts, you can show that If we know we will know all the rest. Hence,   Now, we write the Gamma distribution function as, Now, let's show that this is properly normalized. Since , we have the following integral. Now, you see how Gamma function came out naturally in working with the Gamma distribution function. You might say, it's the normalization constant that you need for making the probability come out right.   By doing the same type of integral as illustrated in showing the normalization calculation above, you can immediately prove that mean and variance of Gamma distribution are as follows.    CDF of Gamma Distribution can be written only as an integral and evaluated numerically for each . The integral part is called incomplete gamma and denoted by lower case gamma . where In Python's scipy.special library, there is function that actually calculates , i.e, the CDF directly.   Waiting Time for the Third Help Desk Call  Suppose that calls arrive at a help desk according to a Poisson process with rate calls per hour. In this example we will compute answers to:   Probability that all three calls arrive within 30 minutes.    Probability that it takes more than one hour to receive three calls.     Let be the random variable for the time for the call arrival. Then, from the discussion in this section, we conclude that will follow the following Gamma distribution.   (a) The probability that all 3 calls arrive within means we are just directly looking for the CDF for this value of . We look this us in Python by , which is .  (b) Probability that it takes more than one hour to receive three calls means the third call must be so that . This is complement of the CDF.     "
},
{
  "id": "sec-Continuous-Probability-Distributions-2-1",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#sec-Continuous-Probability-Distributions-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "CDF "
},
{
  "id": "sec-Continuous-Probability-Distributions-2-4",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#sec-Continuous-Probability-Distributions-2-4",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "inverse cumulative distribution function "
},
{
  "id": "fig-uniform-pdf-cdf",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#fig-uniform-pdf-cdf",
  "type": "Figure",
  "number": "1.8.1",
  "title": "",
  "body": " PDF and CDF of Uniform distribution . Note the value of PDF is uniformly while that of CDF increases linear in the interval.   PDF and CDF of Uniform distribution U(0,10). Note the value of PDF is uniformly 0.1 while that of CDF increases linear in the interval.   "
},
{
  "id": "fig-gaussian-pdf-cdf",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#fig-gaussian-pdf-cdf",
  "type": "Figure",
  "number": "1.8.2",
  "title": "",
  "body": " PDF and CDF of Gaussian distribution .   PDF and CDF of Gaussian distribution N(0,1). Note the bell-shape of the PDF and the soft step of the CDF which goes from 0 to 1.   "
},
{
  "id": "subsec-Normal-Distribution-2-8",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#subsec-Normal-Distribution-2-8",
  "type": "Remark",
  "number": "1.8.3",
  "title": "Doing Gaussian Integrals.",
  "body": "Doing Gaussian Integrals  Here are couple of tricks of doing Gaussian integrals.   "
},
{
  "id": "fig-gaussian-sampling",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#fig-gaussian-sampling",
  "type": "Figure",
  "number": "1.8.4",
  "title": "",
  "body": " Samples from a Gaussian distribution and the theoretical curve. The histogram is based on 1000 sample points.   Samples from a Gaussian distribution N(0,1) and the theoretical curve. The histogram is based on 1000 sample points.   "
},
{
  "id": "fig-exponential-pdf-cdf",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#fig-exponential-pdf-cdf",
  "type": "Figure",
  "number": "1.8.5",
  "title": "",
  "body": " PDF and CDF of Exponential distribution . Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from to .   PDF and CDF of Exponential distribution E(1.0). Note the exponential decaying property of the PDF and the corresponding rise of the CDF which goes from 0 to 1.   "
},
{
  "id": "subsec-Exponential-Distribution-6",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#subsec-Exponential-Distribution-6",
  "type": "Proof",
  "number": "1.8.3.1",
  "title": "Proof of Memoryless Property.",
  "body": "Proof of Memoryless Property  Let's write the left side of Eq. in terms of joint and prior, based on the definition of conditional probabilities given in an earlier section. Since the joint probability, you see that joint probability will simply equal . Therefore, Now, we use the survival function given in Eq. to write the right hand side quantities and then simplify.   "
},
{
  "id": "subsec-Gamma-Distribution-2",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#subsec-Gamma-Distribution-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "shape parameter "
},
{
  "id": "subsec-Gamma-Distribution-3",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#subsec-Gamma-Distribution-3",
  "type": "Proof",
  "number": "1.8.4.1",
  "title": "Derivation.",
  "body": "Derivation  It turns out that it is far easier to just derive the CDF of this distribution which will give the probability of event occuring in less time than . But, this probability os complement of event that , i.e., in which less than events occur in the interval . That is, Now, less than means, we could have events during , which is easy to work out by using Poisson's distribution of counting events. Thus, Use this in Eq. and take derivative to get the PDF. Now, we change the summation dummy index in the first sum. Let , then goes from and . Then, this sum cancels out all the terms of the second term except the last one.   "
},
{
  "id": "fig-Gamma-PDF-various-lambdas",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#fig-Gamma-PDF-various-lambdas",
  "type": "Figure",
  "number": "1.8.6",
  "title": "",
  "body": " Gamma distribution with for various values of the average rate parameter .   Gamma distribution with k=2 for various values of the average rate parameter.   "
},
{
  "id": "fig-Gamma-PDF-various-k-s",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#fig-Gamma-PDF-various-k-s",
  "type": "Figure",
  "number": "1.8.7",
  "title": "",
  "body": " Gamma distribution with for various values of .   Gamma distribution with rate parameter fixed and for various values of k.   "
},
{
  "id": "subsec-Gamma-Distribution-8",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#subsec-Gamma-Distribution-8",
  "type": "Remark",
  "number": "1.8.8",
  "title": "Why is it called Gamma Distribution?",
  "body": "Why is it called Gamma Distribution?  The name has to do with the factorial in the denominator of the PDF. I will now show you how. In mathematics, there is a special function called, the Gamma function , which is defined by the following integral. By doing this integral once using integration by parts, you can show that If we know we will know all the rest. Hence,   Now, we write the Gamma distribution function as, Now, let's show that this is properly normalized. Since , we have the following integral. Now, you see how Gamma function came out naturally in working with the Gamma distribution function. You might say, it's the normalization constant that you need for making the probability come out right.  "
},
{
  "id": "exp-Waiting-Time-for-the-Third-Help-Desk-Call",
  "level": "2",
  "url": "sec-Continuous-Probability-Distributions.html#exp-Waiting-Time-for-the-Third-Help-Desk-Call",
  "type": "Example",
  "number": "1.8.9",
  "title": "Waiting Time for the Third Help Desk Call.",
  "body": " Waiting Time for the Third Help Desk Call  Suppose that calls arrive at a help desk according to a Poisson process with rate calls per hour. In this example we will compute answers to:   Probability that all three calls arrive within 30 minutes.    Probability that it takes more than one hour to receive three calls.     Let be the random variable for the time for the call arrival. Then, from the discussion in this section, we conclude that will follow the following Gamma distribution.   (a) The probability that all 3 calls arrive within means we are just directly looking for the CDF for this value of . We look this us in Python by , which is .  (b) Probability that it takes more than one hour to receive three calls means the third call must be so that . This is complement of the CDF.   "
},
{
  "id": "sec-LLN-and-CLT",
  "level": "1",
  "url": "sec-LLN-and-CLT.html",
  "type": "Section",
  "number": "1.9",
  "title": "Law of Large Numbers and Central Limit Theorem",
  "body": " Law of Large Numbers and Central Limit Theorem   The Law of Large Numbers (LLN) and Central Limit Theorem (CLT) are foundational results in probability and statistics, underpinning many inferential techniques. The LLN ensures that sample averages converge to the population mean, while the CLT describes the distribution of those averages as approximately normal for large samples. This section provides a detailed yet accessible exploration of these theorems, their assumptions, and their applications.   The Setup:   Consider a sequence of independent and identically distributed (i.i.d.) random variables , each with the same probability distribution, mean , and variance , where denotes the expected value. These will be called the true mean and the true variance as compared to their estimates from the sample mean random variable we have defined before: I want to emphasize again that the sample mean is itself a random variable with its own distribution since it's just a sum of other random variables divided by a scaler number. For instance, we have seen in the section on Discrete Random Variables, that if 's were Bernoulli variables with probability and standard deviation ,  ScaledBinomial(n,p) with mean and variance as follows.   In the rest of this section we do not assume any particular distribution for the individual variables, except as required for the theorems we will study below. The Laws of Large Numbers (LLN) and Central Limit Theorem (CLT) describe the behavior of as increases.  The LLN and CLT require that the are independent and identically distributed (i.i.d.) with finite mean . The CLT additionally requires finite variance , and the Strong LLN further requires a slightly stronger condition on the moments of .    The Law of Large Numbers (LLN)  The LLN addresses the question: What happens to the sample mean as the sample size grows large? It comes in two forms: the Weak LLN and the Strong LLN, which differ in their modes of convergence.   Weak LLN: The Weak LLN states that the probability that deviates from the true mean by more than any positive amount approaches zero as : This is called convergence in probability , denoted:    Strong LLN: The Strong LLN states that the sequence of sample means converges to with probability 1 for larger and larger terms in the sequence. This is called almost sure convergence , a stronger condition implying that almost all sample paths of converge to .   Intuitive Example: Consider tossing a fair coin with , so for heads and for tails, with . For , is either 0 or 1. For , suppose we observe 7 heads, so . For , we might get , and for , . As increases, gets closer to , illustrating the LLN. The Strong LLN guarantees this convergence occurs almost surely.   Visualization: The following figure shows the sample mean of rolls of a fair six-sided die (with true mean ) for increasing , converging to .   Illustration of the Law of Large Numbers: Sample means of fair six-sided die rolls converge to the true mean as increases.   Sample means of die rolls converging to 3.5.      Central Limit Theorem  The CLT states that for i.i.d. random variables with finite mean and finite variance , the distribution of the sample mean becomes approximately normal as increases: where denotes convergence in distribution . By subtracting from the sample mean variable and dividing the result by , i.e., shifting and scaling, we obtain the standardized sample mean variable, usually denoted by letter : It is easy to show that the standard sample mean converges to a standard normal distribution, which is usually easy to look up in numerical tables. The probability density of for large is approximately: The cumulative distribution function (CDF) of is:   The CLT is remarkable because it holds regardless of the underlying distribution of (e.g., Bernoulli, exponential, or normal), as long as and are finite. This explains why normal distributions appear in phenomena like measurement errors, test scores, or heights, which are aggregates of many small random effects. Hence, the name Normal!    Visualization:  shows histograms of for a fair six-sided die for different sample sizes, illustrating convergence to a normal distribution.   Histograms of sample means of fair six-sided die rolls for , visually showing convergence to a normal distribution per the CLT.   Histograms showing CLT convergence for die rolls.      LLN vs. CLT  The LLN and CLT are complementary:  LLN: Ensures that (in probability or almost surely), describing the convergence of the sample mean to the true mean.  CLT: Describes the distribution of fluctuations around , stating that , with deviations of order .  In essence, the LLN tells us where the sample mean goes, while the CLT tells us how it fluctuates around that value.    Berry-Esseen Theorem  The CLT states that is approximately normal for large , but how large must be? The Berry-Esseen Theorem quantifies the rate of convergence. Let be the standardized sum, with CDF . The theorem states: where is the third absolute moment, is the standard normal CDF, and is a constant ( , Shevtsova 2011). This implies that the error in the normal approximation decreases as , modulated by the “tail-heaviness” factor .  For example, for a fair six-sided die, is finite, and for , the approximation error is small, ensuring reliable use of the CLT in practice.    Why LLN and CLT Matter?   LLN Applications: The LLN justifies using sample means to estimate population means. For example, in polling, we survey a sample to estimate voter preferences. The LLN ensures that with a large enough sample, the sample mean is close to the true population mean, giving us confidence that the result can be used for decision making.  Consider rolling a fair six-sided die with true mean and variance . For rolls, suppose we compute . The LLN suggests would be approximately . illustrates this convergence.  In statistical mechanics, the LLN applies to time averages in ergodic systems, ensuring that long-term observations of a particle's behavior approximate the population average.   CLT and Confidence Intervals: The CLT enables us to quantify uncertainty in sample means via confidence intervals. We will cover the concwept of Confidence Interval in greater detail in the section on Inferential Statistics. Here, we just show an application. For the die example, suppose we roll times and compute and sample standard deviation Using the sample standard deviation , the standard error (SE) is . A confidence interval in this case will be: This interval suggests that we are confident that true lies between and , consistent with the true mean .   Simulation: The following Python code simulates 10,000 trials of 100 die rolls, computes sample means, and plots a histogram with a 95% confidence interval.  import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Step 1: Simulate dice rolls np.random.seed(42) n_trials = 10000 n_rolls = 100 data = np.random.randint(1, 7, size=(n_trials, n_rolls)) sample_means = data.mean(axis=1) # Step 2: Compute 95% CI mean_est = np.mean(sample_means) std_est = np.std(sample_means, ddof=1) z_975 = norm.ppf(0.975) margin = z_975 * std_est ci_lower, ci_upper = mean_est - margin, mean_est + margin print(f\"Estimated mean = {mean_est:.3f}\") print(f\"95% CI ≈ [{ci_lower:.3f}, {ci_upper:.3f}]\") # Step 3: Plot histogram with CI fig, ax = plt.subplots(figsize=(8,5)) counts, bins, patches = ax.hist(sample_means, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black') plt.axvline(mean_est, color='red', linestyle='--', label=\"Mean of sample means\") plt.axvline(ci_lower, color='black', linestyle='dashed', label=\"95% CI\") plt.axvline(ci_upper, color='black', linestyle='dashed') y_arrow = counts.max() \/ 3 plt.annotate('', xy=(ci_lower, y_arrow), xytext=(ci_upper, y_arrow), arrowprops=dict(arrowstyle='<->', color='black', lw=2)) plt.text(mean_est, y_arrow * 1.1, \"95% CI\", ha='center', fontsize=12) plt.title(\"Sampling Distribution of Dice Means (100 rolls, 10,000 trials)\") plt.xlabel(\"Sample Mean\") plt.ylabel(\"Density\") plt.legend() plt.show()   Histogram of sample means from 10,000 trials of 100 fair six-sided die rolls, with a 95% confidence interval (dashed lines and arrow) around the estimated mean.   Histogram of die roll sample means with 95% CI.     Other Applications: The CLT is crucial for hypothesis testing (e.g., z-tests) and approximating probabilities for sums of random variables, such as total customer purchases in a store.    Large n Binomial Distribution     "
},
{
  "id": "fig-die-rolls-sample-mean-vs-true-mean",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#fig-die-rolls-sample-mean-vs-true-mean",
  "type": "Figure",
  "number": "1.9.1",
  "title": "",
  "body": " Illustration of the Law of Large Numbers: Sample means of fair six-sided die rolls converge to the true mean as increases.   Sample means of die rolls converging to 3.5.   "
},
{
  "id": "subsec-Central-Limit-Theorem-2",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#subsec-Central-Limit-Theorem-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "convergence in distribution "
},
{
  "id": "fig-clt-convergence",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#fig-clt-convergence",
  "type": "Figure",
  "number": "1.9.2",
  "title": "",
  "body": " Histograms of sample means of fair six-sided die rolls for , visually showing convergence to a normal distribution per the CLT.   Histograms showing CLT convergence for die rolls.   "
},
{
  "id": "subsec-Why-Large-n-Matters-5",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#subsec-Why-Large-n-Matters-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "standard error (SE) "
},
{
  "id": "fig-confidence-interval-die-roll",
  "level": "2",
  "url": "sec-LLN-and-CLT.html#fig-confidence-interval-die-roll",
  "type": "Figure",
  "number": "1.9.3",
  "title": "",
  "body": " Histogram of sample means from 10,000 trials of 100 fair six-sided die rolls, with a 95% confidence interval (dashed lines and arrow) around the estimated mean.   Histogram of die roll sample means with 95% CI.   "
},
{
  "id": "sec-t-and-chi-square",
  "level": "1",
  "url": "sec-t-and-chi-square.html",
  "type": "Section",
  "number": "1.10",
  "title": "Chi Square and Student t Distributions",
  "body": " Chi Square and Student t Distributions   Chi square ( ) and Student distibutions are essential for statistical inference, i.e., interpreting and making sense of the data. Previously we have talked about Confidence Interval based on Central Limit Theorem where we need to deal with the data-provided estimates of the mean and the standard deviation. Chi square and t distributions augment that work and show us how to implement the inference more soundly.    Chi Sqaure Distribition  Suppose we have identical amd independent standard normal random variables. Chi square tells us about the probability distribution of the sum of the squares of these variables.  To write the PDF of the sum of the squares variable, let us introduce this variable as and the independent standard normal variables as . where the PDF's of all the variables are identical Then, by appropriately changing variables, we can show that the PDF of is given by We state this result more compactly as Here is said to be the degree of freedom of the distribution. We now derive this result from first principles.  Derivation  First, notice that the variables can be thought as Cartesian axes in an -dimensional real space . With this perspective, is just the square of the spherical radial coordinate in that space. We will exploit this view point below fully.  The probability of an event for which the value of variable is between and will be We can find the probability of this event from the joint probability of all the also as long as we restricted the values to satisfy Eq. . Now, since the are independent variables, joint probability factorizes. where the PDF's are just standard normal PDF for each. For instance, Therefore Putting this back in Eq. gives    The quantity in the square bracket restricted to the spherical shell is just the volume of this -dimensional spherical shell of radius and theickness since is the square of the radius as shown in the figure to the right. where is the Gamma\/factorial function.   The spherical shell in the Z-space.    Substituting the volume of the shell in Eq. we get Comparing this with what we expected in , we find the PDF of the variable to be    The mean and variance is not that challenging a calulation either if you are not afraid of these integrals. first thing I will do to the PDF is to rewrite in variable . You can show that Now, it is much easier to see the Gamma functions in the integrals when you compute expectation values of powers of . Thus, mean will be The last step follows from the recursion relation of Gamma functions. Similarly, we get the following expectation value of the square of random variable. The variance will be  shows plots of the PDF for various values of . If the plots remind you of Gamma distribution, then you are on the right track, since Chi Square distribution is just a Gamma distribution of with substitution, i.e., of a scaled chi-squared variable, as we have seen above in our calculations.   Chi Square PDF for various values of the degree of freedom parameter . With larger the peak shifts to right as expected from the mean being just .   Chi Square PDF for various values of the degree of freedom parameter n.      t-ditribution  The Student t-distribution is a probability distribution used in statistics to model data when sample sizes are small or population variance is unknown as we will see when we study Inferential Statistics. Here we study the math and characteristics of the distribution itself. Student is the pseudonym of William Sealy Gosset who published it in 1908.  The -random variable is constructed from two other random variables, and , with Then, we define the -random variable by You can see that the square root in the denominator is trying to modify the scale of by sampled standard deviation. We will discuss these ideas in a later section. Here we simply ask: what is the PDF of given the PDF's of and ? The ansewer turns out to be the -distribution.   Derivation   Strategy: The transformation we seek in Eq. is a from two-variable space of and to one variable . The trick is to include a fake second variable, which is just the same as , which we call . Thus, a point in space will have the following relation to a point in space. So, we will have a transformation for space to space obtained a joint PDF in . It turns out that margnining out , i.e., integrating out m> in this joint PDF is rather straight forward and leads to the desired .  Since we already know the PDFs of and and since we are assuming they are independent variables, their joint probability distribution is just the product their separate PDFs. Now, to go from to space, we need to preserve probabilities in infintesimal volumes. where is the Jacobian of transformation given by the following determinant. The partial derivatives in the Jacobian are easily obtained from Eqs. and . Therefore, the Jacobian is Hence, All we need now is to express \\rho_{Y,Z} in terms of and by substitutions in Eq. . Notice that when you collect terms containing , they are just what we need for an integral representation of function: With these and we can a simple expression for : Marginitin out requires integration from to . Since Now, we integrate out in to get the -distribution.That is, from afte canceling out the factors of 's, we get     displays the t-distribution PDF for various values of the degree of freedom parameter (df)  and a standard normal, . First thing to notice is that the -distribution is symmetric about the mean similar to . But, for smaller values, the -distribution has fatter tails , meaning rides above the for larger values of the argument. It is interesting that is visually almost inditinguishable from the standard normal.   Student PDF for various values of the degree of freedom (df) parameter .   Student t PDF for various values of the degree of freedom parameter n.    The mean of -distribution is easily calculated by directly evaluating the integral. This is due to the fact that is symmetric in , and when you multiply it with , the integrand becomes an odd function of . The integration of an odd function with symmetric limits will be zero.  A direct calculation of the variance is a little more challenging. It's just too much algebra to get to the same result. So, I will give you an alternate calaculation based on the fact that the random variables and that go into the defining of variable are independent. Recall For Variance of with mean equalto zero is just That amounts to Now, we know that since . Hence, The expectation value on the right side is a much simler Gamma function integral I will leave for you to do. Also, for , you can already show that integral itself divergent. Therefore, we write the variance as    "
},
{
  "id": "subsec-Chi-Sqaure-Distribition-4",
  "level": "2",
  "url": "sec-t-and-chi-square.html#subsec-Chi-Sqaure-Distribition-4",
  "type": "Proof",
  "number": "1.10.1.1",
  "title": "Derivation.",
  "body": "Derivation  First, notice that the variables can be thought as Cartesian axes in an -dimensional real space . With this perspective, is just the square of the spherical radial coordinate in that space. We will exploit this view point below fully.  The probability of an event for which the value of variable is between and will be We can find the probability of this event from the joint probability of all the also as long as we restricted the values to satisfy Eq. . Now, since the are independent variables, joint probability factorizes. where the PDF's are just standard normal PDF for each. For instance, Therefore Putting this back in Eq. gives    The quantity in the square bracket restricted to the spherical shell is just the volume of this -dimensional spherical shell of radius and theickness since is the square of the radius as shown in the figure to the right. where is the Gamma\/factorial function.   The spherical shell in the Z-space.    Substituting the volume of the shell in Eq. we get Comparing this with what we expected in , we find the PDF of the variable to be   "
},
{
  "id": "fig-chi2-PDF",
  "level": "2",
  "url": "sec-t-and-chi-square.html#fig-chi2-PDF",
  "type": "Figure",
  "number": "1.10.1",
  "title": "",
  "body": " Chi Square PDF for various values of the degree of freedom parameter . With larger the peak shifts to right as expected from the mean being just .   Chi Square PDF for various values of the degree of freedom parameter n.   "
},
{
  "id": "subsec-t-ditribution-4",
  "level": "2",
  "url": "sec-t-and-chi-square.html#subsec-t-ditribution-4",
  "type": "Proof",
  "number": "1.10.2.1",
  "title": "Derivation.",
  "body": "Derivation   Strategy: The transformation we seek in Eq. is a from two-variable space of and to one variable . The trick is to include a fake second variable, which is just the same as , which we call . Thus, a point in space will have the following relation to a point in space. So, we will have a transformation for space to space obtained a joint PDF in . It turns out that margnining out , i.e., integrating out m> in this joint PDF is rather straight forward and leads to the desired .  Since we already know the PDFs of and and since we are assuming they are independent variables, their joint probability distribution is just the product their separate PDFs. Now, to go from to space, we need to preserve probabilities in infintesimal volumes. where is the Jacobian of transformation given by the following determinant. The partial derivatives in the Jacobian are easily obtained from Eqs. and . Therefore, the Jacobian is Hence, All we need now is to express \\rho_{Y,Z} in terms of and by substitutions in Eq. . Notice that when you collect terms containing , they are just what we need for an integral representation of function: With these and we can a simple expression for : Marginitin out requires integration from to . Since Now, we integrate out in to get the -distribution.That is, from afte canceling out the factors of 's, we get   "
},
{
  "id": "subsec-t-ditribution-5",
  "level": "2",
  "url": "sec-t-and-chi-square.html#subsec-t-ditribution-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "degree of freedom parameter (df) fatter tails "
},
{
  "id": "fig-student-t-PDF",
  "level": "2",
  "url": "sec-t-and-chi-square.html#fig-student-t-PDF",
  "type": "Figure",
  "number": "1.10.2",
  "title": "",
  "body": " Student PDF for various values of the degree of freedom (df) parameter .   Student t PDF for various values of the degree of freedom parameter n.   "
},
{
  "id": "sec-Point-Estimates-and-Confidence-Intervals",
  "level": "1",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html",
  "type": "Section",
  "number": "1.11",
  "title": "Point Estimates and Confidence Intervals",
  "body": " Point Estimates and Confidence Intervals   Statistics is a useful tool when you are dealing with uncertainty in systems that are too vast to examine every item. In these cases we can collect data on a random subset and use statisical methods to draw and support conclusions that can be reasonably drawn from the data.  One type of question that shows up in the inference are:    Parameter Estimation to answer the question: \"What is the value of the parameter, e.g., mean or standard deviation ? You can estimate not only the means and variances but other parameters as well. Statistical methods also address the question of how good are our estimates - how far off the actual value may be from the estimate and how confident are we in such claims.     Hypothesis Testing to answer questions such as : \"Is the value of the parameter exactly such an such?\" For example, \"Is the average height of an adult human being exactly 5 feet and 4 inches?\" and \"Is average height of Indians greater than the average height of Chinese?\" You can set up hypothesis testing not just of parameters but of pretty much anything, including ML models - \"does this model perform better than the other?\"\".     Both of these topics deserve their own sections. In this section we will tackle issues associated with the point estimation and defer the hypothesis testing to the next section.    Parameter Estimation       Basic Setup  Let's recall the basic notation here again. Suppose we have a popolation from which we have collected samples. Each sampling process itself is a random variable. For instance, if you toss a coin 3 times, you might get HTT; now, your repeat might give you THT, etc. So, the very first toss is giving you H and T in a random way. That is how we can say that if I conduct tosses, each of those tosses will itself be a random variable.  So, suppose, we have performed  random experiments, each represented by its own random variable. Let's denote them as before by We define a sample mean, which is just an average of the 's: and a sample variance In sample variance, we devide by and not because the definition of sample mean has used up one of the degrees of freedom and we are left with one less. This also makes an unbiased estimator of population variance as we will see below.  The population mean and population variance are of course, unknown. We will see below how to use the sample mean and sample variance to get an estimate of these population quantities.  Note that both and are random variables. So, they have their own probability distributions: Sometimes, we also deal with the random variable , the sample standard deviation,   If the variables 's take only values or , i.e., each experiment is an identical Bernoulli trial, then the sample mean is the proportion of 's in the sample. where is the probability of success (i.e., ) in each Bernoulli trial. That is, in this case, sample mean is an estimator of proportion of 's in the population.  Now, we will discuss a procedure called maximum likelihood estimation (MLE) to estimate population parameters based on the data. There is another commonly used method of estimation maximum a posteriori estimation (MAP) based on Bayes' principle will be discussed in a future section.    Maximum Likelihood Estimate  Let's start with an example of estimating the probability of getting a Heads ( ) in a toss of a given coin. We toss it times and note the outcomes, . That's our data with . Each time we toss, we are doing a Bernoulli trial for which we have PMF:   Now, imagine repeating your datapoint experiment many times. Each time you will most likely get different data, Say, the data above was , but it may be some other sequence of , say, the seocnd time, etc. Presumably, you are actually conducting experiment in an -dimensional space of and each collection of data is just one point in this joint space .  The true data points in this -dimensional space maybe concentrated in some region more so than in some other areas; in general, we just do not know. The maximum likelihood estimate is based on assuming that the data you actually got has the highest probability in this -dimensional space , i.e., you are most liley to sample the areas where the actual data points are most concentrated in the space.  Since we are dealing with discrete variables, here we work with PMF. Let represent the joint PMF of . Since, are independent random variables, the joint PMF will factorize. It is convenient to introduce a product symbol, and write the earlier equation as We know expression of the PMF of the individual trials as given in Eq. since we are doing Bernoulli trials here. If we were doing some other experiements, we would have a different expression for them with their own parameters. Thus, This gives us an expression of the joint probability in terms of the data, i.e., and the unknown parameter . P_J of course is a function of the data and parameter . But since our interest is to find that maximizes this joint probability for the given data, we think of as a function of . In this context we call by another name, likelihood function , often denoted by  Thus, in the present example, we have the following function that we need to maximize with respect to the paremeter to obtain the maximum likelihood estimate of . It will be less verbose to write for the sample mean of the data at hand. Let Then, likelihood for the Bernoulli experiments is Notice that taking log both sides will turn this product into a sum. And, since log is a monotonic function of its argument, the maximum of a log of a function will occur at the same place as the function itself. For brevity, let us denote the log of likelihood by . Taking derivative with respect to and setting , we can solve for . Noting that here, we get That is, the MLE estimate of the proportion just the average of the collected data on heads and tails, with heads being and tails being .    MLE estimate is maximum of the log likelihood function.    To show that the extremeum found is the maximum and not the minimum, all we have to do is to check whether the sign of the second derivative when negative or not. Hence, MLE estimate is indeed a maximum of Likelihood function.     What if was not Bernoulli?  We just have to proceed a little differently after Eq. . In that equation, parameters, usually denoted by , of interest will appear in the probability distributions of the individual trials, same for every one of them. Then, the likelihood function will just be The loglikelihood In general, setting the derivative of to zero and solving for the MLE estimate can only be done numercally.    Each Trial of unknown and   We have   In this case, suppose we want an MLE of and . The log likelihood function will be (replacing by the actual parameters here in Eq. .) Now, to find and you will solve the following equations. Show that you get the following answers. The MLE estimate of the variance is a biased estimator. Often, we will use the following unbiased estimate for variance, also called the sample variance . Below we discuss the technical meaning of bias and unbias.   Estimators   From the formulas for MLE etimates of of a Bernoulli and and of a Gaussian , we can write random variables whose expectation values estimate these quantities. We call these random variables estimators . We will denote them with capital letters as we have done for other random variables.  Thus the MLE estimator of of a Bernoulli variable will be Similarly, the MLE estimators of and for a Gaussian\/Normal variable will be If expectation values of these estimators in their corresponding probability distributions yield the appropriate true values, for the Bernoulli case, and and for the Normal case, we say that the estimator is an unbiased estimartor .    Bernoulli estimator is unbiased.  To prove that is unbiased, we calculate it\/s expectation value in the Bernoulli PMF. We know that for expectation value of each is same: Therefore, Hence is an unbiased estimator of .    Gaussian estimator is unbiased.  Here, we know that for each . Hence, just like the Bernoulli , the expectation value of is also unbiased.   Gaussian estimator is not unbiased.  Now, we come to an estimator which is not unbiased. Let's see. Since each , the variance of each , When we expand the parenthesis in Eq. , we will find that we also need: Here is how I got this result. Another term that will show up in our calculation is I leave this part to you to complete. Now, we look at the expectation of the estimator . Hence is not an unbiased estimator of of a Normal variable.    Unbiased Sample Variance   From the result above, it is immediately obvious that had we divided by in the definition of rather than by , the modified quantity, to be denoted by will be an unbiased estimator of . We call this quantity Sample Variance . It is a random variable since it depends upon the random variables 's and .    is unbiased  You can verify that expectation value of sample variance will be the (unknown) population variance by a more direct calculation but a less algebraic way of proving the unbiasedness of is to notice its relation to chi square disrtribution.  Consider a Gaussian variable with mean and variance and sample variance defined by formula defined above. Then, following is true of a related variable : Let's use symbol for to remind us of its distribution. And, rearrange the above relation to Clearly, the expectation value of will be simply But the expectation value of variable is just its degree of freedom, which is here. Hence, Interestingly, although is an unbiased estimator of , its square root is not an unbiased estimator of .      Confidence Interval   From the point estimation section above, we learned how to use maximum likelihood procedure on the sampled data for computing parameters that model distribution of a random variable in nature. For instance, we saw that sample mean gives an unbiased estimate of the probability of success ( 's) in a system. But, how can we be sure that the estimated value is close to the true value since the later is not known? This is where the idea of confidence interval comes in.  A confidence interval for a parameter is an interval of values,  which could contain the true unknown (maybe, even unknowable) value of at a particular level of confidence.  The confidence level is usually written as a percentage using a complementary quantity (alpha), which is written as a fraction of , as follows. The fractional quantity is called confidence coefficient . For instance,, if we want confidence level, we would have Clearly, greater the confidence level (or, equivalently, smaller ), greater should be our confidence that the unknown parameter is contained somewhere inside the confidence interval.  Confidence Interval   A confidence interval of an unknown parameter of a statistical model is defined by the probability of the event that the interval contains the true value of the parameter be equal to the desired confidence level. That is, We will see examples of how we can compute and from the data collected.      The Z Intervals  The confidence interval, called the Z interval , is a confidence interval of the population mean when we are provided with the exact value of the population variance . At this point, let's not worry about how we somehow know the population variance but not the population mean. We are assuming this so that we can apply the Central Limit Theorem (CLT) to this problem. The procedure is best illustrated by a numerical example.  Supose our project is to study height of male adults in a populatoin in which the height varies with a known variance of . We wish to find the confidence interval of the mean height at the confidence level of , i.e., That is, we want to find and so that the following statement is true. where is the true mean of all adult males in the population, which could, in principle, be obtained by measuring heights of every adult male, but, in practice, unpractical.  Notation: We use capital letters for the random variable name and small letters for their values. Thus, we will use for the variable height, for individual heights, for population mean, for sample mean, for the value of the sample mean from the sample collected.  Therefore, we collect a sample by measuring the heights of randomly selected adults. From which, we can, of course, find the sample mean . Say, we found the result . We are also given the population variance of heights to be: Our task is to find values of and from , , and .  The Answer  Let me just give you the answer first and then we will look at the theory based on our thinking that the distribution is more or less like a Gaussian distribution.   Assuming, the included probability is the central part of a Gaussian, we can implement the probability required in Eq. by using the standard normal distribution's CDF, .  We want to include amount of central part of Gaussian and leave out equal amount, say on either side, with total being . That gives . Hence, to get value at the upper end, we need to find which gives probability . That is, we need to solve the following problems for : This will be For , you can use a numeerical tool such as scipy.stats.norm.ppf(k). When I looked up the number for , the value was approximately 1.96.  We call this value . From symmetry of the standard normal about , we arrive at     Calculate the high and low limits of the confidence interval by      So, that's all there is to the -confidence interval.    The Dtails:   Recall that according to CLT, if a variable has population mean , population variance , and points of data, , with sample mean , then, for large enough , the following variable constructed from them has a standard normal distribution. Suppose, instead of Eq. , we start with first trying to find and with the same probability, Can we then somehow transform this to the Eq. ? Following steps show you how. This shows that the and of the confidence interval are How do you get these and ? To find these values, we look at the PDF and CDF of a standard Normal disribution as shown in . The CDF shows that to include the middle of the probability, we need the following values of : The value of corresponds to the probability value Therefore, we solve the following equation to get . Inverting this equation gives This solved our immediate problem of and .   Calculation for the upper and lower values of . The shaded areas on the two sides each have , here, area under the Gaussian curve. The confidence interval is to include the middle , here of the area.   Calculation for the upper and lower values of Z.     Working out the numerical values in the example data we find the confidence interval. Hence, the confidence interval for heights of adult males in the population is . This interval is called Z interval because we use the Standard Normal distribution. It is applicable when we use a known population variance . As has already been pointed out that this is not the case in real settings. The variance must also be estimated from the collected data. We turn to remedy this problem in the next section.   Correct and Wrong Interpretations of Confidence Intervals: How should you interpret the following probability statement? To simplify notation in our discussion, let's rewrite this equation with for the lower end and for the upper end of the Z-interval. with Notice that depends on the data we had collected by measuring the heights of adult males chisen randomly from the population. Therefore, and depend on the data as well. It's highly likely that if we repeat this experiment of randomly selecting adult males and measuring their heights, you will get a different result. Hence, and are also random variables.  But population mean is one definitely number, which would the average of ALL males in the population. Although is not known but there is no uncertainty in what it is supposed to be. Therefore, we cannot say probability of to be in this interval is such and such - you can only assign probability to uncertain events and value of is not uncertain.  Here, the uncertainty is in the confidence interval since both and are uncertain, depending on the data instances. As you repeat collection of data, you will get different from each data instance. Some of those ranges will include some will not. The way to interpret Eq. is that among numerous repeats of data, of the time the range will include . Whether a particular interval will include the is either a yes or no answer, not a probability answer. You might ask about the probability of the event that any one of the confidence interval includes in it - the answer to that probability will be .    The t-Intervals  Let's study the same problem of determining the mean height of a population of male adults in a population. We had collected heights, and computed their sample mean to be . We know nothing about either population mean or population variance . But, in addition to , we can compute sample variance from the 's, e.g., the unbiased sample mean. where I have written everything in terms of the randdom variables for each sampling step, viz., with here, and Suppose, we found that the value of from our data, now denoted by the lower case letter, since we are talking about a value of . How would we now use and to find confidence interval at confidence level ?  This is where the theory of Student distribution comes handy since the following variable obes a distribution of degrees of freedom equal to .    Note that contstructed in Eq. is different than the we had defined when we discussed the -distribution. There, we had the degree of freedom of the -distribution in the definition. Let be the degree of freedom, in place of used nefore so that we separate the from samples taken to for the degree of freedom. where it was assumed that In Eq. we need to deal with , for which we will make use of the result: Let's now make the following manipulations of the expression in Eq. . Here, and Hence, we have the conditions for distribution for the defined variable: which is -distribution of degree of freedom equal .     shows parallels to and illustrates that all we need to do is to use distribution of degrees of freedom here. The parallels of here and in Eq. is obvious with only changes being The value of separating the included exluded excluded probabilities is very close to the corresponding values since we have a fairly high degree of freedom. The difference is atrtributable to the flat tail aspect of the distribution compared to the standard normal distribution.   Calculation for the upper and lower values of . The shaded areas on the two sides each have , here, area under the Gaussian curve. The confidence interval is to include the middle , here of the area.   Calculation for the upper and lower values of T.    Therefore, the confidence interval at confidence interval is Numerically, in our example, we will get That is the distribution will be      Difference Between Two Means   Often we are faced with trying to figure out if the means of two different populations are reasonably same, without having any clue about their actual values. Above we learned what to do with saying something definite about the mean of one population from a sample of the data. So, here we will want to say something about their difference.  For a simpler setting, as we did with the intervals, we could assume that we somehow know the population variances and further that they are equal. That appears to be too drastic an assumption. Instead, we will say, we want to get all the information from the data collected from the two populations. Except that we will assume the distributions are Gaussian\/Normal.  Let give names and to the variables in the two populations. Let their means be denoted by and , respectively, and similarly for their unknown variances. We wish to detemine -Confidence Interval of at the confidence level .  Let data collected from the two populations be   From these data we can evaluate their sample means and sample variances.     Equal Variance Populations: Pooled t-test  Suppose we suspect that for the two populations are identical, then we would pool the data and compute a single pooled best sample variance. Then the confidence interval on can be deduced from the following theorem.    Let two random variables and be normal distributions with equal variances. From the data and , we compute the their sample means , and pooled variance, . Then, the following variable has a -distribution.     We are not going to prove this theorem. From this -distribution, however, you can discover the confidence interval of the population mean difference will be where all the lower case letters for variables indicate the values obtained on the corresponding datasets.    Welch t test  The above described pooled variance works when the variances of the two populations being compared are equal. When we cannot make that assumption, we turn to the Welch's -test for guidance in setting confidence interval for the difference of means.  Let's start with the answer first. Welch showed that upto two moments of the distribution, the following variable has a -distribution of degree of freedom . where where and are sample standard deviation. Using this result, it is straight forward procedure, as shown for single mean case to find the confidence interval for the difference in means of two separate populations. The example at the end of this section illustrates a use of this formula.  The Hard Work:  To start with consider the difference in the sample means: The expectation and variance of is readily calculated from the expectation and variance of means. Therefore, we can construct a normal variable by subtracting the expectation value from and scaling by the square root of its variance. Therefore, the variable defined by should have the standard normal distribution. We can't use to construct a confidence interval that depends solely on the data collected since it has the unknown population variances and . What would happen if we replace them with sample variances? That is, we want to look at the following as a candidate, similar to the -confidence interval for mean from the -confidence interval we saw above: We want to convert this into a variable. First let us convert the numerator to . If this quantity could be transformed into the following form for some , this compisite quantity will go as -distribution. But, it is not possible to do that. So, Welch looked at a closely related quantity that can be turned into a -variable up to two moments of the distribution, i.e, only the mean and the variance. The quantity he constructed is Now, we demand that this be equal to a multiplier of a variable and try to solve for for equating mean and variances of the two sides. Taking expectation and variance of the two sides gives: Now, we can use the following to simplify. The last one is obtained from taking variance of the following and repacing by the value fo sample variance from the collected data. Now, we can rewrite Eqs. and and solve for . This equation is called the Welch Satterthwaite Equation . Now, we all the ingredients to construct a -distribution variable which we can use to find the required confidence interval.   After all this work, the Welch's leads to the following confidence interval at confidence level :     Suppose we wish to compare the population mean heights of adult males from two separate populations. We measure heights of adults from -population and from the -population. From the sample we calculate sample means and variance and find the following values. Find the confidence interval for the difference of their means at confidence level.  We first calculate the degree of freedom . For the confidence interval we need . for that we look up python scipy.stats.t.ppp(78.08, 0.025) for , i.e., confidence level. We get a value of . Therefore, the confidence interval is That is . This says that zero is not in this range. So, if the claim was that the two populations have the same population means, then we could reject that claim at this level of confidence. These decisions are formalized more in the hypothesis testing section.      Confidence Interval of Variance  If you were comparing two normal variables, you would naturally want to compare not only their means but also their variances to conclude if their distributions were more or less same or not. It turns out putting confidence interval on variance is readily done based on the fact that the following scaled sample variance of samples from a Gaussian distribution of population variance varies as . where We have used this result before in the context of comapring two means. I present a proof of this claim below, just for completeness sake. Its okay if you want to skip over the proof. I will work out the formula for the confidence interval after the proof.  So, how do we use Eq. to obtain a confidence interval on ? It's the same routine. We atart with the probability statement and then mainipulate the event to write it in a way so that we can read off the confidence interval.  I will flip the script a bit here, just for fun. We want so that In the last step, we replaced the composite quantity by a variable to make connection with the CDF and inverse CDF of distribution. The quantities and can be read off from a plot of the appropriate probability density, here as shown in , which are labeled in the figure as and , respectively. Therefore, substituting the value of from the data, writing it as , we have This gives the desired upper and lower ends of the confidence interval for the population variance .    Calculation for the upper and lower values of included probability of a PDF distribution. The shaded areas on the two sides each have , here, area under the curve. The confidence interval is to include the middle , here of the area. You use inverse of the CDF to find the values and .   Calculation for the upper and lower values of chi square.    Numerical Example (Continued)  In our heights example, let's look at the first population, where we had sampled adult males with sample mean height equal to and sample variance equal to . We want to get confidence interval for the population variance . First we use scipy.stats.chi2.ppf(x, df) with and for . These give the lower and upper values for the confidence interval to be with      Proof of Eq. .  Let's recall that if we have standard normal variables then sum of their squares has a distribution. This was shown in detail when we discussed the distribution.  For our proof of Eq. , we will start with independent and identically disrtibuted (i.i.d.) Gaussian variables 's that are . Let us construct normal variables from them and sum their squares. Let We want to introduce sample mean so that we can build the samole variance on the right side. We can do this by adding and subtracting in the numerator and expanding. The middle term became zero because . The last term is actually a standard normal variable since the sample mean goes as and we have subtracted the mean of this sample mean and scaled it by its standard deviation. Let's rename it . Then, we have the following equation For brevity in calculations below let us rename the first term on the right side . How can we get distributon of if we know the distributions of and ? Since the rrelations are additive, moment generative function helps here. Moment generating function is defined by expectation value of an exponential function. For example, for : When you work this out for distribution, yiu get The variable can be thought of , i.e., of a sum of standard normal with only one term. Hence, its' moment generating function is Now, we go back to Eq. to determine the moment generating function of :  This is just the moment generating function of a .      Comparing Variances of Two Normal Distributions  Ability to compare variances can help us decide how similar or different two normal distributions are. Suppose we have two popolations and . Suppose we have collected samples from and samples from from which we have computed the sample means and sample variances: We have already tackled the question of comparing means and by finding the confidence interval in . Now, we wish to find someway to compare to . It turns out that we can find the confidence interval on their ratio. where where is the inverse CDF of the -distribution and the the subscripts in its symbol list the numerator and denominator dimensions of the -distribution. We will derive the -distribution PDF below. But here, we present how you would use them in calculating the confidence interval.  Numerical Example  Let's go back to the heights of two population we have studied before. We had 50 samples from population with sample mean equal to and sample variance . We had samples from population with sample mean equal to and sample variance . What will be the , i.e., , confidence interval of ?   shows the included and excluded area and the values of the right and left edges of the excluded probabilities. Therefore, the lower and upper ends of the confidence interval is Interesting the value is in the range found here.   Calculation for the upper and lower values of included probability of a PDF distribution. The shaded areas on the two sides each have , here, area under the curve. The confidence interval is to include the middle , here of the area. You use inverse of the CDF to find the values and .   Calculation for the upper and lower values of chi square.     How did the formula come about?  We will need to convert and variables into distributions. Let From our previous discussions, you know that For brevity of writing, let us introduce the degrees of freedoms by From and and their degrees of freedoms, let us construct a new variable,  Then, as we will see below that has the following ugly-looking PDF: The is called the numerator variable and the denominator variable and and the numerator and denominator degrees of freedoms. This formula has already been programmed into stat packages. We just need to understand how to use it to get confidence interval. By now, I expect you to know the drill that you just look us the function, which takes in three parameters . Thus, we require the following. From the definition of CDF, , we can say that Now, we brign in the values of in terms of and : This shows the confidence interval of , both unknwon quantities.   Derivation of PDF of the F-distribution  First let's copy the basic variables from definitions. where and are independent variables. Since they are assumed to be independent, their joint probability will factorize. In transforming from to , we are seeking a probability along lines of in the -plane. The trick is to introduce another variable which is identical to then go from two-variable world of to a two-variable world of . You can then integrate out the to obtain the probability density of just .  The transformation between the two sets of variables are: By equating probability in an infinitesimal intervals in the and planes, we get where the subscript is instruction to cast the quantities within the brackets into variables using the transformations above and is the Jacobian. Since, we already know that and are and distributions, we can immediately write down the joint PDF of . Now, we separate out -dependent parts so we can integrate over this auxiliary variable. We get Integrating away just requires the following integral. Changing the variable to this integral becomes The integral part, now simply a function. Now, you can put all the factors together, which results in the cancellation of 's to obtain the PDF of the variable. This is of course normalized to probability as you can verify by doing the following integral.     "
},
{
  "id": "sec-Point-Estimates-and-Confidence-Intervals-2-2",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#sec-Point-Estimates-and-Confidence-Intervals-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Parameter Estimation Hypothesis Testing "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-5",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "likelihood function "
},
{
  "id": "thm-MLE-is-maximum",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#thm-MLE-is-maximum",
  "type": "Theorem",
  "number": "1.11.1",
  "title": "",
  "body": "  MLE estimate is maximum of the log likelihood function.    To show that the extremeum found is the maximum and not the minimum, all we have to do is to check whether the sign of the second derivative when negative or not. Hence, MLE estimate is indeed a maximum of Likelihood function.    "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-7",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-7",
  "type": "Remark",
  "number": "1.11.2",
  "title": "What if <span class=\"process-math\">\\(X\\)<\/span> was not Bernoulli?",
  "body": "What if was not Bernoulli?  We just have to proceed a little differently after Eq. . In that equation, parameters, usually denoted by , of interest will appear in the probability distributions of the individual trials, same for every one of them. Then, the likelihood function will just be The loglikelihood In general, setting the derivative of to zero and solving for the MLE estimate can only be done numercally.  "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-8",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-8",
  "type": "Example",
  "number": "1.11.3",
  "title": "Each Trial <span class=\"process-math\">\\(X \\sim \\mathcal{N}(\\mu, \\sigma^2)\\)<\/span> of unknown <span class=\"process-math\">\\(\\mu\\)<\/span> and <span class=\"process-math\">\\(\\sigma^2\\)<\/span>.",
  "body": " Each Trial of unknown and   We have   In this case, suppose we want an MLE of and . The log likelihood function will be (replacing by the actual parameters here in Eq. .) Now, to find and you will solve the following equations. Show that you get the following answers. The MLE estimate of the variance is a biased estimator. Often, we will use the following unbiased estimate for variance, also called the sample variance . Below we discuss the technical meaning of bias and unbias.  "
},
{
  "id": "def-estimators",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#def-estimators",
  "type": "Definition",
  "number": "1.11.4",
  "title": "Estimators.",
  "body": "Estimators   From the formulas for MLE etimates of of a Bernoulli and and of a Gaussian , we can write random variables whose expectation values estimate these quantities. We call these random variables estimators . We will denote them with capital letters as we have done for other random variables.  Thus the MLE estimator of of a Bernoulli variable will be Similarly, the MLE estimators of and for a Gaussian\/Normal variable will be If expectation values of these estimators in their corresponding probability distributions yield the appropriate true values, for the Bernoulli case, and and for the Normal case, we say that the estimator is an unbiased estimartor .   "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-10",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-10",
  "type": "Proof",
  "number": "1.11.1.2.1",
  "title": "Bernoulli <span class=\"process-math\">\\(\\Pi_\\text{MLE}\\)<\/span> estimator is unbiased..",
  "body": "Bernoulli estimator is unbiased.  To prove that is unbiased, we calculate it\/s expectation value in the Bernoulli PMF. We know that for expectation value of each is same: Therefore, Hence is an unbiased estimator of .   "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-11",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-11",
  "type": "Proof",
  "number": "1.11.1.2.2",
  "title": "Gaussian <span class=\"process-math\">\\(M_\\text{MLE}\\)<\/span> estimator is unbiased..",
  "body": "Gaussian estimator is unbiased.  Here, we know that for each . Hence, just like the Bernoulli , the expectation value of is also unbiased.  "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-12",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-12",
  "type": "Proof",
  "number": "1.11.1.2.3",
  "title": "Gaussian <span class=\"process-math\">\\(\\Sigma_\\text{MLE}^2\\)<\/span> estimator is not unbiased..",
  "body": "Gaussian estimator is not unbiased.  Now, we come to an estimator which is not unbiased. Let's see. Since each , the variance of each , When we expand the parenthesis in Eq. , we will find that we also need: Here is how I got this result. Another term that will show up in our calculation is I leave this part to you to complete. Now, we look at the expectation of the estimator . Hence is not an unbiased estimator of of a Normal variable.   "
},
{
  "id": "def-unbiased-sample-variance",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#def-unbiased-sample-variance",
  "type": "Definition",
  "number": "1.11.5",
  "title": "Unbiased Sample Variance.",
  "body": "Unbiased Sample Variance   From the result above, it is immediately obvious that had we divided by in the definition of rather than by , the modified quantity, to be denoted by will be an unbiased estimator of . We call this quantity Sample Variance . It is a random variable since it depends upon the random variables 's and .   "
},
{
  "id": "subsubsec-Maximum-Likelihood-Estimate-14",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Maximum-Likelihood-Estimate-14",
  "type": "Proof",
  "number": "1.11.1.2.4",
  "title": "<span class=\"process-math\">\\(S^2\\)<\/span> is unbiased.",
  "body": "is unbiased  You can verify that expectation value of sample variance will be the (unknown) population variance by a more direct calculation but a less algebraic way of proving the unbiasedness of is to notice its relation to chi square disrtribution.  Consider a Gaussian variable with mean and variance and sample variance defined by formula defined above. Then, following is true of a related variable : Let's use symbol for to remind us of its distribution. And, rearrange the above relation to Clearly, the expectation value of will be simply But the expectation value of variable is just its degree of freedom, which is here. Hence, Interestingly, although is an unbiased estimator of , its square root is not an unbiased estimator of .  "
},
{
  "id": "subsec-Confidence-Interval-2-2",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Confidence-Interval-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "confidence interval "
},
{
  "id": "subsec-Confidence-Interval-2-3",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Confidence-Interval-2-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "confidence coefficient "
},
{
  "id": "def-confidence-interval",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#def-confidence-interval",
  "type": "Definition",
  "number": "1.11.6",
  "title": "Confidence Interval.",
  "body": "Confidence Interval   A confidence interval of an unknown parameter of a statistical model is defined by the probability of the event that the interval contains the true value of the parameter be equal to the desired confidence level. That is, We will see examples of how we can compute and from the data collected.   "
},
{
  "id": "subsubsec-Z-Intervals-2",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Z-Intervals-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Z interval "
},
{
  "id": "subsubsec-Z-Intervals-6",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Z-Intervals-6",
  "type": "Remark",
  "number": "1.11.7",
  "title": "The Answer.",
  "body": "The Answer  Let me just give you the answer first and then we will look at the theory based on our thinking that the distribution is more or less like a Gaussian distribution.   Assuming, the included probability is the central part of a Gaussian, we can implement the probability required in Eq. by using the standard normal distribution's CDF, .  We want to include amount of central part of Gaussian and leave out equal amount, say on either side, with total being . That gives . Hence, to get value at the upper end, we need to find which gives probability . That is, we need to solve the following problems for : This will be For , you can use a numeerical tool such as scipy.stats.norm.ppf(k). When I looked up the number for , the value was approximately 1.96.  We call this value . From symmetry of the standard normal about , we arrive at     Calculate the high and low limits of the confidence interval by      So, that's all there is to the -confidence interval.  "
},
{
  "id": "fig-confidence-interval",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#fig-confidence-interval",
  "type": "Figure",
  "number": "1.11.8",
  "title": "",
  "body": " Calculation for the upper and lower values of . The shaded areas on the two sides each have , here, area under the Gaussian curve. The confidence interval is to include the middle , here of the area.   Calculation for the upper and lower values of Z.   "
},
{
  "id": "subsubsec-Z-Intervals-10",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-Z-Intervals-10",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Z interval "
},
{
  "id": "subsubsec-t-Intervals-4",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsubsec-t-Intervals-4",
  "type": "Proof",
  "number": "1.11.2.2.1",
  "title": "",
  "body": " Note that contstructed in Eq. is different than the we had defined when we discussed the -distribution. There, we had the degree of freedom of the -distribution in the definition. Let be the degree of freedom, in place of used nefore so that we separate the from samples taken to for the degree of freedom. where it was assumed that In Eq. we need to deal with , for which we will make use of the result: Let's now make the following manipulations of the expression in Eq. . Here, and Hence, we have the conditions for distribution for the defined variable: which is -distribution of degree of freedom equal .   "
},
{
  "id": "fig-t-confidence-interval",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#fig-t-confidence-interval",
  "type": "Figure",
  "number": "1.11.9",
  "title": "",
  "body": " Calculation for the upper and lower values of . The shaded areas on the two sides each have , here, area under the Gaussian curve. The confidence interval is to include the middle , here of the area.   Calculation for the upper and lower values of T.   "
},
{
  "id": "thm-pooled-sample-variance-t-distribution",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#thm-pooled-sample-variance-t-distribution",
  "type": "Theorem",
  "number": "1.11.10",
  "title": "",
  "body": "  Let two random variables and be normal distributions with equal variances. From the data and , we compute the their sample means , and pooled variance, . Then, the following variable has a -distribution.    "
},
{
  "id": "subsec-Welch-t-test-4",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Welch-t-test-4",
  "type": "Proof",
  "number": "1.11.3.2.1",
  "title": "The Hard Work:.",
  "body": "The Hard Work:  To start with consider the difference in the sample means: The expectation and variance of is readily calculated from the expectation and variance of means. Therefore, we can construct a normal variable by subtracting the expectation value from and scaling by the square root of its variance. Therefore, the variable defined by should have the standard normal distribution. We can't use to construct a confidence interval that depends solely on the data collected since it has the unknown population variances and . What would happen if we replace them with sample variances? That is, we want to look at the following as a candidate, similar to the -confidence interval for mean from the -confidence interval we saw above: We want to convert this into a variable. First let us convert the numerator to . If this quantity could be transformed into the following form for some , this compisite quantity will go as -distribution. But, it is not possible to do that. So, Welch looked at a closely related quantity that can be turned into a -variable up to two moments of the distribution, i.e, only the mean and the variance. The quantity he constructed is Now, we demand that this be equal to a multiplier of a variable and try to solve for for equating mean and variances of the two sides. Taking expectation and variance of the two sides gives: Now, we can use the following to simplify. The last one is obtained from taking variance of the following and repacing by the value fo sample variance from the collected data. Now, we can rewrite Eqs. and and solve for . This equation is called the Welch Satterthwaite Equation . Now, we all the ingredients to construct a -distribution variable which we can use to find the required confidence interval.   After all this work, the Welch's leads to the following confidence interval at confidence level :   "
},
{
  "id": "subsec-Welch-t-test-5",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Welch-t-test-5",
  "type": "Example",
  "number": "1.11.11",
  "title": "",
  "body": " Suppose we wish to compare the population mean heights of adult males from two separate populations. We measure heights of adults from -population and from the -population. From the sample we calculate sample means and variance and find the following values. Find the confidence interval for the difference of their means at confidence level.  We first calculate the degree of freedom . For the confidence interval we need . for that we look up python scipy.stats.t.ppp(78.08, 0.025) for , i.e., confidence level. We get a value of . Therefore, the confidence interval is That is . This says that zero is not in this range. So, if the claim was that the two populations have the same population means, then we could reject that claim at this level of confidence. These decisions are formalized more in the hypothesis testing section.  "
},
{
  "id": "fig-sigma-square-chi2-confidence-interval",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#fig-sigma-square-chi2-confidence-interval",
  "type": "Figure",
  "number": "1.11.12",
  "title": "",
  "body": " Calculation for the upper and lower values of included probability of a PDF distribution. The shaded areas on the two sides each have , here, area under the curve. The confidence interval is to include the middle , here of the area. You use inverse of the CDF to find the values and .   Calculation for the upper and lower values of chi square.   "
},
{
  "id": "subsec-Estimating-Variance-6",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Estimating-Variance-6",
  "type": "Example",
  "number": "1.11.13",
  "title": "Numerical Example (Continued).",
  "body": "Numerical Example (Continued)  In our heights example, let's look at the first population, where we had sampled adult males with sample mean height equal to and sample variance equal to . We want to get confidence interval for the population variance . First we use scipy.stats.chi2.ppf(x, df) with and for . These give the lower and upper values for the confidence interval to be with     "
},
{
  "id": "subsec-Estimating-Variance-7",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Estimating-Variance-7",
  "type": "Proof",
  "number": "1.11.4.1",
  "title": "Proof of Eq. (1.11.39)..",
  "body": "Proof of Eq. .  Let's recall that if we have standard normal variables then sum of their squares has a distribution. This was shown in detail when we discussed the distribution.  For our proof of Eq. , we will start with independent and identically disrtibuted (i.i.d.) Gaussian variables 's that are . Let us construct normal variables from them and sum their squares. Let We want to introduce sample mean so that we can build the samole variance on the right side. We can do this by adding and subtracting in the numerator and expanding. The middle term became zero because . The last term is actually a standard normal variable since the sample mean goes as and we have subtracted the mean of this sample mean and scaled it by its standard deviation. Let's rename it . Then, we have the following equation For brevity in calculations below let us rename the first term on the right side . How can we get distributon of if we know the distributions of and ? Since the rrelations are additive, moment generative function helps here. Moment generating function is defined by expectation value of an exponential function. For example, for : When you work this out for distribution, yiu get The variable can be thought of , i.e., of a sum of standard normal with only one term. Hence, its' moment generating function is Now, we go back to Eq. to determine the moment generating function of :  This is just the moment generating function of a .   "
},
{
  "id": "subsec-Difference-in-Variance-of-Two-Normal-Distributions-3",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Difference-in-Variance-of-Two-Normal-Distributions-3",
  "type": "Example",
  "number": "1.11.14",
  "title": "Numerical Example.",
  "body": "Numerical Example  Let's go back to the heights of two population we have studied before. We had 50 samples from population with sample mean equal to and sample variance . We had samples from population with sample mean equal to and sample variance . What will be the , i.e., , confidence interval of ?   shows the included and excluded area and the values of the right and left edges of the excluded probabilities. Therefore, the lower and upper ends of the confidence interval is Interesting the value is in the range found here.   Calculation for the upper and lower values of included probability of a PDF distribution. The shaded areas on the two sides each have , here, area under the curve. The confidence interval is to include the middle , here of the area. You use inverse of the CDF to find the values and .   Calculation for the upper and lower values of chi square.    "
},
{
  "id": "subsec-Difference-in-Variance-of-Two-Normal-Distributions-4",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Difference-in-Variance-of-Two-Normal-Distributions-4",
  "type": "Remark",
  "number": "1.11.16",
  "title": "How did the formula (1.11.42) come about?",
  "body": "How did the formula come about?  We will need to convert and variables into distributions. Let From our previous discussions, you know that For brevity of writing, let us introduce the degrees of freedoms by From and and their degrees of freedoms, let us construct a new variable,  Then, as we will see below that has the following ugly-looking PDF: The is called the numerator variable and the denominator variable and and the numerator and denominator degrees of freedoms. This formula has already been programmed into stat packages. We just need to understand how to use it to get confidence interval. By now, I expect you to know the drill that you just look us the function, which takes in three parameters . Thus, we require the following. From the definition of CDF, , we can say that Now, we brign in the values of in terms of and : This shows the confidence interval of , both unknwon quantities.  "
},
{
  "id": "subsec-Difference-in-Variance-of-Two-Normal-Distributions-5",
  "level": "2",
  "url": "sec-Point-Estimates-and-Confidence-Intervals.html#subsec-Difference-in-Variance-of-Two-Normal-Distributions-5",
  "type": "Proof",
  "number": "1.11.5.1",
  "title": "Derivation of PDF of the F-distribution.",
  "body": "Derivation of PDF of the F-distribution  First let's copy the basic variables from definitions. where and are independent variables. Since they are assumed to be independent, their joint probability will factorize. In transforming from to , we are seeking a probability along lines of in the -plane. The trick is to introduce another variable which is identical to then go from two-variable world of to a two-variable world of . You can then integrate out the to obtain the probability density of just .  The transformation between the two sets of variables are: By equating probability in an infinitesimal intervals in the and planes, we get where the subscript is instruction to cast the quantities within the brackets into variables using the transformations above and is the Jacobian. Since, we already know that and are and distributions, we can immediately write down the joint PDF of . Now, we separate out -dependent parts so we can integrate over this auxiliary variable. We get Integrating away just requires the following integral. Changing the variable to this integral becomes The integral part, now simply a function. Now, you can put all the factors together, which results in the cancellation of 's to obtain the PDF of the variable. This is of course normalized to probability as you can verify by doing the following integral.   "
},
{
  "id": "sec-Hypothesis-Tests",
  "level": "1",
  "url": "sec-Hypothesis-Tests.html",
  "type": "Section",
  "number": "1.12",
  "title": "Hypothesis Tests",
  "body": " Hypothesis Tests   Often we want to know if a new data contradicts an established belief presented as a hypothesis about the value(s) of one or more parameters of a statistical model. hypothesis testing provides a process for determining whether a hypothesis should be rejected based upon the data. This topic is of fundamental importance in applications of statistics to real life.  Suppose we have a statistical distribution of a random variable , given by either a PMF or a PDf or a CDF which contains a parameter. For example,it may be a Binomial distribution with parameters and . A hypothesis here may be about the value of for some fixed , say . Following are examples of hypotheses. Hypthesis is an example of a simple hypothesis since it completely defines the distribution to be . In caqse of a simple hypothesis, we can also write hypothesis in a distribution notation. Hypothesis and (or ) is an example of composite hopthesis since there are more than one distribution depending upon the value of , e.g., distributions and satisfy .  There are basically two approaches to implement hypothesis testing as we will illustrate in the following sections.    Rejection region or critical region or critical value     The p-Value.             Hypothesis tests usually follow a pattern of three steps:   State a null hypothesis , denoted by , which will usually be a simple hypothesis and an alternate hypothesis , denoted by either or . There is no expectation that alternate hypothesis be just the complement of the null hypothesis. For instance, and , which leaves the possibility that unaccounted for.    We now, need to decide on a test statistic , which is a random variable. By a test statisic, we mean a function of the data, , whose distribution can be used to find the rejection\/critical region for the rejection of the hypothesis. We then, collect data and compute the test statistic's value on that data.  For instance, sample mean will be the statistic for the proportion in the Binomial distribution. Another test statistic for the would be the normalized sample mean.     Use probability theory to draw conclusion whether null hypothesis can be rejected, either because the data was in the rejection region or that the p Value was small enough.         Simple Tests using Rejection Regions   Problem: Suppose we want to know if a coin is a fair coin. We can start with the assumption that the coin is a fair coin, meaning the probability of getting Heads ( ) is equal to the probability of getting Tails ( ).    The Hypotheses: Let us use the symbol for the number , just to comply with notation of using the subscript with anything that is the null hypothesis. Writing in terms of a null hypothesis and an alternate hypothesis, we will express it as follows. The null hypothesis is a simple hypothesis and it is of course specifying the Bernoulli distribution completely and we can write it more explicitly as follows. Hypothesis is a composite hypothesis. We can pick any value of other than to specify one of the infintely many Bernoulli distributons corresponding to the hypothesis .  Since is asking us to look both above and below , this test is called a two-tailed test . If alternate hypothesis was claiming that , we will look only the region above and that would have been a one-tail test . Similarly for the case .   The Data and the Statistic: To obtain the data, let's toss the coin times. Suppose, we get Heads and Tails. We use real number for the Heads and for the Tails.  Our hypothesis is about the proportion , which can be estimated by th sample mean of the random variables, from our data collection process as we have seen before. Using the data we get Clearly, . So, should we say that claim is false? Not, so fast. We need to decide on the criteria upon which we will decide whether this value is too far from . For that we use probability theory.   Use probability theory: Assuming is true, we can work out its consequences and compare how the test statistic fits in with those consequences. Here, our experiment for finding is a scaled experiment since Assuming to be true, With large , specifically, if and , which is the case here since , the sample mean of the data can be seen to follow a Gaussian distribution where That is, if is true, we expect sample mean, It's almost always easier to work with a standard normal variable , which can be obtained by subtracting away the mean and scaling by the square root of the variance. A plot of is shown in . We can visually see the areas that have to be included or exclued at a particluar confidence level.  While we have talked about confidence level where we discussed the confidence interval, here, is called significance level of the test. We will see below that it represents the probability of error, so-called the type I error, that we would reject the hypothesis when in actuality it ws true.  Suppose, we choose the significane level . Our rejection values of will be on either side of that are far away so that the net probability in the rejection regions add up to . We use the inverse CDF of the standard normal to find the rejection regions in the -sampe space, which can of course, be translated in the original sample space of . We need to translate these critical values of back to the sample space through the value of . Our data gave us the value as shown in Eq. . This value is not in either of the rejection regions. Hence, we will not reject the null hypothesis in favor of , even though this value is in the domain of . Had we obtained Heads, out of tosses of the coin, we would have rejected the hypothesis.   Calculation for rejection values of . The shaded areas on the two sides each have , here, area under the Gaussian curve. We see that if from our data were or less than , we will reject .   Calculation for rejection values of Z.    Case of a One-Tail Test  Suppose, instead of the Alternate Hypothesi in Eq. , we had a different Alternate Hypothesis but the same null hypothesis. With the same experimental result of Heads in tosses, as above, let us decide whether we can reject at the same level. Going through the same steps, but now, we need the entire of the rejection region will now be all in the direction . That gives the critical value of . This gives the rejection region in p to be: The value of is in the rejection region. Hence, we reject that the coin is fair in favor of the alternate hypothesis . Same data but different alternate hypothesis gives you a different decision!   What if we had chosen a different level?  A short answer is you might have a different conclusion about rejecting . That is why you cannot just go -shopping till you get the decision you want. (If you want to do -shopping, you better do the p-Value as I explain below.) You must first set the value BEFORE you do the analysis. For instance, if you had set in the two-tail case above, you would have rejected while you were not able to reject at . If you had set , you wouldn't have rejected either. So, your conclusion does depend on you level of confidence you desire. Recall the level of confidence in complement of , i.e., .     Type I and TypeII Errors  In the example above regarding whether the coin is fair or not, we analyzed the data collected from tosses assuming the coin was fair (the null hypothesis, ), and found that at significance level we were unable to reject since our data had Heads. We also noticed that, had we got Heads, which is entirely possible, we would have rejected .  This shows that just a variation in data collection, can lead us to the opposite conclusions, regardless of whether happens to be true or not in reality. This makes two types of error possible in our decision.  We say that we have made a Type I Error if we rejected the null hypothesis in favor of even though it shouldn't have been. The probability of rejecting assuming is the correct distribution is what was. That is why is also called significance of the test. Of course, the complement of this probability will be not to reject . Rejecting when is true will be an error, i.e., is a level of error we would need to avoid.  A Type II Error occurs when we fail to reject in favor of when we should have rejected it. In the examples above, while was a simple hypothesis: the alternate hypothesis was a composite hypothesis, giving us distribution in terms of variable . To indicate the -dependence, let's write this hypothesis as So, based on the choice of the value of , we will get different probabilities of the event Do not reject , i.e., of making an error of this type. This probability is denoted by , which will depende on the value of . The compliment of this probability will be the probability that when is true, we do reject , as we should. This is called power of the test, usually denoted by letter . Since, there are too many uses of letter , we will just refrain from that practice here.    To evaluate a statistical hypothesis for some parameter , we define a critical region. This region is chosen to balance the probabilities of Type I and Type II errors, keeping both as low as feasible. However, reducing the Type I error probability ( ) typically increases the Type II error probability ( ), as these are inversely related. In most practical scenarios, controlling is prioritized due to its importance in hypothesis testing.  The process for selecting the critical region is as follows:   Specify a desired level for the Type I error probability, .    Identify a critical region that minimizes the Type II error probability, , for a specific alternative hypothesis parameter, .    If the resulting is unacceptably high, consider increasing to the maximum acceptable level.    If remains too large, increase the sample size to improve the test's power.     A hypothesis test is deemed most powerful if it achieves the smallest possible for a given . The critical region of a most powerful test generally depends on the specific alternative parameter .    Statistical Model and Example for Hypothesis Testing  This example introduces the concepts of Type I error probability ( ), Type II error probability ( ), and power ( ) using a hypothesis test for a normal distribution mean.  Statistical Model: Consider an independent random sample from a normal distribution , where the standard deviation is known. We test the null hypothesis against the one-sided alternative .  The test statistic is:     which follows under . The critical region for the test is to reject if , where is the upper -quantile of the standard normal distribution (e.g., ).  Recall key definitions:    , the probability of a false positive (Type I error), controlled by the test design.  Under the alternative , the test statistic follows , where .  , where is the standard normal cumulative distribution function, representing the Type II error probability.  Power: , the probability of correctly rejecting when is true.    This model provides exact calculations and a closed-form power function, making it ideal for illustrating these concepts even though, for simplicity, we assume known .  Example Illustration: Set , , , and . Then , and the non-centrality parameter is .    At , , as the power under the null equals the Type I error rate.  For a small effect, , , so , and .  For a larger effect, , , so , and .  As , .    To visualize these relationships, in the upper part of we plot and the power function ) against . This \"power curve\" illustrates how the test's ability to detect alternatives improves as moves away from . At , the power equals , then rises toward 1.   Type II error probability and power of the test( ) versus . Power rises as you go further away from the null hypothesis 's .   Power and beta plots.    For a specific alternative, say , the sampling distributions under and can be plotted in the lower part of , with the right-tail under shaded to show and the left-tail under up to the critical value shaded to show . The code that generated these figures is provided in program listing below the figure.   Sampling distributiions under and to visually show the areas corresponding to the Type I and Type II errors.   Sampling distributions H0 and H1 to illustrate Type I and Type II errors.     #Visualizing Type I and Type II Errors and Power in Hypothesis Testing #This example demonstrates plotting the power function and sampling #distributions for a one-sided z-test for a normal mean. import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Parameters mu0 = 0 # Null mean sigma = 1 # Known standard deviation n = 25 # Sample size alpha = 0.05 # Type I error rate z_alpha = norm.ppf(1 - alpha) # Critical value, approx 1.645 # Function to compute power and beta for a given theta def power_theta(theta, mu0, sigma, n, z_alpha): delta = (theta - mu0) * np.sqrt(n) \/ sigma beta = norm.cdf(z_alpha - delta) power = 1 - beta return power, beta # Range of theta values for power curve (theta >= mu0) theta_values = np.linspace(mu0, 1, 100) # Compute power and beta for each theta powers = np.array([power_theta(theta, mu0, sigma, n, z_alpha)[0] for theta in theta_values]) betas = np.array([power_theta(theta, mu0, sigma, n, z_alpha)[1] for theta in theta_values]) # Plot 1: Power curve and beta vs theta plt.figure(figsize=(10, 6)) plt.plot(theta_values, powers, label='Power(theta) = 1 - beta(thet', color='blue')) plt.plot(theta_values, betas, label='beta(theta)', color='red', linestyle='--') plt.axhline(y=alpha, color='green', linestyle=':', label='alpha = 0.05') plt.axvline(x=mu0, color='black', linestyle='-', label='H0: mu = mu0') plt.title('Power and beta as Functions of Alternative Mean theta') plt.xlabel('Alternative Mean theta (under H1: mu \\gt mu0)') plt.ylabel('Probability') plt.legend() plt.grid(True) plt.show() # Fixed theta for distribution plot (example: theta = 0.5) fixed_theta = 0.5 delta_fixed = (fixed_theta - mu0) * np.sqrt(n) \/ sigma critical_value = z_alpha # On the Z-scale # Z values for plotting density z_values = np.linspace(-4, 6, 500) # Densities density_h0 = norm.pdf(z_values, loc=0, scale=1) # Under H0: N(0,1) density_h1 = norm.pdf(z_values, loc=delta_fixed, scale=1) # Under H1: N(delta,1) # Plot 2: Sampling distributions with shaded α and β plt.figure(figsize=(10, 6)) plt.plot(z_values, density_h0, label='Under H0: N(0,1)', color='blue') plt.plot(z_values, density_h1, label=f'Under H1: N({delta_fixed:.2f},1) for theta={fixed_theta}', color='red') # Shade alpha: Rejection region under H0 (right of critical_value) plt.fill_between(z_values, density_h0, where=(z_values \\geq critical_value), color='blue', alpha=0.3, label='alpha (Type I error)') # Shade beta: Acceptance region under H1 (left of critical_value) plt.fill_between(z_values, density_h1, where=(z_values \\leq critical_value), color='red', alpha=0.3, label='beta (Type II error)') plt.axvline(x=critical_value, color='black', linestyle='--', label=f'Critical value z_alpha ≈ {z_alpha:.3f}') plt.title(f'Sampling Distributions Showing alpha and beta for theta = {fixed_theta}') plt.xlabel('Test Statistic Z') plt.ylabel('Density') plt.legend() plt.grid(True) plt.show()    The p Value Approach  In the examples of rejection region approach, we saw that you failed to reject the null hypothesis if was too low. That is, there is a largest significance lavel at which we will be able to reject the null hypothsis, . That is called the value of the test.  Traditional low values such as or is considered statistically significant. Recall that we set the value of or BEFORE we do the analysis. Here we would do the same and demand that the value obtained from the data be less than for the test to be able to reject .   Let's look at the same example as the two-tail test presented above. Due to the misunderstanding of letter being used for proportion in the example and the same letter used for value, I will change the proportion symbol to and the corresponding random variable to . The hypotheses were The data from tosses had shown that . We had introduced a standard normal variable to implement expectations assuming were true. The information about being true is in the choice of the numerical values and . The value corresponding to the data is obtained when we put the value fo the sample mean in this statistic. By inserting this in the CDF of the standard normal we find Because of the two-tail nature of the alternate hypothesis, this must correspond to since we would have the probability above it. This gives . That is, the value will be . This is too high value for rejecting since , the threshold value.  Of course, you can, then, wrongly turn around, and say, let's go back and set just so that you can use your data to reject in favor of . That is why, people insist that for rejecting some well-established fact embodied in the null hypothesis, your value must be very very small. From an NIH paper:    The best practice is to report all values for all variables within a study design rather than only providing values for variables with significant findings. Including all values provides evidence for study validity and limits suspicion for selective reporting\/data mining.     Chi Square Goodness of Fit Test  Suppose we are dealing with a -class random variable or a real-valued variable which has been binned into bins. Each new data point is placed into one of the cartegories and the random variables for the bins are the counts of samples in each class\/bin. Such systems are governed by a multinomial distribution over the random variables which are constrained to add up to the total number of data points. where each variable can rane between and . We call the collection of the variables a multinomial random variable and will denote it in vector notation. Let be the probability of next data to be in the class (bin) and . Then, we can write the distribution expected as By this, we mean the following probability mass function (PMF.) See the discussion on multinomial distribution in the section on discrete probability distributions for the following results. Let the null hypothesis be Let the alternate hypothesis be The test is performed by Pearson test statistic  : For large values of , it can be shown, although with some difficulty, that the Pearson test statistic has a distribution with degrees of freedm equal to , one less than the number of classes\/bins. Thus, to find the region for rejecting with significance level , we just need to find the rejection region above . The value of will be such that following probability statement is true. Using the inverse CDF of , this will be We use symbol for the value rather than .  Is the Die a Fair Die?  Suppose we roll a six-sided die times and find the following numbers of faces 1 through 6: . Is the die fair at significance level ?  By a fair die, we mean each side should have the proportion: With , for every side, we get . Now, we calculate the Pearson test statistic: Now, we compare this to the critical value at the significance level which comes from the inverse distribution. Since , we can reject the fair die hypothesis .   Testing a Distribution  Suppose we want to test if the random number generator that generates real number in the interval is uniformly distributed at significance . Now, we do not have classes like we did in the six-sided die. So, we create bins. Let us create bins of equal widths of and whenever we get a random number, we will place it in one of the bins.  Suppose, we obtain samples tht result in the following frequencies: . Our hypothesis is This will give . Now, we calculate the Pearson test statistic. The critical value here is Since , we do not reject , i.e., the claim that the random number generator that gave us the values we observed is uniform at the significance level .     "
},
{
  "id": "sec-Hypothesis-Tests-2-2",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#sec-Hypothesis-Tests-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "simple hypothesis composite hopthesis "
},
{
  "id": "sec-Hypothesis-Tests-2-6",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#sec-Hypothesis-Tests-2-6",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "null hypothesis alternate hypothesis test statistic test statisic, "
},
{
  "id": "subsec-Hypothesis-Test-for-One-Proportion-4",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-Hypothesis-Test-for-One-Proportion-4",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "two-tailed test one-tail test "
},
{
  "id": "subsec-Hypothesis-Test-for-One-Proportion-8",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-Hypothesis-Test-for-One-Proportion-8",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "significance level "
},
{
  "id": "fig-z-rejection-zones",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#fig-z-rejection-zones",
  "type": "Figure",
  "number": "1.12.1",
  "title": "",
  "body": " Calculation for rejection values of . The shaded areas on the two sides each have , here, area under the Gaussian curve. We see that if from our data were or less than , we will reject .   Calculation for rejection values of Z.   "
},
{
  "id": "subsec-Hypothesis-Test-for-One-Proportion-11",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-Hypothesis-Test-for-One-Proportion-11",
  "type": "Remark",
  "number": "1.12.2",
  "title": "Case of a One-Tail Test.",
  "body": "Case of a One-Tail Test  Suppose, instead of the Alternate Hypothesi in Eq. , we had a different Alternate Hypothesis but the same null hypothesis. With the same experimental result of Heads in tosses, as above, let us decide whether we can reject at the same level. Going through the same steps, but now, we need the entire of the rejection region will now be all in the direction . That gives the critical value of . This gives the rejection region in p to be: The value of is in the rejection region. Hence, we reject that the coin is fair in favor of the alternate hypothesis . Same data but different alternate hypothesis gives you a different decision!  "
},
{
  "id": "subsec-Hypothesis-Test-for-One-Proportion-12",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-Hypothesis-Test-for-One-Proportion-12",
  "type": "Remark",
  "number": "1.12.3",
  "title": "What if we had chosen a different <span class=\"process-math\">\\(\\alpha\\)<\/span> level?",
  "body": "What if we had chosen a different level?  A short answer is you might have a different conclusion about rejecting . That is why you cannot just go -shopping till you get the decision you want. (If you want to do -shopping, you better do the p-Value as I explain below.) You must first set the value BEFORE you do the analysis. For instance, if you had set in the two-tail case above, you would have rejected while you were not able to reject at . If you had set , you wouldn't have rejected either. So, your conclusion does depend on you level of confidence you desire. Recall the level of confidence in complement of , i.e., .  "
},
{
  "id": "subsec-TypeI-and-TypeII-Errors-4",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-TypeI-and-TypeII-Errors-4",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Type I Error significance "
},
{
  "id": "subsec-TypeI-and-TypeII-Errors-5",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-TypeI-and-TypeII-Errors-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Type II Error power "
},
{
  "id": "subsec-TypeI-and-TypeII-Errors-6",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-TypeI-and-TypeII-Errors-6",
  "type": "Remark",
  "number": "1.12.4",
  "title": ".",
  "body": " To evaluate a statistical hypothesis for some parameter , we define a critical region. This region is chosen to balance the probabilities of Type I and Type II errors, keeping both as low as feasible. However, reducing the Type I error probability ( ) typically increases the Type II error probability ( ), as these are inversely related. In most practical scenarios, controlling is prioritized due to its importance in hypothesis testing.  The process for selecting the critical region is as follows:   Specify a desired level for the Type I error probability, .    Identify a critical region that minimizes the Type II error probability, , for a specific alternative hypothesis parameter, .    If the resulting is unacceptably high, consider increasing to the maximum acceptable level.    If remains too large, increase the sample size to improve the test's power.     A hypothesis test is deemed most powerful if it achieves the smallest possible for a given . The critical region of a most powerful test generally depends on the specific alternative parameter .  "
},
{
  "id": "subsec-TypeI-and-TypeII-Errors-7",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-TypeI-and-TypeII-Errors-7",
  "type": "Example",
  "number": "1.12.5",
  "title": "Statistical Model and Example for Hypothesis Testing.",
  "body": " Statistical Model and Example for Hypothesis Testing  This example introduces the concepts of Type I error probability ( ), Type II error probability ( ), and power ( ) using a hypothesis test for a normal distribution mean.  Statistical Model: Consider an independent random sample from a normal distribution , where the standard deviation is known. We test the null hypothesis against the one-sided alternative .  The test statistic is:     which follows under . The critical region for the test is to reject if , where is the upper -quantile of the standard normal distribution (e.g., ).  Recall key definitions:    , the probability of a false positive (Type I error), controlled by the test design.  Under the alternative , the test statistic follows , where .  , where is the standard normal cumulative distribution function, representing the Type II error probability.  Power: , the probability of correctly rejecting when is true.    This model provides exact calculations and a closed-form power function, making it ideal for illustrating these concepts even though, for simplicity, we assume known .  Example Illustration: Set , , , and . Then , and the non-centrality parameter is .    At , , as the power under the null equals the Type I error rate.  For a small effect, , , so , and .  For a larger effect, , , so , and .  As , .    To visualize these relationships, in the upper part of we plot and the power function ) against . This \"power curve\" illustrates how the test's ability to detect alternatives improves as moves away from . At , the power equals , then rises toward 1.   Type II error probability and power of the test( ) versus . Power rises as you go further away from the null hypothesis 's .   Power and beta plots.    For a specific alternative, say , the sampling distributions under and can be plotted in the lower part of , with the right-tail under shaded to show and the left-tail under up to the critical value shaded to show . The code that generated these figures is provided in program listing below the figure.   Sampling distributiions under and to visually show the areas corresponding to the Type I and Type II errors.   Sampling distributions H0 and H1 to illustrate Type I and Type II errors.    "
},
{
  "id": "subsec-chi-sq-goodness-of-fit-test-3",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-chi-sq-goodness-of-fit-test-3",
  "type": "Example",
  "number": "1.12.8",
  "title": "Is the Die a Fair Die?",
  "body": "Is the Die a Fair Die?  Suppose we roll a six-sided die times and find the following numbers of faces 1 through 6: . Is the die fair at significance level ?  By a fair die, we mean each side should have the proportion: With , for every side, we get . Now, we calculate the Pearson test statistic: Now, we compare this to the critical value at the significance level which comes from the inverse distribution. Since , we can reject the fair die hypothesis .  "
},
{
  "id": "subsec-chi-sq-goodness-of-fit-test-4",
  "level": "2",
  "url": "sec-Hypothesis-Tests.html#subsec-chi-sq-goodness-of-fit-test-4",
  "type": "Example",
  "number": "1.12.9",
  "title": "Testing a Distribution.",
  "body": "Testing a Distribution  Suppose we want to test if the random number generator that generates real number in the interval is uniformly distributed at significance . Now, we do not have classes like we did in the six-sided die. So, we create bins. Let us create bins of equal widths of and whenever we get a random number, we will place it in one of the bins.  Suppose, we obtain samples tht result in the following frequencies: . Our hypothesis is This will give . Now, we calculate the Pearson test statistic. The critical value here is Since , we do not reject , i.e., the claim that the random number generator that gave us the values we observed is uniform at the significance level .  "
},
{
  "id": "sec-Stats-Miscellaneous-Topics",
  "level": "1",
  "url": "sec-Stats-Miscellaneous-Topics.html",
  "type": "Section",
  "number": "1.13",
  "title": "Miscellaneous Topics",
  "body": " Miscellaneous Topics   Probably this and the next section will be an Appendix to this chapter.     Discrete Approximation to the Cumulative Distribution Function via Binning  We begin by recalling the cumulative distribution function (CDF) for a continuous random variable, then describe how to construct a discrete approximation through binning.  For a continuous random variable with probability density function (PDF) , the CDF is the probability that is less than or equal to :   The PDF is the derivative of the CDF:   The CDF is non-decreasing, continuous, and ranges from 0 to 1 for continuous .  To create a discrete version, we divide the real line into mutually exclusive and exhaustive bins with boundaries . The -th bin is the interval , with the first bin and the last .  The probability mass for the -th bin is: where is the continuous CDF. These satisfy and .  Define a discrete random variable taking values with probability mass function (PMF) , where if falls in the -th bin. The CDF of , denoted , is: for , where is the floor function. As is discrete, is a step function, jumping by at each integer from 1 to .  The discrete CDF coarsely approximates the continuous CDF . As increases and bin widths decrease, becomes a finer staircase approximation to , especially when bin indices are mapped to representative values (e.g., bin midpoints).  For independent samples from , the counts in each bin, , follow a multinomial distribution: , with . The empirical proportions estimate , and the empirical CDF approximates .   Example: Consider a standard normal random variable , with PDF and CDF , computed via the standard normal CDF .  Discretize into bins with boundaries , , , , , . The bins are: , , , , .  The bin probabilities, using , are:                   The discrete random variable (bin index) has PMF , and its CDF is:   To visualize, plot the continuous CDF against , and overlay the step-function , mapping each bin to a representative value (e.g., the midpoint of finite bins or an approximate value for unbounded bins).   A continuous CDF and it's discretization by the binning method.   A continuous CDF and it's discretization by the binning method.    The following Python code generates this plot using scipy.stats.norm :  import numpy as np import matplotlib.pyplot as plt from scipy.stats import norm # Boundaries for 5 bins boundaries = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf] # Compute p_i using standard normal CDF p = np.diff(norm.cdf(boundaries)) # p[0] = Phi(-1.5) - Phi(-inf), etc. # Discrete CDF G(y): cumulative sum of p_i g = np.cumsum(p) # g[0] = p1, g[1] = p1+p2, ..., g[4] = 1 # For plotting G as step function, use boundaries (replace -inf, inf with finite values) plot_boundaries = [-2.5, -2, -1, 0, 1.0, 2.0] # Finite proxies for -inf, inf g_steps = np.insert(g, 0, 0) # Start at 0 before first boundary # Continuous CDF: plot over x range x = np.linspace(-4, 4, 500) f_x = norm.cdf(x) # Plot continuous and discretized CDFs plt.figure(figsize=(8, 6)) plt.plot(x, f_x, label='Continuous CDF F(x) = Φ(x)', color='blue') plt.step(plot_boundaries, g_steps, where='post', label='Discrete CDF G(y)', color='red', linestyle='--') plt.title('Continuous vs. Discretized CDF of N(0, 1)') plt.xlabel('x') plt.ylabel('Cumulative Probability') plt.legend() plt.grid(True) plt.show()  This code produces a plot comparing the smooth continuous CDF with the step-like discrete approximation. As the number of bins increases, the steps in become smaller, improving the approximation to when is scaled to bin midpoints.    Moment Generating Functions     "
},
{
  "id": "fig-continuous-discretized-cdf",
  "level": "2",
  "url": "sec-Stats-Miscellaneous-Topics.html#fig-continuous-discretized-cdf",
  "type": "Figure",
  "number": "1.13.1",
  "title": "",
  "body": " A continuous CDF and it's discretization by the binning method.   A continuous CDF and it's discretization by the binning method.   "
},
{
  "id": "sec-Additional-Mathematical-Techniques",
  "level": "1",
  "url": "sec-Additional-Mathematical-Techniques.html",
  "type": "Section",
  "number": "1.14",
  "title": "Additional Mathematical Techniques",
  "body": " Additional Mathematical Techniques  I will be adding all the other mathematical techniques we will need when we write other chapters. I might even just list the topic here and present the detail at the point of use.  "
},
{
  "id": "sec-Statistical-Decision-Theory",
  "level": "1",
  "url": "sec-Statistical-Decision-Theory.html",
  "type": "Section",
  "number": "2.1",
  "title": "Statistical Decision Theory",
  "body": " Statistical Decision Theory   Statistics is the application of probability theory to real-world problems. Basically, two types of problems are handled by using statitics: (1) the estiamtion of parameters of a statistical model, and (2) predictions of future outcomes assuming the underlying probabilistic model is fully known. Both of these are necessary to make an optimized decision in an uncertain environment.  Optimization is done by minimizing a loss function, or, alternately, maximizing a reward function. The choice of loss function is critical since that provides the constraint under which decisions will be optimal. We will see that the loss function for discrete random variables and continuous random variables will be different.  As an illustration of the optimization process let us consider two scalar random variables and . Suppose, we have made independent and identical observations of pairs: We will use superscipts to indicate data instances, i.e., data points, and reserve the subscript indices for components of vector, matrix or tensor-like variables. Suppose the experiment from which the data was collected can be modeled by some statistical model, given by a joint distribution function or the corresponding joint probability density (or PMF in case of discrete variables) with parameters .  We want to predict that goes with the new assuming the relation between and are statistical. That is, we seek a function which will give the best  from . We will denote this optimal function by placing a hat on the symbol as in . All we then we need to predict that goes with is   What is the meaning of best function here? We say that there is a loss function that will be minimized if we chose the right  . You would think that the optimal function would be obtained by This equation is minimizing a function of random variables and . What we should be doing is to minimize its expectation value. Since we are in a joint distribution of , this optimization process should be   It helps to separate out the and parts if we write the PDF of a joint distribution in terns of a conditional and a marginal distriution. Using this and integrating over inside the integral Since is , the minimization of the quantities just inside the square brackets is enough to find the optimal . Hence, you don't need the full joint probability density, just the conditional density is enough. The supervised learning algorithms of Machine Learning are all based on this equation. Various algorithms try to learn the function from the data and use them to make predictions on new datapoints. In this section, I will illustrate the application of this formula to a regression task . In the next section, we will discuss a binary classification task .     Regression  A common loss function for regression tasks for predicting continuous variable is the squared error . For arbitrary point in the random variable space of , the error will be Therefore, the equation for optimal becomes Setting the functional derivative of the integral on the right side to zero shoudl give us the extremum function, say . This will give us (Think of a regular partial derivative with as an independent variable.) That is the extremum of the squared loss occurs at \\tilde{f}. Is it mimimum? To show that it is a minimum, try finding the second derivative and use the expectation symbol to hide away the integration. That means the extremum value is actualy a minimum. Thus, the optimal funtion that minimizes the squared error is the one that maps to the mean value of in the conditional probability . Hence, in terms of the random variables, the optimal function is This is a theretical result of the prediction function. To apply this result to obtain that gies with , further assumptions are made. For instance, as we will see below in k-NN algorithm, each datapoint in the dataset is given probaility , where is the number of nearest neighbors of in the training dataset.  In some approached, we invoke a functional form of , e.g., a linear function of , then you can use the points in the dataset to estimate parameters of that function which you can then use to make prediction for the new point .    The kNN Method  A straightforward way of implementing the result in Eq. for making prediction of for a new point with is to average the values of points in the dataset closest to their -distances.  The simplest is, of course, to just look at the nearest neightbor , say distance to from is smallest, assuming unique for simplicity, then, you would say that that is same as .  If you have degenracy, say, distances to and from are smallest and equal. Then, for , we would just averate the 's. We might want to extend the averaging to more than the nearest average. That's where ( Nearest Neighbors) algorithm comes in. You go to the location and expand out and count upto closest neighbors.. Say, indices for them are : . Then, to predict you would average their 's, which is essentially approximating and truning the integral into a sum.   For scalar functions , the k-NN method works quite well if you have sufficiently dense coverage of the by the data, but for vector variables, such as with several components, say -coponents, the number of neighbors grows exponetially with , and the method become impractical. You can prove a theoretical result that shows that at some point it will become impossible since almost all datapoints will be included in any search and hence you will be predicting same for every . This result goes by the ominous phrase the curse of dimensionality .    The Simple Linear Regresson  For a scalar variable, a linear dependency of on will be where is the and the slope. The task is to find the optimal values of and that minimizes the squared error, where we now include a factor for cancling out a factor that results when you take derivatives. Assuming the the dataset is representative of the entire , we find approximation of the by setting and repacing the integration by the sum. Let's call this expectation value of by  Now, to find the optimal function , here the optimal parameters and , we just have to take partials with each and set to zero. These result in two linear equations in and which are easily solved. We denote the solutions by and . You will find that the following sums help in doing the algebra. After algebra that you should do out, you will find the following expressions for the optimal parameters.     shows a simple linear regression on a dataset consisting of datapoints generated by adding noise to a linear deterministic function by where Here, we have chosen known values of and to generate the data and want to know how well the simple linear regression produces estimates of these parameters.  The simple linear regression using datapoints gave the predicting funtion to be with Compare them to the true values: From the data, we appear to get a better estimate for the slope than that of the intercept.  Note that since the dataset, coming from a random sampling process, is a random variable. Therefore, the estimates and deduced from using are also random variables with their own distributions.   The estimates we gave above are just estimates of their mean values. And, of course, you can do an entire analysis of the significance level of these estimates. While important in statistics books, they will be largely ignored in the Machine Learning work. We take the view that the estimates of the parameters give us the best we can do.   Illustration of a simple linear regression.   The image shows that linear fit to a scatter data.    The figure was generated by the following Python code.  import numpy as np import matplotlib.pyplot as plt # Sample dataset np.random.seed(42) # for reproducibility X = np.linspace(0, 10, 100) Y = 1.0+ 2.0 * X + np.random.normal(0, 1, 100) # Y = 2X + 1 + noise # Calculate means x_bar = np.mean(X) y_bar = np.mean(Y) # Calculate beta_1 (slope) numerator = np.sum((X - x_bar) * (Y - y_bar)) denominator = np.sum((X - x_bar) ** 2) beta_1 = numerator \/ denominator # Calculate beta_0 (intercept) beta_0 = y_bar - beta_1 * x_bar # Print results print(f\"Estimated beta_0 (intercept): {beta_0:.2f}\") print(f\"Estimated beta_1 (slope): {beta_1:.2f}\") # Plot the data and regression line plt.scatter(X, Y, facecolors='none', edgecolors='blue', color='blue', label='Data points') plt.plot(X, beta_0 + beta_1 * X, color='red', label='Regression line') plt.xlabel('X') plt.ylabel('Y') plt.title('Simple Linear Regression Example') plt.legend() plt.grid(True) plt.show()    "
},
{
  "id": "sec-Statistical-Decision-Theory-2-6",
  "level": "2",
  "url": "sec-Statistical-Decision-Theory.html#sec-Statistical-Decision-Theory-2-6",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "supervised learning "
},
{
  "id": "subsec-Squared-Error-Loss-Function-2",
  "level": "2",
  "url": "sec-Statistical-Decision-Theory.html#subsec-Squared-Error-Loss-Function-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "squared error "
},
{
  "id": "subsec-KNN-method-5",
  "level": "2",
  "url": "sec-Statistical-Decision-Theory.html#subsec-KNN-method-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "curse of dimensionality "
},
{
  "id": "subsec-The-Simple-Linear-Regresson-3",
  "level": "2",
  "url": "sec-Statistical-Decision-Theory.html#subsec-The-Simple-Linear-Regresson-3",
  "type": "Example",
  "number": "2.1.1",
  "title": "",
  "body": "  shows a simple linear regression on a dataset consisting of datapoints generated by adding noise to a linear deterministic function by where Here, we have chosen known values of and to generate the data and want to know how well the simple linear regression produces estimates of these parameters.  The simple linear regression using datapoints gave the predicting funtion to be with Compare them to the true values: From the data, we appear to get a better estimate for the slope than that of the intercept.  Note that since the dataset, coming from a random sampling process, is a random variable. Therefore, the estimates and deduced from using are also random variables with their own distributions.   The estimates we gave above are just estimates of their mean values. And, of course, you can do an entire analysis of the significance level of these estimates. While important in statistics books, they will be largely ignored in the Machine Learning work. We take the view that the estimates of the parameters give us the best we can do.   Illustration of a simple linear regression.   The image shows that linear fit to a scatter data.    The figure was generated by the following Python code.  import numpy as np import matplotlib.pyplot as plt # Sample dataset np.random.seed(42) # for reproducibility X = np.linspace(0, 10, 100) Y = 1.0+ 2.0 * X + np.random.normal(0, 1, 100) # Y = 2X + 1 + noise # Calculate means x_bar = np.mean(X) y_bar = np.mean(Y) # Calculate beta_1 (slope) numerator = np.sum((X - x_bar) * (Y - y_bar)) denominator = np.sum((X - x_bar) ** 2) beta_1 = numerator \/ denominator # Calculate beta_0 (intercept) beta_0 = y_bar - beta_1 * x_bar # Print results print(f\"Estimated beta_0 (intercept): {beta_0:.2f}\") print(f\"Estimated beta_1 (slope): {beta_1:.2f}\") # Plot the data and regression line plt.scatter(X, Y, facecolors='none', edgecolors='blue', color='blue', label='Data points') plt.plot(X, beta_0 + beta_1 * X, color='red', label='Regression line') plt.xlabel('X') plt.ylabel('Y') plt.title('Simple Linear Regression Example') plt.legend() plt.grid(True) plt.show()  "
},
{
  "id": "sec-Binary-Classification",
  "level": "1",
  "url": "sec-Binary-Classification.html",
  "type": "Section",
  "number": "2.2",
  "title": "Binary Classification",
  "body": " Binary Classification       The Classification Variable  In addition to the regression task discussed in the previous section, we are also interested in classification tasks . In a classification task, is a discrete variable with values are the classes or categories, and the task of the algorithm is to predict a class to which the feature variable goes with. In the case of -classes, we treat the variable as a collection of discrete variables, each taking either or showing the outcome being that class observed or not. This is the one-hot reprsentation that we have discussed before. We usually, think of te collection as a column vector, which is sometimes typed as a transpose of a row vector. In any observation of the variable , you find only one of these variables to have a and the rest zeros, e.g., if your first observation of , i.e., is a class , then, your is represented by the following. In the theory of supervised learning given in the last section for continuous , we now replace the conditional PDF of by the corresponding Probability Mass Function (PMF) and the integration by a summation. We will also use for the optimal function in place of to empahsize that it will be also a discrete variables of the same type as . Thus, Eq. is replaced by In a binary classification , can take one of the two values: yes\/no, true\/false, up\/down, cat\/dog, 0\/1, etc. Hence, a binary classification will be a collection of two random variables. In any observation, you will observe one of the two states: Although, it is fine to think of observations in this language of vectors. However, for a two state, there is a much simpler way of thinking of the variable - just use the value of class , i.e., for : This makes algebra simpler since, now is just a scaler with two values, and .    Binary Classification with Scalar  In this setup, we have a scalar variable (a single real-valued feature) and a binary response variable that takes two values, typically , representing two classes. The goal is binary classification: given a value of , predict the class label .   Predicting Probability   Since, predicted is either zero or , it is natural to predict probability that for a given X. That is the function that we seek will be probability of event when has a particular value, . Note that if we know the conditional probability mass function, then this function has a definite value for each value of . We can write it more generally without specifying as     The Loss Function   Since we trying to predict conditional probability of given an , if actual value was , then from a loss function we expect that it be small when predicted probability is closer to than if it closer to zero. Negative log of probability captures this aspect well as shown in . Thus, we expect the loss function to have the following form.   We can combine these two expressions into one.    Negative log likelihood loss functions for the two classes of a binary classification task.   The image shows negative log likelihood functions of twobinary classes. The functions give zero loss for the correct class and maximum loss for the other class.    The optimal value of parameters will be obtained when we minimize it's expectation value using the dataset. Recall that the data consists of pairs for , where the are points on the real line since we have a scalar , and indicates the class, which is either or . Weighing each datapoint equally by a factt gives us the following expectation, to be denoted by , where The expected loss in Eq. is called Binary Cross Entropy . As before, we get the optimum value of the parameter(s) by solving the following extremum finding equation. one for each . We have already shown how it is dome in the case of linear regression. Now, in the following we will discuss a linear model for classification that is called the logistic regression.   The Logistic Regression Model   Now, we will specify statistical model for that is used in a common algoritm called Logistic Regression . Logistic regression here will try to find the best separation point on the axis that will minimize the net error on the dataset even though the data for some values of are not so clear since there are datapoints that correspond to both classes as shown in .   Illustration of a binary classification overlap in for the two classes.   Illustration of a binary classification overlap in X for the two classes.    If consisted of two variables, say , then it will find a straight line in the plane that is the decision line that minimizes the error. For higher dimensional situations, logistic regression predicts the decision planes and hyperplanes that minizations of error. These decision point\/planes are not perfect if no linear boundary can separate the points in the dataset. That is, we will never get a zero error on in these cases.  In logistic regression we models the probability by a sigmoid function of a linear combination of a constant and a linear function of : where sigmoid function, writing ( ) The range for is from to . A plot of sigmoid versus the original variable for different choices of the scale (or slope) parameter is shown in . You can see that sigmoid function's output is restricted to the range suporting it's use as a probability.  An usefu fact about sigmoid is about its derivative, which can be written in terms of sigmoid itself as you can verify by explictily doing it.    Sigmoid function.   Sigmoid function rises from zero to 1 when z goes from minus infinity to infinity.    The decision boundary depends on the choice of the threshold. For instance, if we choose , then we will classify into two classes as follows. This translates into the following decision for the value: Thereofore, the decision point on the -axis for a scalar using the optimal values of the parameters will be That is, we predict if and otherwise.   Estimating Optimal Parameters From The Data   Eq. has boiled the task of classification of a new data to comparing it to the optimal . The optimal and are the values of these parameters that minimize the loss function in Eq. . Writing the loss function using the logistic regression model's probability function . With for , will have To find it's extremum point in the plane of parameters , in principle, we just need to solve the following for and . For simpler notation, let's write for . With Unlike what happened in the case of linear regression, it's not possible to solve these equations analytically. Among several numerical algorithms, gradient descent method is the most commonly used method.     Gradient Descent for Logistic Regression  To minimize the cross-entropy loss defined in , we use the gradient descent algorithm, as the partial derivatives and cannot be solved analytically. Gradient descent iteratively updates the parameters and to reduce by moving in the direction of the negative gradient. The amount of update to apply in each iteration is controlled by the learning rate parameter .  The update rule for each parameter is: where is the learning rate, a positive scalar controlling the step size. Using the sigmoid function’s derivative, , where , and noting that and , the gradients are: where .  Starting from an initial guess for , gradient descent iteratively applies these updates until converges to a minimum or a maximum number of iterations is reached. The process can be visualized in the plane, where the loss forms a surface, and each update moves the parameters along the negative gradient toward the minimum. This is illustrated in .   Gradient descent in the plane, showing the loss surface for with arrows indicating parameter updates toward the minimum of the loss function.   Gradient descent showing arrows as updates move the point in the parameter space towards the point where minimum of loss occurs.    Following program shows the python code for the simulation that find the optimal values of and by gradient descent method. To produce larger arrow for the figure I used a large learning rate , which is named alpha in the program. For a fast and yet reliable convergence, learning rate has to be chosen carefully. Often, one needs to do experimentation to find an effective learning rate for the problem at han.  import numpy as np import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D # Generate synthetic data np.random.seed(42) n = 50 X = np.concatenate([np.random.normal(0, 1, n), np.random.normal(2, 1, n)]) Y = np.concatenate([np.zeros(n), np.ones(n)]) # Sigmoid function def sigmoid(z): return 1 \/ (1 + np.exp(-z)) # Compute loss def compute_loss(beta0, beta1, X, Y): z = beta0 + beta1 * X sigma = sigmoid(z) loss = -(1\/len(X)) * np.sum(Y * np.log(sigma + 1e-10) + (1 - Y) * np.log(1 - sigma + 1e-10))#small values to prevent log of zero. return loss # Compute gradients def compute_gradients(beta0, beta1, X, Y): z = beta0 + beta1 * X sigma = sigmoid(z) grad_beta0 = (1\/len(X)) * np.sum(sigma - Y) grad_beta1 = (1\/len(X)) * np.sum((sigma - Y) * X) return grad_beta0, grad_beta1 # Gradient descent alpha = 1.0 num_iterations = 10 beta0, beta1 = 2.0, -3.0 # Initial guess path = [(beta0, beta1)] for _ in range(num_iterations): grad_beta0, grad_beta1 = compute_gradients(beta0, beta1, X, Y) beta0 -= alpha * grad_beta0 beta1 -= alpha * grad_beta1 path.append((beta0, beta1)) # Create mesh for loss surface beta0_range = np.linspace(-3, 3, 50) beta1_range = np.linspace(-3, 3, 50) B0, B1 = np.meshgrid(beta0_range, beta1_range) Loss = np.array([[compute_loss(b0, b1, X, Y) for b0 in beta0_range]\\\\ for b1 in beta1_range]) # Plot fig = plt.figure(figsize=(10, 8)) ax = fig.add_subplot(111, projection='3d') ax.plot_surface(B0, B1, Loss, cmap='viridis', alpha=0.3) path = np.array(path) loss_path = [compute_loss(b0, b1, X, Y) for b0, b1 in path] ax.plot(path[:, 0], path[:, 1], loss_path, 'r.-', label='Gradient Descent Path', markersize=10) for i in range(len(path)-1): ax.quiver(path[i, 0], path[i, 1], loss_path[i], path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1], loss_path[i+1] - loss_path[i], color='red', arrow_length_ratio=0.1) # Add projection onto the (beta_0, beta_1) plane z_min = np.min(Loss) - 0.05 # Slightly below min for visibility ax.plot(path[:, 0], path[:, 1], z_min * np.ones_like(path[:, 0]), 'b--', label='Projected Path') for i in range(len(path)-1): ax.quiver(path[i, 0], path[i, 1], z_min, path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1], 0, color='blue', arrow_length_ratio=0.5) # Add vertical lines from path to projection for i in range(len(path)): ax.plot([path[i, 0], path[i, 0]], [path[i, 1], path[i, 1]], [loss_path[i], z_min], 'k--', alpha=0.5) ax.set_xlabel(r'$\\beta_0$') ax.set_ylabel(r'$\\beta_1$') ax.set_zlabel(r'$\\mathcal{L}_D$') ax.set_title('Gradient Descent on Loss Surface with Projected Path') ax.legend() plt.tight_layout() plt.savefig('gradient_descent_projected.png', dpi=300) plt.show()  A logistic regression solution of a binary classificaiton problem is shown in . The top plot shows how the parameters and the resulting decision boundary in the -space changes as we train the parameters. With sufficcient iterations of the algorithm, the values stabilize, corresponding to the minimum of the error discovered by the gradient descenct method. Since the data for the two classes have considerable overlap, no value of can separate the two classes.   The evolution of the parmeters and the decision boundary with iterations of the logistic regression algorithm.   The evolution of the parmeters and the decision boundary with iterations of the logistic regression algorithm. The parameters change initially, but they stabilize to their asymptotic values.    import numpy as np import matplotlib.pyplot as plt import matplotlib.gridspec as gridspec # Generate the data np.random.seed(42) n = 50 X0 = np.random.normal(0, 1, n) X1 = np.random.normal(2, 1, n) X = np.concatenate([X0, X1]) Y = np.concatenate([np.zeros(n), np.ones(n)]) # Define the sigmoid function def sigmoid(z): return 1 \/ (1 + np.exp(-z)) # Compute gradients def compute_gradients(beta0, beta1, X, Y): z = beta0 + beta1 * X sigma = sigmoid(z) grad_beta0 = (1 \/ len(X)) * np.sum(sigma - Y) grad_beta1 = (1 \/ len(X)) * np.sum((sigma - Y) * X) return grad_beta0, grad_beta1 # Gradient descent parameters alpha = 1.0 # Learning rate num_iterations = 200 # Number of iterations beta0 = 0.0 # Initial beta_0 beta1 = 0.0 # Initial beta_1 # Lists to store decision boundaries and parameters boundaries = [] beta0_vals = [] beta1_vals = [] iterations_list = [] # Run gradient descent for iteration in range(1, num_iterations + 1): grad_beta0, grad_beta1 = compute_gradients(beta0, beta1, X, Y) beta0 -= alpha * grad_beta0 beta1 -= alpha * grad_beta1 # Compute decision boundary if beta1 is not zero if abs(beta1) > 1e-10: boundary = -beta0 \/ beta1 beta0_vals.append(beta0) beta1_vals.append(beta1) boundaries.append(boundary) iterations_list.append(iteration) # Print final values print(f\"Final beta_0: {beta0:.2f}\") print(f\"Final beta_1: {beta1:.2f}\") print(f\"Final decision boundary: {boundaries[-1]:.2f}\") # Create figure with GridSpec for different subplot sizes fig = plt.figure(figsize=(10, 12)) # Overall figure size gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1]) # Top plot taller, bottom plot shorter # Top subplot: Evolution of decision boundary and parameters ax1 = fig.add_subplot(gs[0]) ax1.plot(iterations_list, boundaries, marker='o', markersize=2, linestyle='-', color='blue', label=r'$x_{\\text{decision}}$') ax1.plot(iterations_list, beta0_vals, marker='x', markersize=2, linestyle='-', color='red', label=r'$\\beta_0$') ax1.plot(iterations_list, beta1_vals, marker='s', markersize=2, linestyle='-', color='green', label=r'$\\beta_1$') ax1.set_xlabel('Iteration') ax1.set_ylabel(r'Decision Boundary $x$, $\\beta_0$, $\\beta_1$') ax1.set_title('Evolution of Decision Boundary and Parameters During Gradient Descent') ax1.grid(True) ax1.legend() # Bottom subplot: 1D data with decision boundary ax2 = fig.add_subplot(gs[1]) ax2.scatter(X[Y==0], np.full(n, 0.1), marker='o', facecolors='none', edgecolors='blue', label='Class 0') ax2.scatter(X[Y==1], np.full(n, -0.1), marker='x', color='green', label='Class 1') ax2.axvline(boundaries[-1], color='red', linestyle='--', label='Boundary') ax2.set_xlabel('X') ax2.set_ylim(-0.3, 0.3) ax2.set_yticks([-0.1, 0.1]) ax2.set_yticklabels(['Class 1', 'Class 0']) ax2.set_title('1D Binary Classification Example with Overlap (Displaced Points)') ax2.grid(True) ax2.legend() # Adjust layout to prevent overlap plt.tight_layout() plt.show()   "
},
{
  "id": "subsecThe-Classification-Variable-2",
  "level": "2",
  "url": "sec-Binary-Classification.html#subsecThe-Classification-Variable-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "binary classification "
},
{
  "id": "fig-binary_cross_entropy_loss",
  "level": "2",
  "url": "sec-Binary-Classification.html#fig-binary_cross_entropy_loss",
  "type": "Figure",
  "number": "2.2.1",
  "title": "",
  "body": " Negative log likelihood loss functions for the two classes of a binary classification task.   The image shows negative log likelihood functions of twobinary classes. The functions give zero loss for the correct class and maximum loss for the other class.   "
},
{
  "id": "subsec-binary-classification-scalar-feature-9",
  "level": "2",
  "url": "sec-Binary-Classification.html#subsec-binary-classification-scalar-feature-9",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Binary Cross Entropy "
},
{
  "id": "subsec-binary-classification-scalar-feature-11",
  "level": "2",
  "url": "sec-Binary-Classification.html#subsec-binary-classification-scalar-feature-11",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Logistic Regression "
},
{
  "id": "fig-binary_classification_overlap",
  "level": "2",
  "url": "sec-Binary-Classification.html#fig-binary_classification_overlap",
  "type": "Figure",
  "number": "2.2.2",
  "title": "",
  "body": " Illustration of a binary classification overlap in for the two classes.   Illustration of a binary classification overlap in X for the two classes.   "
},
{
  "id": "fig-sigmoid_function",
  "level": "2",
  "url": "sec-Binary-Classification.html#fig-sigmoid_function",
  "type": "Figure",
  "number": "2.2.3",
  "title": "",
  "body": " Sigmoid function.   Sigmoid function rises from zero to 1 when z goes from minus infinity to infinity.   "
},
{
  "id": "subsec-binary-classification-scalar-feature-17",
  "level": "2",
  "url": "sec-Binary-Classification.html#subsec-binary-classification-scalar-feature-17",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "decision boundary "
},
{
  "id": "subsec-gradient-descent-2",
  "level": "2",
  "url": "sec-Binary-Classification.html#subsec-gradient-descent-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "learning rate "
},
{
  "id": "fig-gradient_descent_projected",
  "level": "2",
  "url": "sec-Binary-Classification.html#fig-gradient_descent_projected",
  "type": "Figure",
  "number": "2.2.4",
  "title": "",
  "body": " Gradient descent in the plane, showing the loss surface for with arrows indicating parameter updates toward the minimum of the loss function.   Gradient descent showing arrows as updates move the point in the parameter space towards the point where minimum of loss occurs.   "
},
{
  "id": "subsec-gradient-descent-6",
  "level": "2",
  "url": "sec-Binary-Classification.html#subsec-gradient-descent-6",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "learning rate "
},
{
  "id": "fig-gradient_descent_with_overlap",
  "level": "2",
  "url": "sec-Binary-Classification.html#fig-gradient_descent_with_overlap",
  "type": "Figure",
  "number": "2.2.5",
  "title": "",
  "body": " The evolution of the parmeters and the decision boundary with iterations of the logistic regression algorithm.   The evolution of the parmeters and the decision boundary with iterations of the logistic regression algorithm. The parameters change initially, but they stabilize to their asymptotic values.   "
},
{
  "id": "sec-Multi-Feature-Problems",
  "level": "1",
  "url": "sec-Multi-Feature-Problems.html",
  "type": "Section",
  "number": "2.3",
  "title": "Multi-Feature Problems",
  "body": " Multi-Feature Problems   In supervised learning tasks, our objective is to predict the value of a target variable , given the feature variable . Both and can be a multivariate variable - for instance, you may want to predict price of a house based on it's size and neighborhood . In this section, while we will keep the target variable a simple scalar, e.g., a rel number or binary or -categories, we will look at problems that have multiple independent feature variables in each instance of the data.  Let consists of variables, Usually, a vector notation is used to organize the variables in a column vector or it's transposed as a row vector. The datasets for such systems can be organized into table, e.g., a Excel sheet or a csv file. They are called structured data .  A dataset will consist of observations on and will look the same superficially, but each point will have more details. with each , now having more details.   Besides the structured data, we also have unstructured data. For instance, when there is an important order information within the components of the variable, representing them into independent e.g., if is a pixels photograph, then you could represent this as a long -element vector, but that would lose the information about the neighboring pixels. A better representation would be to represent this by a matrix, or a 2-dimensional tensor, with components, now labelled with two indices. Often, books use a bold face letter to represent such variables. We might also do that if we want to emphasize the multidimensionality aspect. Non-table data like these are referred to as unsrtructured data . This is a catch all phrase and includes any data that is not organized as a table\/sheets.    Multi-Feature Linear Regression   In a linear regression, we model the predictor function by a constant plus a linear function of . When was a scaler we had Now, we have that has components. That means, we will now have one term for each component. That will result in the following empirical squared loss from the data. Optimized parameters will minimize this . The derivative with respect has a different form than the other derivatives. Dropping a common factor ., the results are: where the indices . These equations are linear in the parameters and can be solved analytically. A matrix language provides a more compact way to see the solution.  Although, the matrix-form solution itself is not very illuminating, the language used to build the solution will be used throughout in more afvanced algorithms. Therefore, we take time to present it here.  Notice, we have parameters, . We organize them in a parameter vector . We modify -dimensional into an -dimensional vector whose first element is and rest of them same as those of . Then, is just a dot product of and . The next level of compacting the notation is to represent the part of the dataset by rows in a matrix that now includes a at the first column location. Let's denote this matrix . It is called the design matrix and isusually denoted without the subscript , but we want it there to remind us that it is a data-dependent quantity. The -part of the dataset is now a -length vector. Let's represent it by . Using matrix multiplications, the empirical squared loss function now takes this really compact form. Setting the derivative of this loss function with vector gives the following equation for the optimal parameters . For calculations of this step see below. Therefore, if the inverse of square matrix exists, then, Using the estimated can be used in to predict for a new by where    Taking Derivatives With Respect To A Vector  In Eq. I gve you the result of taking derivative of the loss function with respect to vector . This actually is a compact way of writing the derivatives with respect to each component of the vector . What does mean? We then work out expression for a generic row, say the row by writing out explicitly. For simplification in notation, let us drop the subscript from our symbols, except for . We are also going to pull on the since is constant. where is Kronecker delta, which is when and otherwise. Now, continuiung with the calculations, Setting this to zero will yield equations in unknown 's. Neglecting the common factor and writing the non- terms on the right side, we can collect all equations in a matrix form. The left side is and the right side is .      Multi-Feature Logistic Regression   In multivariate logistic regression, we model the probability that a binary outcome equals 1 given a vector of features . For this treatment, we focus on the two-dimensional case where , so . The model assumes a linear relationship in the logit space:     where is the linear predictor, and is the sigmoid (logistic) function that maps to a probability between 0 and 1. The parameters (intercept), , and (coefficients for and ) are estimated from data.  To train the model, meaning finding the optimal values of the parameters , , and , we minimize the data-based empirical binary cross-entropy loss (also known as the negative log-likelihood, up to a constant factor) as we did for scalar when we discussed the logistic regression for binary classification for a scalar :     where is the number of observations, is the observed label for the -th data point, , and loss for the datapoint.  To find optimized parameters, we need the derivative of the loss function with respect to the parameters. A detailed calculation is shown below.   In matrix form, if is the design matrix (with a column of 1s for the intercept), the gradient vector is , where and are -dimensional vectors with each component referring to the data instants.  Optimization by gradient descent method used these derivatives in each iteration through the dataset. The initial values of the parameters are set randomly or some other procedure and then evolved iteratively. Let us denote the values of the parameters at iteration by a superscript index , e.g., , then after the next iteration, the values will be updated via gradient descent with the following rule. where is the learning rate . The updates are repeated until convergence (e.g., when the empirical loss stabilizes).  After you find the optimal parameters, you can use them to deduce the boundary decision that you use to predict the class to which a new would belong. In the case of a two-dimensional , the decision boundary will be the following straight line.  shows an example decision boudary.   Decision boundary in a binary classification problem that sseparates -space into the regions of two classes.   The plot shows a line that is produced by the logistic regression algorithm that spearates the (x1, x2)-space into the regions of two classes.      Derivatives of the Loss Function  The loss is convex in , so we can use gradient descent to find the minimum. The partial derivatives (gradient components) are derived as follows:  Start with the chain rule on a single term in the ampirical loss function: Dropping the indices from the symbols for simplicity, we have The derivative on the right side is The last derivative is readily done since . For the other derivative Putting these altogether, we have Therefore, the derivative of the empirical loss function will be   In matrix form, if is the design matrix (with a column of 1s for the intercept), the gradient vector is , where and are -dimensional vectors with each component referring to the data instants.    A Python Program for the Logistic Regression  The following program implements logistic regression for a two-variate and using gradient descent algorithm. In the future, we will learn to use python packages that do a lot of the calculations behind the scene.  The program below plots the value of the loss function at each iteration shown in . You can see that the loss stadily goes to zero for the dataset used.   Loss function versus iteration.   The plot shows loss decreasign with iteration.    import numpy as np import matplotlib.pyplot as plt # Generate data (reproducible) np.random.seed(42) n = 50 X0 = np.random.normal(-2, 1, (n, 2)) Y0 = np.zeros(n) X1 = np.random.normal(2, 1, (n, 2)) Y1 = np.ones(n) X = np.vstack((X0, X1)) Y = np.hstack((Y0, Y1)) indices = np.arange(2 * n) np.random.shuffle(indices) X = X[indices] Y = Y[indices].reshape(-1, 1) # Add intercept column X_with_intercept = np.hstack((np.ones((2 * n, 1)), X)) # Sigmoid function def sigmoid(z): return 1 \/ (1 + np.exp(-z)) # Loss function def loss(X_with_intercept, Y, beta): z = X_with_intercept @ beta p = sigmoid(z) return -np.mean(Y * np.log(p + 1e-10) + (1 - Y) * np.log(1 - p + 1e-10)) # Gradient descent with loss tracking beta = np.zeros((3, 1)) learning_rate = 0.1 iterations = 1000 n_samples = X_with_intercept.shape[0] losses = [] for i in range(iterations): z = X_with_intercept @ beta p = sigmoid(z) gradient = (1 \/ n_samples) * (X_with_intercept.T @ (p - Y)) beta -= learning_rate * gradient current_loss = loss(X_with_intercept, Y, beta) losses.append(current_loss) # Plot loss vs. iteration plt.figure(figsize=(8, 6)) plt.plot(range(iterations), losses, 'b-') plt.xlabel('Iteration') plt.ylabel('Loss') plt.title('Loss vs. Iteration in Gradient Descent') plt.grid(True) plt.show()    "
},
{
  "id": "sec-Multi-Feature-Problems-2-2",
  "level": "2",
  "url": "sec-Multi-Feature-Problems.html#sec-Multi-Feature-Problems-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "structured data "
},
{
  "id": "sec-Multi-Feature-Problems-2-4",
  "level": "2",
  "url": "sec-Multi-Feature-Problems.html#sec-Multi-Feature-Problems-2-4",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "unsrtructured data "
},
{
  "id": "subsec-Multi-Feature-Linear-Regression-2-3",
  "level": "2",
  "url": "sec-Multi-Feature-Problems.html#subsec-Multi-Feature-Linear-Regression-2-3",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "design matrix "
},
{
  "id": "fig-logistic_regression_multivariate",
  "level": "2",
  "url": "sec-Multi-Feature-Problems.html#fig-logistic_regression_multivariate",
  "type": "Figure",
  "number": "2.3.1",
  "title": "",
  "body": " Decision boundary in a binary classification problem that sseparates -space into the regions of two classes.   The plot shows a line that is produced by the logistic regression algorithm that spearates the (x1, x2)-space into the regions of two classes.   "
},
{
  "id": "fig-loss_vs_iteration_logistic_regression_mv",
  "level": "2",
  "url": "sec-Multi-Feature-Problems.html#fig-loss_vs_iteration_logistic_regression_mv",
  "type": "Figure",
  "number": "2.3.2",
  "title": "",
  "body": " Loss function versus iteration.   The plot shows loss decreasign with iteration.   "
},
{
  "id": "sec-MultiClass-Classification",
  "level": "1",
  "url": "sec-MultiClass-Classification.html",
  "type": "Section",
  "number": "2.4",
  "title": "Multi-Class Classification",
  "body": " Multi-Class Classification   In binary classification, the target could be one of two categories — 0\/1, yes\/no, true\/false, up\/down, etc. In multiclass classification tasks, we are asked to predict one most likely class from three or more discrete target classes. This is to be distinguished from a multilabel prediction task , where you would predict two or more most likely classes from several posible classes.  The extension from the binary classification task is straightforward, but at times can be confusing. We work out the details for a -class problem.  First, we will discuss the loss function for the -class problem. Just like the binary classification problem, we model the conditional probabilities directly. We will see that a softmax function models the probabilities of the classes given such that their sum properly equals . Once, we have predictions of the probabilities , we predict the class by simply finding the class that has the highest among them.     Loss Function of Multiclass Classification  First, let us recall the binary classification for an -dimensional feature vector . There is one variable which is encoded as , with denoting class and denoting class . The conditional probabilities of the two classes given are We model the conditional probability . In the logistic regression algorithm, we further assume a mathematical form of this conditional probability: where the sigmoid function is The quantity is called the score . The loss function for optimizing the parameters ’s is written as When , you only have the first term with log of the probability corresponding to class , and when , you only have the second term, whose argument to the log is the probability for class . Let us introduce an indicator\/kronecker delta variable to write this expression more succinctly. Then, the loss function is a sum over the two classes. For instance, if you have a datapoint , i.e., this datapoint belongs to class , then the value of this loss function will be The binary loss formula in Eq. can be clearly generalized to a -class classification problem, by just summing over classes, provided the probabilities add up to . So, how should we model ? The sigmoid representation of for the binary case gives a clue if we rewrite the sigmoid a little differently: That means Let's denote the denominator by the letter and then which satisfies the probability sum requirement automatically. This suggests that we should write similar formulas for a -class task. Since we have more than two classes, we will write scores for each with an index for the class: with This representation of the probabilities is called the softmax function over the classes. Explicitly, Using the softmax function, the -class loss in Eq. can be written compactly as where is the score for class coming from the input vector and the parameters ’s: We find the optimal values of parameters by seeking the minimum of the expected value of the loss over the training dataset: For brevity in writing, let us denote the class corresponding to the class in the datapoint by . For each datapoint and the parameters for its true class , we will have a score: which we can use in the loss function. Thus, we optimize the following loss evaluated from all the datapoints in the dataset: From the indices that label the parameters, it appears that we have parameters, but since probabilities over all classes must add up to , only parameters are independent. This is included in the softmax function since shifting all scores by the same amount does not change the softmax output.    Derivative of Multiclass Loss Function  Taking the derivative of with respect to parameters with (classes) and (bias part and components of feature vector ) will proceed the same way for each datapoint. So, it's better to drop the index from the notation when taking derivatives and restore them in the final results. Then, the required derivatives of one term of the empirical loss function without the factor, to be denoted by , will be where the sum on classes is using a dummy index rather than , which is the index of a particular parameter with respect to which we seek the derivative. Now, the components on the right side can be worked out individually. Thus, we have the following derivatives for the empirical loss function with respect to the parameters. These derivatives are then used in the gradient descent algorithm to find their optimum values as has been illustrated before.    The Iris Classification Problem  The Iris dataset is one of the most famous datasets in machine learning. Collected by the botanist Ronald A. Fisher in 1936, it contains measurements of 150 iris flowers from three different species:   Iris setosa    Iris versicolor    Iris virginica   Each flower is described by four features:   Sepal length    Sepal width    Petal length    Petal width   Sepals are the outer green leaf-like structures that protect the flower bud. Petals are the colorful parts that attract pollinators.  The task is to use these numerical features to predict the species of the iris. This makes it a classic multiclass classification problem with:   Input matrix X of shape (n_samples, 4)    Target labels y with 3 possible categories (Setosa, Versicolor, Virginica)   Because the Iris dataset is small, structured, and easy to visualize, it is often used to demonstrate the fundamentals of machine learning models.  In this project, we implement multiclass logistic regression (softmax regression) from scratch using NumPy. We optimize the model using gradient descent and evaluate it on the Iris dataset.  The program below produces the following output:   Every 100 Epoch print the progress:    Epoch 100, Loss: 0.3407    Epoch 200, Loss: 0.2707    Epoch 300, Loss: 0.2293    Epoch 400, Loss: 0.2011    Epoch 500, Loss: 0.1807    Epoch 600, Loss: 0.1652    Epoch 700, Loss: 0.1531    Epoch 800, Loss: 0.1433    Epoch 900, Loss: 0.1353    Epoch 1000, Loss: 0.1286    Print out the performance at the end of 1000 epochs (i.e., iterations through the entire training dataset) Training accuracy: (i.e., ), Test accuracy: (i.e., ). Here, training accuracy is the accuracy measured on the trtaining dataset and the test accuracy is the one measured on the test dataset.  import numpy as np import matplotlib.pyplot as plt from sklearn.datasets import load_iris from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics import accuracy_score # ------------------------- # 1. Load Iris dataset # ------------------------- X, y = load_iris(return_X_y=True) num_classes = len(np.unique(y)) # 3 classes # Normalize features scaler = StandardScaler() X = scaler.fit_transform(X) # One-hot encode targets y_onehot = np.eye(num_classes)[y] # Train\/test split X_train, X_test, y_train, y_test = train_test_split( X, y_onehot, test_size=0.2, random_state=42 ) # ------------------------- # 2. Softmax + loss # ------------------------- def softmax(logits): exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) return exp_logits \/ np.sum(exp_logits, axis=1, keepdims=True) def cross_entropy_loss(y_true, y_pred): eps = 1e-15 return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)) # ------------------------- # 3. Training with gradient descent # ------------------------- def train_softmax_classification(X, y, num_classes, lr=0.1, epochs=500): n_samples, n_features = X.shape W = np.zeros((n_features, num_classes)) b = np.zeros((1, num_classes)) losses = [] for epoch in range(epochs): # Forward pass logits = X @ W + b y_pred = softmax(logits) # Loss loss = cross_entropy_loss(y, y_pred) losses.append(loss) # Gradients grad_W = (X.T @ (y_pred - y)) \/ n_samples grad_b = np.mean(y_pred - y, axis=0, keepdims=True) # Update W -= lr * grad_W b -= lr * grad_b if (epoch+1) % 100 == 0: print(f\"Epoch {epoch+1}, Loss: {loss:.4f}\") return W, b, losses # ------------------------- # 4. Train model # ------------------------- W, b, losses = train_softmax_classification(X_train, y_train, num_classes, lr=0.1, epochs=1000) # ------------------------- # 5. Predictions \\amp; Accuracy # ------------------------- def predict(X, W, b): logits = X @ W + b return np.argmax(softmax(logits), axis=1) y_pred_train = predict(X_train, W, b) y_pred_test = predict(X_test, W, b) train_acc = accuracy_score(np.argmax(y_train, axis=1), y_pred_train) test_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred_test) print(f\"Training accuracy: {train_acc:.4f}\") print(f\"Test accuracy: {test_acc:.4f}\") # ------------------------- # 6. Plot loss curve # ------------------------- plt.figure(figsize=(6,4)) plt.plot(losses, label=\"Training Loss\") plt.xlabel(\"Epochs\") plt.ylabel(\"Cross-Entropy Loss\") plt.title(\"Loss Curve (Iris Softmax Regression)\") plt.legend() plt.show()    Predicting Hand Written Digits  In this section, we will build a multiclass logistic regression model (also known as softmax regression) from scratch in Python. Instead of relying on pre-built machine learning libraries, we will implement the mathematics ourselves using NumPy and optimize the parameters with gradient descent.  For data, we will use the classic MNIST handwritten digits dataset , a benchmark in machine learning that contains 70,000 grayscale images of digits 0 through 9. Each image is a 28×28 pixel grid, which we flatten into a feature vector of dimension 784. The target labels are integers from 0 to 9. To simplify the task and keep training efficient, we will restrict the dataset to only the first five digits (0–4). This gives us a five-class classification problem with inputs and one-hot encoded outputs , where is the number of samples.  This exercise illustrates how the softmax function converts raw model scores into probabilities across multiple classes and how cross-entropy loss guides learning. It also demonstrates the practical use of MNIST as a stepping stone toward understanding more complex models like neural networks.  The program below produces the following output:   Every 50 Epoch print the progress:    Epoch 50, Loss: 0.1036    Epoch 100, Loss: 0.0939    Epoch 150, Loss: 0.0886    Epoch 200, Loss: 0.0850    Epoch 250, Loss: 0.0822    Epoch 300, Loss: 0.0800    Epoch 350, Loss: 0.0782    Epoch 400, Loss: 0.0766    Epoch 450, Loss: 0.0752    Epoch 500, Loss: 0.0740    Print out the performance at the end of 500 epochs (i.e., iterations through the entire training dataset) Training accuracy: 0.9797, Test accuracy: 0.9671. Here, training accuracy is the accuracy measured on the trtaining dataset and the test accuracy is the one measured on the test dataset.  The program also plots loss versus iterations and some example prediction labels and the corresponding photos in the test dataset.   Predicted labels and actual digits images.   Predicted labels and actual digits images.    import numpy as np import matplotlib.pyplot as plt from sklearn.preprocessing import StandardScaler from sklearn.metrics import accuracy_score from sklearn.model_selection import train_test_split from sklearn.datasets import fetch_openml mnist = fetch_openml('mnist_784', version=1, as_frame=False) X, y = mnist.data, mnist.target.astype(int) # Train\/test split (e.g., 80\/20) X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 ) # ------------------------- # 1. Load MNIST and restrict to digits 0–4 # ------------------------- # (X_train, y_train), (X_test, y_test) = mnist.load_data() # keep only digits 0-4 train_mask = y_train < 5 test_mask = y_test < 5 X_train = X_train[train_mask] y_train = y_train[train_mask] X_test = X_test[test_mask] y_test = y_test[test_mask] # flatten images from 28x28 → 784 X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) # normalize scaler = StandardScaler() X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test) # one-hot labels num_classes = len(np.unique(y_train)) y_train_oh = np.eye(num_classes)[y_train] y_test_oh = np.eye(num_classes)[y_test] # ------------------------- # 2. Softmax + loss # ------------------------- def softmax(logits): exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) return exp_logits \/ np.sum(exp_logits, axis=1, keepdims=True) def cross_entropy_loss(y_true, y_pred): eps = 1e-15 return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)) # ------------------------- # 3. Gradient descent training # ------------------------- def train_softmax_classification(X, y, num_classes, lr=0.1, epochs=300): n_samples, n_features = X.shape W = np.zeros((n_features, num_classes)) b = np.zeros((1, num_classes)) losses = [] for epoch in range(epochs): logits = X @ W + b y_pred = softmax(logits) loss = cross_entropy_loss(y, y_pred) losses.append(loss) grad_W = (X.T @ (y_pred - y)) \/ n_samples grad_b = np.mean(y_pred - y, axis=0, keepdims=True) W -= lr * grad_W b -= lr * grad_b if (epoch+1) % 50 == 0: print(f\"Epoch {epoch+1}, Loss: {loss:.4f}\") return W, b, losses # ------------------------- # 4. Train # ------------------------- W, b, losses = train_softmax_classification(X_train, y_train_oh, num_classes, lr=0.5, epochs=500) # ------------------------- # 5. Evaluate # ------------------------- def predict(X, W, b): logits = X @ W + b return np.argmax(softmax(logits), axis=1) y_pred_train = predict(X_train, W, b) y_pred_test = predict(X_test, W, b) train_acc = accuracy_score(y_train, y_pred_train) test_acc = accuracy_score(y_test, y_pred_test) print(f\"Training accuracy: {train_acc:.4f}\") print(f\"Test accuracy: {test_acc:.4f}\") # ------------------------- # 6. Plot loss curve # ------------------------- plt.figure(figsize=(6,4)) plt.plot(losses, label=\"Training Loss\") plt.xlabel(\"Epochs\") plt.ylabel(\"Cross-Entropy Loss\") plt.title(\"Loss Curve\") plt.legend() plt.show() # ------------------------- # 7. Plot sample predictions # ------------------------- fig, axes = plt.subplots(2, 5, figsize=(10,5)) indices = np.random.choice(len(X_test), size=10, replace=False) for ax, idx in zip(axes.flat, indices): img = X_test[idx].reshape(28, 28) ax.imshow(img, cmap=\"gray\") ax.axis(\"off\") pred = y_pred_test[idx] true = y_test[idx] ax.set_title(f\"P:{pred}, T:{true}\") plt.suptitle(\"Sample Predictions (P=Predicted, T=True)\") plt.show()     Model Evaluations     "
},
{
  "id": "sec-MultiClass-Classification-2-1",
  "level": "2",
  "url": "sec-MultiClass-Classification.html#sec-MultiClass-Classification-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "multiclass classification multilabel prediction task "
},
{
  "id": "subsec-Loss-Function-of-Multi-class-Classification-2",
  "level": "2",
  "url": "sec-MultiClass-Classification.html#subsec-Loss-Function-of-Multi-class-Classification-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "score softmax function "
},
{
  "id": "subsec-derivative-of-multiclass-loss-function-3",
  "level": "2",
  "url": "sec-MultiClass-Classification.html#subsec-derivative-of-multiclass-loss-function-3",
  "type": "Example",
  "number": "2.4.1",
  "title": "The Iris Classification Problem.",
  "body": " The Iris Classification Problem  The Iris dataset is one of the most famous datasets in machine learning. Collected by the botanist Ronald A. Fisher in 1936, it contains measurements of 150 iris flowers from three different species:   Iris setosa    Iris versicolor    Iris virginica   Each flower is described by four features:   Sepal length    Sepal width    Petal length    Petal width   Sepals are the outer green leaf-like structures that protect the flower bud. Petals are the colorful parts that attract pollinators.  The task is to use these numerical features to predict the species of the iris. This makes it a classic multiclass classification problem with:   Input matrix X of shape (n_samples, 4)    Target labels y with 3 possible categories (Setosa, Versicolor, Virginica)   Because the Iris dataset is small, structured, and easy to visualize, it is often used to demonstrate the fundamentals of machine learning models.  In this project, we implement multiclass logistic regression (softmax regression) from scratch using NumPy. We optimize the model using gradient descent and evaluate it on the Iris dataset.  The program below produces the following output:   Every 100 Epoch print the progress:    Epoch 100, Loss: 0.3407    Epoch 200, Loss: 0.2707    Epoch 300, Loss: 0.2293    Epoch 400, Loss: 0.2011    Epoch 500, Loss: 0.1807    Epoch 600, Loss: 0.1652    Epoch 700, Loss: 0.1531    Epoch 800, Loss: 0.1433    Epoch 900, Loss: 0.1353    Epoch 1000, Loss: 0.1286    Print out the performance at the end of 1000 epochs (i.e., iterations through the entire training dataset) Training accuracy: (i.e., ), Test accuracy: (i.e., ). Here, training accuracy is the accuracy measured on the trtaining dataset and the test accuracy is the one measured on the test dataset.  import numpy as np import matplotlib.pyplot as plt from sklearn.datasets import load_iris from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.metrics import accuracy_score # ------------------------- # 1. Load Iris dataset # ------------------------- X, y = load_iris(return_X_y=True) num_classes = len(np.unique(y)) # 3 classes # Normalize features scaler = StandardScaler() X = scaler.fit_transform(X) # One-hot encode targets y_onehot = np.eye(num_classes)[y] # Train\/test split X_train, X_test, y_train, y_test = train_test_split( X, y_onehot, test_size=0.2, random_state=42 ) # ------------------------- # 2. Softmax + loss # ------------------------- def softmax(logits): exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) return exp_logits \/ np.sum(exp_logits, axis=1, keepdims=True) def cross_entropy_loss(y_true, y_pred): eps = 1e-15 return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)) # ------------------------- # 3. Training with gradient descent # ------------------------- def train_softmax_classification(X, y, num_classes, lr=0.1, epochs=500): n_samples, n_features = X.shape W = np.zeros((n_features, num_classes)) b = np.zeros((1, num_classes)) losses = [] for epoch in range(epochs): # Forward pass logits = X @ W + b y_pred = softmax(logits) # Loss loss = cross_entropy_loss(y, y_pred) losses.append(loss) # Gradients grad_W = (X.T @ (y_pred - y)) \/ n_samples grad_b = np.mean(y_pred - y, axis=0, keepdims=True) # Update W -= lr * grad_W b -= lr * grad_b if (epoch+1) % 100 == 0: print(f\"Epoch {epoch+1}, Loss: {loss:.4f}\") return W, b, losses # ------------------------- # 4. Train model # ------------------------- W, b, losses = train_softmax_classification(X_train, y_train, num_classes, lr=0.1, epochs=1000) # ------------------------- # 5. Predictions \\amp; Accuracy # ------------------------- def predict(X, W, b): logits = X @ W + b return np.argmax(softmax(logits), axis=1) y_pred_train = predict(X_train, W, b) y_pred_test = predict(X_test, W, b) train_acc = accuracy_score(np.argmax(y_train, axis=1), y_pred_train) test_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred_test) print(f\"Training accuracy: {train_acc:.4f}\") print(f\"Test accuracy: {test_acc:.4f}\") # ------------------------- # 6. Plot loss curve # ------------------------- plt.figure(figsize=(6,4)) plt.plot(losses, label=\"Training Loss\") plt.xlabel(\"Epochs\") plt.ylabel(\"Cross-Entropy Loss\") plt.title(\"Loss Curve (Iris Softmax Regression)\") plt.legend() plt.show()  "
},
{
  "id": "subsec-derivative-of-multiclass-loss-function-4",
  "level": "2",
  "url": "sec-MultiClass-Classification.html#subsec-derivative-of-multiclass-loss-function-4",
  "type": "Example",
  "number": "2.4.2",
  "title": "Predicting Hand Written Digits.",
  "body": "Predicting Hand Written Digits  In this section, we will build a multiclass logistic regression model (also known as softmax regression) from scratch in Python. Instead of relying on pre-built machine learning libraries, we will implement the mathematics ourselves using NumPy and optimize the parameters with gradient descent.  For data, we will use the classic MNIST handwritten digits dataset , a benchmark in machine learning that contains 70,000 grayscale images of digits 0 through 9. Each image is a 28×28 pixel grid, which we flatten into a feature vector of dimension 784. The target labels are integers from 0 to 9. To simplify the task and keep training efficient, we will restrict the dataset to only the first five digits (0–4). This gives us a five-class classification problem with inputs and one-hot encoded outputs , where is the number of samples.  This exercise illustrates how the softmax function converts raw model scores into probabilities across multiple classes and how cross-entropy loss guides learning. It also demonstrates the practical use of MNIST as a stepping stone toward understanding more complex models like neural networks.  The program below produces the following output:   Every 50 Epoch print the progress:    Epoch 50, Loss: 0.1036    Epoch 100, Loss: 0.0939    Epoch 150, Loss: 0.0886    Epoch 200, Loss: 0.0850    Epoch 250, Loss: 0.0822    Epoch 300, Loss: 0.0800    Epoch 350, Loss: 0.0782    Epoch 400, Loss: 0.0766    Epoch 450, Loss: 0.0752    Epoch 500, Loss: 0.0740    Print out the performance at the end of 500 epochs (i.e., iterations through the entire training dataset) Training accuracy: 0.9797, Test accuracy: 0.9671. Here, training accuracy is the accuracy measured on the trtaining dataset and the test accuracy is the one measured on the test dataset.  The program also plots loss versus iterations and some example prediction labels and the corresponding photos in the test dataset.   Predicted labels and actual digits images.   Predicted labels and actual digits images.    import numpy as np import matplotlib.pyplot as plt from sklearn.preprocessing import StandardScaler from sklearn.metrics import accuracy_score from sklearn.model_selection import train_test_split from sklearn.datasets import fetch_openml mnist = fetch_openml('mnist_784', version=1, as_frame=False) X, y = mnist.data, mnist.target.astype(int) # Train\/test split (e.g., 80\/20) X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 ) # ------------------------- # 1. Load MNIST and restrict to digits 0–4 # ------------------------- # (X_train, y_train), (X_test, y_test) = mnist.load_data() # keep only digits 0-4 train_mask = y_train < 5 test_mask = y_test < 5 X_train = X_train[train_mask] y_train = y_train[train_mask] X_test = X_test[test_mask] y_test = y_test[test_mask] # flatten images from 28x28 → 784 X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) # normalize scaler = StandardScaler() X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test) # one-hot labels num_classes = len(np.unique(y_train)) y_train_oh = np.eye(num_classes)[y_train] y_test_oh = np.eye(num_classes)[y_test] # ------------------------- # 2. Softmax + loss # ------------------------- def softmax(logits): exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) return exp_logits \/ np.sum(exp_logits, axis=1, keepdims=True) def cross_entropy_loss(y_true, y_pred): eps = 1e-15 return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1)) # ------------------------- # 3. Gradient descent training # ------------------------- def train_softmax_classification(X, y, num_classes, lr=0.1, epochs=300): n_samples, n_features = X.shape W = np.zeros((n_features, num_classes)) b = np.zeros((1, num_classes)) losses = [] for epoch in range(epochs): logits = X @ W + b y_pred = softmax(logits) loss = cross_entropy_loss(y, y_pred) losses.append(loss) grad_W = (X.T @ (y_pred - y)) \/ n_samples grad_b = np.mean(y_pred - y, axis=0, keepdims=True) W -= lr * grad_W b -= lr * grad_b if (epoch+1) % 50 == 0: print(f\"Epoch {epoch+1}, Loss: {loss:.4f}\") return W, b, losses # ------------------------- # 4. Train # ------------------------- W, b, losses = train_softmax_classification(X_train, y_train_oh, num_classes, lr=0.5, epochs=500) # ------------------------- # 5. Evaluate # ------------------------- def predict(X, W, b): logits = X @ W + b return np.argmax(softmax(logits), axis=1) y_pred_train = predict(X_train, W, b) y_pred_test = predict(X_test, W, b) train_acc = accuracy_score(y_train, y_pred_train) test_acc = accuracy_score(y_test, y_pred_test) print(f\"Training accuracy: {train_acc:.4f}\") print(f\"Test accuracy: {test_acc:.4f}\") # ------------------------- # 6. Plot loss curve # ------------------------- plt.figure(figsize=(6,4)) plt.plot(losses, label=\"Training Loss\") plt.xlabel(\"Epochs\") plt.ylabel(\"Cross-Entropy Loss\") plt.title(\"Loss Curve\") plt.legend() plt.show() # ------------------------- # 7. Plot sample predictions # ------------------------- fig, axes = plt.subplots(2, 5, figsize=(10,5)) indices = np.random.choice(len(X_test), size=10, replace=False) for ax, idx in zip(axes.flat, indices): img = X_test[idx].reshape(28, 28) ax.imshow(img, cmap=\"gray\") ax.axis(\"off\") pred = y_pred_test[idx] true = y_test[idx] ax.set_title(f\"P:{pred}, T:{true}\") plt.suptitle(\"Sample Predictions (P=Predicted, T=True)\") plt.show()  "
},
{
  "id": "sec-Evaluating-Predictions",
  "level": "1",
  "url": "sec-Evaluating-Predictions.html",
  "type": "Section",
  "number": "2.5",
  "title": "Evaluating Predictions",
  "body": " Evaluating Predictions   So far we have talked about learning parameters for models of the conditional probability in various types of problems using a dataset of values called the training dataset . We call the with optimized parameters is called a trained model .  To evaluate the quality of a trained model, so as not to introduce bias in the process we need a fresh dataset, which was not used during the training of the model, or, in any way could have biased the training process. This pristine set-aside-for-test dataset is called the test dataset . Often, the strategy is to split all the data we have into two subsets for training and testing purposes. We use the test subset solely for testing.  We will use the following notation for the training and test subsets:   The quality of a model is then tested on the test dataset using a scoring function that closely aligns with the objective of the project. Since the evaluation metrics depend on the type of questions we want to answer, it is best to describe them with examples as we will illustrate in this section.        Evaluating Regression Tasks  In regression tasks, the trained model is usually a probability density function , where are the estimated or trained parameters. Suppose, we want to predict the best given an , what should be the predictor function and how should we evaluate it? The answer depends on whether is a skewed distribution or not.   Non-skewed Distribution   Suppose, is not skewed. Then, the best prediction for is the mean value of . Therefore, the predictor function will be where I have included the parameter indicator so that we are aware that the predictor function will depend upon the parameters of the model for which are estimated\/trained using the trainign dataset .  To evaluate the quality of this predictor, the average value of the Mean Squared Error is the score function that also minimizes the training error when used during the training f the model. The formula for squared error is Evaluating its average over the test set gives us the mean squared error. That is, mean squared error is empirical expectation of the squared error (a random variable) in the distribution represented by the test dataset, with each datapoint weighing equally. When we use this same formula during the training, we minimze the following error function. where 's are to be adjusted during the training. You should note different uses of the squared error during training and testing for this task.   Non-skewed Distribution   If is skewed or if you suspect that the training data has outliers that have influenced the estimation process, then, a better prediction of best  will be the median of the distribution. The score function in that case would be the expectation of the absolute difference between the predictor and the true. This score function is called mean absolute error (MAE)      Evaluating Binary Classification  In a binary classification, the task is to predict whether a given belongs to one of the two classes, (i.e., True\/Success) or (i.e., False\/Failure). We have modeled the conditional probability mass function , i.e., probability of outcome being for a given , as a sigmoid in the logistic regression algorithm, whose parameters were optimized by using the training data .  For brevity, let us denote the trained probability by This will be our predictor when combined with a threshold for decision. Thus, at the end of training, we end up with a trained model, , which directly predicts by the following decision rules: Note that depends upon and the training dataset. We will just show it as . In our examples, we used the threshold . We will discuss below a systematic way of choosing appropriate threshold for a problem. For now, suppose, we have rule for converting trained model into predictions on the outcomes.   Accuracy   To evaluate the trained model, a simple measure is the percentage of the times we got the prediction right. This is called accuracy . The score function for this measure will just be a Kronecker delta of the predictor and the corresponding . As a random variable, accuracy will by The accuracy will be the expectation value of this random variable. When it is evaluated on the test dataset we will get the following.    Confusion Matrix   For many problems, accuracy turns out to be a good enough metric for evaluating the quality of predictions. Accuracy, however, misses some details that you might care about. For instance, it lumps together being right on both and cases and you might care more about being right on, say the outcome than on the outcome . In these cases, you might want to look more closely at when you are wrong and when you are right.  Just as there are two events when you are right, i.e., you predict and data (i.e., true) value is also and you predict when the fata value of also , there are two ways of being wrong: you predict but the data value of and yoou predict but the data value is . Suppose, we refer to as positive and as negative , then we call these various types of outcomes by different names.    True Positive (TP) : You predict and data (i.e., true) value is also .   True Negative (TN) : You predict and data (i.e., true) value is also .   False Positive (FP) : You predict but the data (i.e., true) value is .   False Negative (FN) : You predict but the data (i.e., true) value is .   Suppose, there are points in the test dataset and you find , , , and . It is a common practice to present these numbers in a matrix, aptly called the confusion matrix .          Actual  Positive  Negative  Totals    Predicted    Predicted    Positive  TP  FP     Negative  FN  TN           Totals Actuals                Actual  Positive  Negative  Totals    Predicted    Predicted    Positive       Negative             Totals Actuals        From the confusion matrix, you can further deduce important metric of the performance of a model. For instance, we may want to know, What percentage of the actual were we able to predict to be ? It's called recall or True Positive Rate (TPR) . All the positives in the real data are now in the TP and FN since FN are actually positives which got incorrectly predicted to be negatives. Sometimes, it's more important to know what proportions of all my positive predictions was actuall prositive in the test dataset. This measure is called precision . There is also a False Positive Rate (FPR) . This measures mistakes in predictions made by the model when looking at the negatives in the test data. It is calculated by dividing the FP (incorrect predictions on negatives in the test data) by the number of datapoints that were actually negatives. The accuracy that we have discussed before will be It would be a good exercise for the student to find the values of these quantities from the table of values given above.   Role of Threshold in Decision   As you know that the decision to predict a value of the target is based on comparing the predicted probability (which is the probability of the event that given an ) to a threshold , which we had taken to be . That is if , we predicted that is category for the given . This basically says, that the given is more than likely to be choice .  Now, imagine, is a picture captured by a security camera and opens the door to a high security building. You really want a more stringent requirement that to let people in the building. When you increase the threshold, you lower the False Positives since you do not predict as many positives as you did before, but also increase the False Negatives because now some of the true will be rejected by prediction.  One way to implement different importance of different outcomes is to associate cost to different outcomes in a matrix. This matrix is called cost matrix , . The elements of will be as follows. For a binary classificaiton, and . The costs for the correct predictions would, of course, be zero. Thus,               Actual  Positive  Negative      Predicted       Positive       Negative                      The error incurred in predicting the test data can be then weighed accordingly. Thus, in case of the security identification, we might impose a heavy cost of False positive and low for False Negative. How heavy should the relative ratios be? That depends on the requirements of the system. For instance, you could demand a -fold cost when testing. During the training, you might also make the decision to reach harder by increasing the threshold. Increasing the threshod will lower th false positives, but will also increase false negatives, leading the system to deny entry to the building to some who should be let in. Another instance for including cost in the training process occurs when we have an imbalanced dataset , e.g. may be you have spam email for every regular mail, and you are trying to predict if a given email is a spam or not. You can immediately see that if you predicted every email not be to be spam email, you will only be wrong of the time. That is not wuite what you want. So, you would have to try to balance the two classes of data in your training set. One strategy is to weigh each regular email class just for each spam email times. We will discuss these strategies of training imbalanced data in a later chapter.   "
},
{
  "id": "sec-Evaluating-Predictions-2-1",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#sec-Evaluating-Predictions-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "the training dataset trained model "
},
{
  "id": "sec-Evaluating-Predictions-2-2",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#sec-Evaluating-Predictions-2-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "the test dataset "
},
{
  "id": "sec-Evaluating-Predictions-2-4",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#sec-Evaluating-Predictions-2-4",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "scoring function "
},
{
  "id": "subsec-Evaluating-Regression-Tasks-5",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Regression-Tasks-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Mean Squared Error score function "
},
{
  "id": "subsec-Evaluating-Regression-Tasks-7",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Regression-Tasks-7",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "mean absolute error (MAE) "
},
{
  "id": "subsec-Evaluating-Binary-Classification-5",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Binary-Classification-5",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "accuracy accuracy "
},
{
  "id": "subsec-Evaluating-Binary-Classification-8",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Binary-Classification-8",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "True Positive (TP) True Negative (TN) False Positive (FP) False Negative (FN) confusion matrix "
},
{
  "id": "subsec-Evaluating-Binary-Classification-10",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Binary-Classification-10",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "recall True Positive Rate (TPR) precision False Positive Rate (FPR) accuracy "
},
{
  "id": "subsec-Evaluating-Binary-Classification-14",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Binary-Classification-14",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "cost matrix "
},
{
  "id": "subsec-Evaluating-Binary-Classification-16",
  "level": "2",
  "url": "sec-Evaluating-Predictions.html#subsec-Evaluating-Binary-Classification-16",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "imbalanced dataset "
},
{
  "id": "sec-Bayesian-Methods",
  "level": "1",
  "url": "sec-Bayesian-Methods.html",
  "type": "Section",
  "number": "2.6",
  "title": "Bayesian Methods",
  "body": " Bayesian Methods   In supervised learning, we often model the conditional probability . This naturally appears in Bayes’ theorem as the posterior probability . Recall the identity: Equivalently, Bayes’ theorem can be written as When we use Bayes’ theorem to model conditional probabilities and guide decision-making, the resulting approaches are called Bayesian methods .  For comparing possible outcomes given the same , the denominator cancels out, so we only need relative probabilities: Let us call the right side un-normalized probabilities . Then, to make decision of which outcome is more likely, we just compare their unnormalized probabilities rather than the actual probabilities since the two woudl give the same answer. Therefore, we choose over by Thus the unnormalized probabilites are sufficient to decide which outcome of is more probable for a given without needing to know the full distribution . All we need are:  The conditional distributions , and  The prior probabilities , i.e., the marginal probabilities of each outcome of .    If is a discrete variable with classes, then, you would just use the function . Basically, arg max goes over the list of values generated on the right side for various values of , i.e., for , and then outputs the index of , which corresponds to the largest value in the list. If you have ties, then it outputs the first in the list. You can also have it printout all the items in the tie and output one randomly from that sublist.    Naive Bayes Algorithm  The Naive Bayes algorithm applies Bayes’ theorem under a simplifying assumption: the features of are conditionally independent given the class label . Formally, This assumption is rarely true in practice, but it makes the computation of tractable, and the resulting classifier often performs surprisingly well.  Substituting this factorization into Bayes’ theorem, we obtain A Naive Bayes classifier assigns the label    Example: Suppose we want to classify whether an email is spam or not spam based on two binary features: “contains the word ‘free’” and “contains the word ‘money’.” From training data we estimate: If an email contains both words ( ), then Since , the classifier predicts spam .   Example: Suppose we want to classify whether an email is spam or not spam based on two binary features: “contains the word ‘deal’” and “contains the word ‘win’.” From training data we estimate: Now consider three scenarios:  If the email contains only “deal” ( ): The classifier predicts not spam .  If the email contains only “win” ( ): Again the classifier predicts not spam .  If the email contains both words ( ): Now the classifier predicts spam .     In this example, I present a complete Python program that uses the Naive Bayes classifier from scikit-learn. I use scikit-learn library to save the hassle of programming from scratch, which we have practiced on simpler problems before to get a sense of the math steps. Later on, we will use other libraries such as TensorFlow and PyTorch.   Loads a dataset that’s freely available online (the SMS Spam Collection dataset hosted by UCI) by downloading the dataset (4,827 ham, 747 spam). Uses a bag-of-words (CountVectorizer) to turn text into features.    Splits into train\/test sets.    Trains the Naive Bayes model.    Evaluates it using a confusion matrix.    Prints precision, recall, F1-score.     When you run the program, you would get outputs like the following. Vocabulary size: 7441 Accuracy: 0.9883408071748879 Confusion Matrix: [[961 5] [ 8 141]] Classification Report: precision recall f1-score support ham 0.99 0.99 0.99 966 spam 0.97 0.95 0.96 149 accuracy 0.99 1115 macro avg 0.98 0.97 0.97 1115 weighted avg 0.99 0.99 0.99 1115 New message predictions (0=ham, 1=spam): Message: Free offer! Win a $1000 gift card now!... -> spam Message: Hey, let's meet tomorrow at 5pm.... -> ham  # Naive Bayes Example: SMS Spam Detection # Step 1: Import libraries import pandas as pd from sklearn.feature_extraction.text import CountVectorizer from sklearn.model_selection import train_test_split from sklearn.naive_bayes import MultinomialNB from sklearn.metrics import confusion_matrix, classification_report import seaborn as sns import matplotlib.pyplot as plt # Step 2: Load dataset (SMS Spam Collection from UCI repository) url = \"https:\/\/archive.ics.uci.edu\/ml\/machine-learning-databases\/00228\/smsspamcollection.zip\" df = pd.read_csv(url, compression=\"zip\", sep=\"\\t\", names=[\"label\", \"message\"]) print(\"Dataset shape:\", df.shape) print(df.head()) # Step 3: Preprocess labels df[\"label\"] = df[\"label\"].map({\"ham\": 0, \"spam\": 1}) # ham=0, spam=1 # Step 4: Train\/test split X_train, X_test, y_train, y_test = train_test_split( df[\"message\"], df[\"label\"], test_size=0.2, random_state=42 ) # Step 5: Convert text to numerical features vectorizer = CountVectorizer(stop_words=\"english\") X_train_vec = vectorizer.fit_transform(X_train) X_test_vec = vectorizer.transform(X_test) # Step 6: Train Naive Bayes classifier model = MultinomialNB() model.fit(X_train_vec, y_train) # Step 7: Predictions y_pred = model.predict(X_test_vec) # Step 8: Confusion matrix cm = confusion_matrix(y_test, y_pred) plt.figure(figsize=(5, 4)) sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=[\"Ham\", \"Spam\"], yticklabels=[\"Ham\", \"Spam\"]) plt.xlabel(\"Predicted\") plt.ylabel(\"Actual\") plt.title(\"Confusion Matrix\") plt.show() # Step 9: Classification report print(\"\\nClassification Report:\\n\") print(classification_report(y_test, y_pred, target_names=[\"Ham\", \"Spam\"]))    "
},
{
  "id": "sec-Bayesian-Methods-2-1",
  "level": "2",
  "url": "sec-Bayesian-Methods.html#sec-Bayesian-Methods-2-1",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Bayesian methods "
},
{
  "id": "subsec-Naive-Bayes-2",
  "level": "2",
  "url": "sec-Bayesian-Methods.html#subsec-Naive-Bayes-2",
  "type": "Paragraph (with a defined term)",
  "number": "",
  "title": "",
  "body": "Naive Bayes "
},
{
  "id": "subsec-Naive-Bayes-6",
  "level": "2",
  "url": "sec-Bayesian-Methods.html#subsec-Naive-Bayes-6",
  "type": "Example",
  "number": "2.6.1",
  "title": "",
  "body": " In this example, I present a complete Python program that uses the Naive Bayes classifier from scikit-learn. I use scikit-learn library to save the hassle of programming from scratch, which we have practiced on simpler problems before to get a sense of the math steps. Later on, we will use other libraries such as TensorFlow and PyTorch.   Loads a dataset that’s freely available online (the SMS Spam Collection dataset hosted by UCI) by downloading the dataset (4,827 ham, 747 spam). Uses a bag-of-words (CountVectorizer) to turn text into features.    Splits into train\/test sets.    Trains the Naive Bayes model.    Evaluates it using a confusion matrix.    Prints precision, recall, F1-score.     When you run the program, you would get outputs like the following. Vocabulary size: 7441 Accuracy: 0.9883408071748879 Confusion Matrix: [[961 5] [ 8 141]] Classification Report: precision recall f1-score support ham 0.99 0.99 0.99 966 spam 0.97 0.95 0.96 149 accuracy 0.99 1115 macro avg 0.98 0.97 0.97 1115 weighted avg 0.99 0.99 0.99 1115 New message predictions (0=ham, 1=spam): Message: Free offer! Win a $1000 gift card now!... -> spam Message: Hey, let's meet tomorrow at 5pm.... -> ham  # Naive Bayes Example: SMS Spam Detection # Step 1: Import libraries import pandas as pd from sklearn.feature_extraction.text import CountVectorizer from sklearn.model_selection import train_test_split from sklearn.naive_bayes import MultinomialNB from sklearn.metrics import confusion_matrix, classification_report import seaborn as sns import matplotlib.pyplot as plt # Step 2: Load dataset (SMS Spam Collection from UCI repository) url = \"https:\/\/archive.ics.uci.edu\/ml\/machine-learning-databases\/00228\/smsspamcollection.zip\" df = pd.read_csv(url, compression=\"zip\", sep=\"\\t\", names=[\"label\", \"message\"]) print(\"Dataset shape:\", df.shape) print(df.head()) # Step 3: Preprocess labels df[\"label\"] = df[\"label\"].map({\"ham\": 0, \"spam\": 1}) # ham=0, spam=1 # Step 4: Train\/test split X_train, X_test, y_train, y_test = train_test_split( df[\"message\"], df[\"label\"], test_size=0.2, random_state=42 ) # Step 5: Convert text to numerical features vectorizer = CountVectorizer(stop_words=\"english\") X_train_vec = vectorizer.fit_transform(X_train) X_test_vec = vectorizer.transform(X_test) # Step 6: Train Naive Bayes classifier model = MultinomialNB() model.fit(X_train_vec, y_train) # Step 7: Predictions y_pred = model.predict(X_test_vec) # Step 8: Confusion matrix cm = confusion_matrix(y_test, y_pred) plt.figure(figsize=(5, 4)) sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=[\"Ham\", \"Spam\"], yticklabels=[\"Ham\", \"Spam\"]) plt.xlabel(\"Predicted\") plt.ylabel(\"Actual\") plt.title(\"Confusion Matrix\") plt.show() # Step 9: Classification report print(\"\\nClassification Report:\\n\") print(classification_report(y_test, y_pred, target_names=[\"Ham\", \"Spam\"]))  "
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
