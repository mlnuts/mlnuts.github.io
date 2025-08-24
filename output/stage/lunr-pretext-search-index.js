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
  "body": " Descriptive Statistics   Descriptive statistics summarize key features of a dataset, providing insights into its central tendency, dispersion, and shape. This process, known as Exploratory Data Analysis (EDA) , helps identify patterns and trends before applying advanced statistical methods. Common measures include mean, median, mode, variance, standard deviation, range, quartiles, and visualizations like histograms and boxplots. These tools are essential for understanding data in fields like education, finance, and science.    Measures of Central Tendency  Measures of central tendency describe the \"typical\" value in a dataset.     Mean (Average) : The mean, denoted , is the sum of all data points divided by their count. For a dataset with points, the mean is: where .  Example: For student grades , the mean is:   However, mean of a dataset can be misleading if you have a few ouliers since mean is very senisitve to outliers. For instance, say you have a dataset of income, which is , , , , . Clearly, most of the income is in the area, but the mean of this dataset is , skewed by the outlier. In this case, the median would better represents the typical value in the dataset.     Median : The median is the middle value in a sorted dataset, where of the data lies below and above. For odd , it's the middle value; for even , it's the average of the two middle values.  Example: For (sorted), median = . For , median . For incomes , , , , , median = , robust to the outlier.     Mode : The mode is the most frequent value. A dataset may have no mode, one mode (unimodal), or multiple modes (bimodal or multimodal).  Example: has mode 90. is bimodal . has no mode.     Comparison : Consider incomes , , , , . Mean = , median = , mode = none. The median best reflects the typical income due to the outlier. See for a visual comparison.   Density plot of incomes with mean, median, and no mode.   Density plot showing central tendency measures.       Measures of Dispersion  Dispersion measures how spread out data is around the central tendency.     Variance and Standard Deviation : Variance ( ) measures average squared deviation from the mean; standard deviation ( ) is its square root, in the same units as the data.  For a population: where is the population (true) mean. The data collected from a polulation is called sample. From the sample we can only calculate as estimate of the corresponding population quantities. We define estimate of sample variance by keeping the same divisor as in the true variance definitionor, define with a divisor , which is called an unbiased estimate of variance. where is the sample mean.   Example: For grades , . Population variance: This will give the standard deviation , : Sample variance, on the other hand, will be: , and sample standard deviation .   illustrated the tighter vs. wider spread for a low variance (e.g., ) vs. high variance (e.g., ).   Comparing low and high variance datasets.   Comparison of low and high variance data.        Range and Quartiles : Range = max - min. Quartiles divide sorted data into four parts: Q1 (25th percentile), Q2 (median, 50th), Q3 (75th). Use linear interpolation: position = , where .  Example: For grades , . Median . , . Range . . Outliers: These grades have no outliers.      Distribution Shape   Histogram : Histograms show frequency distributions by grouping data into bins of equal size from min to a bin that includes the max data. So, if you have data from to with a bin size . Then, bins will have , , , till you have exhausted all data. The last bin may extend beyond the data as in the example below.  Example: For grades  , with bin size from to , see .   Histogram of Grades    Bin  Range  Data  Count  Frequency    1    4  0.333    2    3  0.333    3    2  0.222    4    1  0.111     Many computer libraries have histogram plotting routines. For instance was generated from the Python program listed after it. The histogram has been decorated with the mean and median of the data also.   Histogram of grades with mean and median.   Histogram with mean and median lines.     Example Histogram  import matplotlib.pyplot as plt import numpy as np data = [70, 72, 75, 75, 80, 82, 85, 93, 95, 100] bins = [70, 80, 90, 100, 110] freq_arr, bins_arr = np.histogram(data, bins) # returns frequency width = bins_arr[1:] - bins_arr[:-1] plt.figure(figsize=(8, 5)) plt.hist(data, bins=bins, edgecolor='black', alpha=0.7) # this is just plt.bar(bins_arr[:-1], freq_arr, width) mean = np.mean(data) median = np.median(data) plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.1f}') plt.axvline(median, color='green', linestyle='-', label=f'Median = {median:.1f}') plt.xlabel('Grade') plt.ylabel('Frequency') plt.title('Histogram of Student Grades') plt.xticks(bins) plt.grid(axis='y', alpha=0.3) plt.legend() plt.savefig('histogram.png') plt.show()    Boxplot : Boxplots show min, Q1, median, Q3, max (whiskers), and outliers (points beyond Q1 IQR or Q3 IQR).  Example: For grades with an outlier , , , , . Outliers: ≤ is above . See .   Boxplot of grades with annotated quartiles and outlier.   Boxplot with one outlier.     Updated boxplot with annotations  import matplotlib.pyplot as plt import numpy as np data = [70, 75, 80, 85, 90, 95, 100, 150] plt.figure(figsize=(8, 4)) bp = plt.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red')) q1, median, q3 = np.percentile(data, [25, 50, 75]) plt.text(q1 - 5, 1.1, 'Q1', ha='right') plt.text(median, 1.1, 'Median', ha='center') plt.text(q3 + 5, 1.1, 'Q3', ha='left') plt.text(150, 1.3, 'Outlier', ha='center') plt.title('Boxplot of Student Grades') plt.xlabel('Grade') plt.grid(True, alpha=0.3) plt.savefig('boxplot.png') plt.show()    Skewness : Skewness tells us about the shape of the distribution, specifically if it's \"tilted\" to one side. In a positively skewed distribution (right skew), the tail on the right side is longer, while in a negatively skewed distribution (left skew), the tail on the left side is longer.  Positive skew example: Imagine a dataset of household incomes: . The income of 1 million is much higher than the others, causing the data to be right-skewed. Most people earn a lower income, but a few very high incomes stretch the right side of the distribution, creating a longer right tail.  Negative skew example: Think of a set of exam scores . If most students score high but a few perform very poorly, the data is left-skewed. The low scores create a long tail on the left side of the distribution.  See for a visual representation.   Histograms comparing a normal distribution and a right-skewed distribution.   Comparison of normal vs. skewed distributions.    To help visualize the difference, here’s a Python code that generates two types of distributions: a normal (symmetrical) one and a right-skewed one. The plot will show how the shapes of these two distributions differ.   Skewness Visualization  import matplotlib.pyplot as plt import numpy as np from scipy.stats import norm, skewnorm np.random.seed(42) normal_data = np.random.normal(50, 10, 1000) # Normal distribution skewed_data = np.random.exponential(20000, 1000) # Right-skewed distribution fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # normal distribution ax1.hist(normal_data, bins=30, edgecolor='black', alpha=0.7) ax1.set_title('Normal Distribution') ax1.set_xlabel('Value') ax1.set_ylabel('Frequency') #Plot right-skewed distribution ax2.hist(skewed_data, bins=30, edgecolor='black', alpha=0.7) ax2.set_title('Right-Skewed Distribution') ax2.set_xlabel('Value') ax2.set_ylabel('Frequency') plt.tight_layout() plt.savefig('skewness-comparison.png') plt.show()     Kurtosis: Checking Out the Tails   After digging into mean, median, mode, and skewness, let’s talk kurtosis . This stat zooms in on the \"tailedness\" of our data—how often we see wild outliers compared to a normal distribution. It’s like checking the edges of our data’s shape to see if they’re loaded with extreme values or totally chill.    Types of Kurtosis  Kurtosis comes in three flavors:   Mesokurtic : Matches a normal distribution. Not too many outliers, not too few. It’s the just-right vibe.  Leptokurtic : Sharp peak, heavy tails. Think lots of outliers, like stock prices during a market rollercoaster.  Platykurtic : Flatter peak, light tails. Fewer outliers, like steady daily temps.     How’s It Calculated?  Kurtosis measures how much data hangs out in the tails. The formula for excess kurtosis (comparing to a normal distribution) is:     Here, is the number of data points, is each data point, is the mean, and is the standard deviation. Don’t stress the math—Python will handle it in a bit!    Real-World Examples  Let's connect kurtosis to mean, median, mode, and skewness:   Leptokurtic Example : Take stock returns: {5, 7, 8, 8, 10, 12, 12, 12, 12, 50} . That 50 is a massive outlier, bulking up the tails (like skewness, but focused on extremes). This is leptokurtic—expect some crazy swings.  Platykurtic Example : Now daily temperatures: {30, 32, 33, 34, 35, 36, 38} . No wild outliers, just values chilling around the mean and median. This is platykurtic—nice and calm.   While skewness shows if our data’s lopsided, kurtosis tells us if outliers are stealing the show.    Seeing Kurtosis in Action  Picture kurtosis with histograms (like we used for mean and skewness):   Leptokurtic : Sharp peak, chunky tails (lots of outliers).  Platykurtic : Flatter top, skinny tails (few outliers).  Mesokurtic : Classic bell curve, balanced tails.   Check out this chart to see the difference:   Kurtosis Comparison: The top subplot shows the full distributions, with leptokurtic (sharp peak, heavy tails with more outliers, like stock returns), mesokurtic (normal distribution, balanced tails), and platykurtic (flat peak, light tails with fewer outliers, like temperatures). The bottom subplot zooms in on the right tail, showing how leptokurtic tails decay slower (higher density at large values) compared to mesokurtic and platykurtic tails, which drop off faster.     import numpy as np import matplotlib.pyplot as plt from scipy.stats import kurtosis, t, norm, uniform # Data for kurtosis calculation stock_data = [5, 7, 8, 8, 10, 12, 12, 12, 12, 50] # Leptokurtic temp_data = [30, 32, 33, 34, 35, 36, 38] # Platykurtic # Calculate kurtosis kurt_stock = kurtosis(stock_data, fisher=True) kurt_temp = kurtosis(temp_data, fisher=True) print(f\"Stock Returns Kurtosis: {kurt_stock:.2f} (Leptokurtic)\") print(f\"Temperature Kurtosis: {kurt_temp:.2f} (Platykurtic)\") # Generate data for plotting distributions x = np.linspace(-10, 10, 200) # Wider range to show tails # Leptokurtic: Student's t-distribution (df=3 for heavy tails) lepto = t.pdf(x, df=3) * 1.2 # Scale for visibility # Mesokurtic: Normal distribution meso = norm.pdf(x) # Platykurtic: Uniform-like distribution (approximated) platy = uniform.pdf(x, loc=-2, scale=4) * 0.8 # Flat, light tails # Create figure with two subplots fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # Full distribution plot (top subplot) ax1.plot(x, lepto, label='Leptokurtic (Heavy Tails)', color='#FF5733') ax1.plot(x, meso, label='Mesokurtic (Normal)', color='#33FF57') ax1.plot(x, platy, label='Platykurtic (Light Tails)', color='#3357FF') ax1.set_title('Kurtosis Comparison: Full Distribution') ax1.set_xlabel('Value') ax1.set_ylabel('Density') ax1.legend() ax1.grid(True) # Tail-focused plot (bottom subplot) ax2.plot(x, lepto, label='Leptokurtic (Heavy Tails)', color='#FF5733') ax2.plot(x, meso, label='Mesokurtic (Normal)', color='#33FF57') ax2.plot(x, platy, label='Platykurtic (Light Tails)', color='#3357FF') ax2.set_title('Kurtosis Comparison: Right Tail Focus') ax2.set_xlabel('Value') ax2.set_ylabel('Density') ax2.set_xlim(3, 10) # Focus on right tail ax2.set_ylim(0, 0.05) # Zoom in on low density ax2.legend() ax2.grid(True) # Adjust layout and save plt.tight_layout() plt.savefig('kurtosis.png', dpi=300, bbox_inches='tight') # Save both subplots plt.show()     Calculating Kurtosis with Python  Since we’re all about descriptive stats, let’s compute kurtosis with Python (more tools coming up later). Here’s a script for our stock returns example:   from scipy.stats import kurtosis import numpy as np # Stock returns data (leptokurtic) data = [5, 7, 8, 8, 10, 12, 12, 12, 12, 50] # Calculate excess kurtosis kurt = kurtosis(data, fisher=True) print(f\"Excess Kurtosis: {kurt:.2f}\") # Positive value means leptokurtic (heavy tails)!   Run this, and you’ll see a positive kurtosis, confirming big outliers in our stock returns. Try the temperature data {30, 32, 33, 34, 35, 36, 38} for a negative kurtosis and that platykurtic vibe.    Why Kurtosis Matters  Kurtosis wraps up our descriptive stats crew—mean, median, mode, and skewness. It’s like a heads-up about outliers. High kurtosis (leptokurtic) yells “watch out for big swings!”—think risky stocks. Low kurtosis (platykurtic) says “smooth sailing”—like predictable weather. It’s another piece of your data’s story.      Conclusion  In summary, descriptive statistics provide essential tools for understanding and summarizing datasets, helping to reveal underlying patterns and trends. By examining measures of central tendency (mean, median, mode), we can identify the \"typical\" values of a dataset, while measures of dispersion (variance, standard deviation, range, and quartiles) show how spread out the data is. Visualizations like histograms, boxplots, and density plots enhance this understanding by allowing us to visually inspect the distribution and shape of the data. As we move forward, the next section will explore the tools available for calculating these descriptive statistics, giving us the means to automate and refine these analyses in practice.   "
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
  "body": " Computation and Visualization Tools   Exploratory Data Analysis (EDA) is a critical step in understanding your data before applying advanced techniques like machine learning. It involves summarizing the main characteristics of a dataset, often using visual methods, to uncover patterns, spot anomalies, test hypotheses, and check assumptions.  In this section, we focus on Python-based tools that enable efficient and effective data analysis, tailored for machine learning workflows. While languages like R are powerful for statistics, we emphasize Python due to its widespread use in data science and machine learning communities. Key tools include NumPy for numerical computations, Pandas for data manipulation, and Matplotlib\/Seaborn for visualization. These libraries integrate seamlessly, allowing you to load, clean, analyze, and visualize data in a streamlined manner.  A typical EDA workflow includes: loading data, inspecting its structure, handling missing values, computing summary statistics, exploring distributions, and visualizing relationships. Using Jupyter notebooks ensures reproducibility and documentation of your analysis.    The Power of NumPy and SciPy  While Python lists and loops are flexible, they are slow for large-scale numerical work. The NumPy library provides fast, memory-efficient arrays and vectorized operations. These make Python competitive with lower-level languages for scientific and machine learning tasks. The SciPy library builds on NumPy, adding advanced tools for statistics, optimization, and more.  Let us begin with a simple example: computing descriptive statisticsand plotting a histogram ( ) of rolls of a six-sided die. Notice that NumPy computes the mean and standard deviation in a single line. This would require explicit loops in plain Python.  import numpy as np import matplotlib.pyplot as plt #Set Seed for reproducibility np.random.seed(seed=42) # Generate 10,000 simulated die rolls rolls = np.random.randint(1, 7, size=50) mean = np.mean(rolls) std = np.std(rolls) print(f\"mean = {mean}, std = {std}\") #(np.float64(3.4999), np.float64(1.7086251753968744)) fig, ax = plt.subplots() ax.hist(rolls, bins=6, color=\"b\", alpha=0.25) plt.xlabel('Face Value') plt.ylabel('Frequency') plt.title('Histogram of Die Rolls') plt.savefig(\"np-die-histogram.png\") plt.show()   Histogram of simulated die rolls using NumPy.   Histogram of simulated die rolls using NumPy.    The real strength of NumPy comes from vectorization , which eliminates explicit loops.   import time N = 10_000_000 x = np.random.rand(N) # Vectorized: compute sum of squares t0 = time.time() s1 = np.sum(x**2) t1 = time.time() # Loop version s2 = 0.0 for xi in x: s2 += xi**2 t2 = time.time() (t1 - t0, t2 - t1) # compare runtimes   The vectorized NumPy version runs in milliseconds, while the loop can take seconds. This difference is crucial in machine learning, where datasets often have millions of entries.   Vectorized NumPy operation vs. Python loop runtime.    NumPy also includes a linalg package for linear algebra. The following code snippet demonstrates how NumPy solves systems of equations and computes eigenvalues—core operations in data science, physics, and engineering.   A = np.array([[3, 1], [1, 2]]) b = np.array([9, 8]) # Solve Ax = b x = np.linalg.solve(A, b) # Eigenvalues and eigenvectors e_vals, e_vecs = np.linalg.eig(A) (x, e_vals)]   For advanced tasks, SciPy extends NumPy. For example, hypothesis testing or optimization. SciPy provides one-line solutions for statistical inference and numerical optimization.   from scipy import stats, optimize # Hypothesis test: is sample mean = 0? sample = np.random.normal(0, 1, size=100) t_stat, p_val = stats.ttest_1samp(sample, 0) print(f\"t-statistic = {t_stat}, p-value={p_val}\") # t-statistic = 0.8998073723146639, p-value=0.37040629150553495    from scipy import stats, optimize # Optimization: minimize f(x) = (x-3)^2 f = lambda x: (x-3)**2 res = optimize.minimize(f, x0=0) # [2.99999998]   Together, NumPy and SciPy form the numerical backbone of Python’s scientific ecosystem.    Pandas: Data Manipulation and Analysis   Pandas is a powerful, flexible library for data manipulation and analysis, built on NumPy. Its core data structures are:  Series : A one-dimensional labeled array for sequences of data.  DataFrame : A two-dimensional labeled table, similar to a spreadsheet or SQL table, ideal for tabular data.    Pandas is designed for cleaning, transforming, analyzing, and visualizing data. It supports multiple file formats (CSV, Excel, JSON, SQL) and integrates with NumPy, Matplotlib, Seaborn, and Scikit-learn, making it a cornerstone for EDA in machine learning.    Why Use Pandas?   Handles structured data efficiently (e.g., tabular data).  Supports data cleaning (missing values, duplicates, outliers).  Enables grouping, aggregation, and statistical summaries.  Scales to large datasets with optimized performance.     EDA Workflow with Pandas:   Load data ( pd.read_csv() , pd.read_excel() ).  Inspect structure ( head() , info() , describe() ).  Clean data (handle missing values, remove duplicates).  Compute statistics and explore distributions.  Visualize (integrate with Matplotlib\/Seaborn).     Example: Analyzing Student Data Let’s use a realistic dataset of student scores, including a missing value, loaded from a CSV file.   Creating and loading sample student data  import pandas as pd # Create sample CSV data (in practice, load from disk) data = \"\"\"Name,Age,Score,Passed Alice,25,85.5,True Bob,30,90.0,True Carol,27,88.0,True Dave,22,76.5,False Eve,28,,True\"\"\" with open('students.csv', 'w') as f: f.write(data) # Load data df = pd.read_csv('students.csv') print(df)   Output as a table:   Student DataFrame    Name  Age  Score  Passed    Alice  25  85.5  True    Bob  30  90.0  True    Carol  27  88.0  True    Dave  22  76.5  False    Eve  28  NaN  True     Inspect the data using common Pandas methods:   Inspecting DataFrame  # Inspect data print(\"First 3 rows:\") print(df.head(3)) print(\"\\nLast 2 rows:\") print(df.tail(2)) print(\"\\nShape:\", df.shape) # (5, 4) print(\"\\nColumns:\", df.columns.tolist()) print(\"\\nInfo:\") print(df.info()) print(\"\\nDescriptive Statistics:\") print(df.describe())   Output of df.describe() :   Descriptive Statistics from df.describe()     Age  Score    count  5.000000  4.000000    mean  26.400000  85.000000    std  3.209361  5.958188    min  22.000000  76.500000    25%  24.250000  83.250000    50%  26.000000  86.750000    75%  27.750000  88.500000    max  30.000000  90.000000     Clean and transform the data (e.g., handle missing values, filter, add columns, group, sort):   Data cleaning and transformation  # Handle missing values print(\"Missing values:\\n\", df.isnull()) df['Score'] = df['Score'].fillna(df['Score'].mean()) # Fill NaN with mean # Filter rows high_scorers = df[df['Score'] > 85] print(\"\\nHigh scorers:\\n\", high_scorers) # Add new column df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C') print(\"\\nDataFrame with Grade:\\n\", df) # Group and aggregate grouped = df.groupby('Passed')['Score'].agg(['mean', 'count']) print(\"\\nGrouped by Passed:\\n\", grouped) # Sort by Score df_sorted = df.sort_values(by='Score', ascending=False) print(\"\\nSorted by Score:\\n\", df_sorted) # Chain operations result = df[df['Age'] > 25][['Name', 'Score']].sort_values(by='Score') print(\"\\nChained operations (Age > 25, select columns, sort):\\n\", result)   Visualize the score distribution:   Histogram of student scores using Pandas and Matplotlib.   Histogram from Pandas DataFrame.     Generating histogram from Pandas  import matplotlib.pyplot as plt import pandas as pd # Assuming df from previous code df['Score'].hist(bins=5, edgecolor='black', alpha=0.7) plt.xlabel('Score') plt.ylabel('Frequency') plt.title('Distribution of Student Scores') plt.grid(True, alpha=0.3) plt.savefig('pandas-histogram.png', dpi=300) plt.show()   For further learning, explore Python for Data Analysis by Wes McKinney (free online) and Kaggle’s Pandas course .    Visualization with Matplotlib and Seaborn  Visualization is a cornerstone of EDA, making patterns and relationships in data intuitive. Matplotlib provides customizable, low-level plotting, while Seaborn, built on Matplotlib, offers high-level statistical visualizations with attractive defaults.   Matplotlib Key Features:   Flexible plots: histograms, boxplots, scatter plots, line plots.  Customizable axes, labels, and styles.  Integration with Pandas for direct plotting.     Seaborn Advantages:   Statistical plots: histplot with KDE, boxplot, pairplot for correlations.  Attractive themes and color palettes.  Simplified syntax for complex visualizations.     Example: Visualizing Student Data Using the student DataFrame, create a histogram and boxplot with Matplotlib, and a histplot with KDE and pairplot with Seaborn.   Matplotlib histogram and boxplot of student scores.   Matplotlib plots from Pandas.     Matplotlib histogram and boxplot  import matplotlib.pyplot as plt import pandas as pd # Assuming df from previous code fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) df['Score'].hist(bins=5, ax=ax1, edgecolor='black', alpha=0.7) ax1.set_title('Histogram of Scores') ax1.set_xlabel('Score') ax1.set_ylabel('Frequency') ax1.grid(True, alpha=0.3) df.boxplot(column='Score', ax=ax2) ax2.set_title('Boxplot of Scores') ax2.set_ylabel('Score') ax2.grid(True, alpha=0.3) plt.tight_layout() plt.savefig('matplotlib-plots.png', dpi=300) plt.show()    Real-World Example: Load a larger dataset (e.g., from Kaggle) and visualize distributions and correlations.   EDA with a larger dataset  import pandas as pd import seaborn as sns import matplotlib.pyplot as plt # Sample larger dataset (simulated for book) np.random.seed(42) n = 100 data = pd.DataFrame({ 'Age': np.random.normal(25, 5, n), 'Score': np.random.normal(85, 10, n), 'Hours_Studied': np.random.normal(20, 5, n) }) data['Score'] = data['Score'].clip(0, 100) # Ensure valid scores # Basic EDA print(data.describe()) print(\"\\nMissing values:\\n\", data.isnull().sum()) # Correlation matrix print(\"\\nCorrelation matrix:\\n\", data.corr()) # Visualization plt.figure(figsize=(10, 4)) plt.subplot(1, 2, 1) sns.histplot(data['Score'], kde=True) plt.title('Distribution of Scores') plt.subplot(1, 2, 2) sns.scatterplot(x='Hours_Studied', y='Score', data=data) plt.title('Score vs. Hours Studied') plt.tight_layout() plt.savefig('.\/images\/essential-probability-and-statistics\/eda-large-dataset.png', dpi=300) plt.show() # Pairplot sns.pairplot(data, diag_kind='kde') plt.savefig('.\/images\/essential-probability-and-statistics\/eda-pairplot.png', dpi=300) plt.show()    EDA on a larger dataset: histogram and scatter plot.   EDA visualizations for larger dataset.     Pairplot showing relationships in larger dataset.   Pairplot for larger dataset.      NumPy, Pandas, Matplotlib, and Seaborn form a powerful toolkit for EDA. Start with NumPy for numerical operations, use Pandas for data manipulation and cleaning, and leverage Matplotlib\/Seaborn for insightful visualizations. Practice with real datasets (e.g., from Kaggle) in Jupyter notebooks to build skills. For advanced machine learning pipelines, you can explore TensorFlow’s Data API later, but mastering these foundational tools is key for beginners. Resources like Python for Data Analysis and Kaggle’s Pandas course offer hands-on learning.   "
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
  "body": " Numerical and Categorical Data   When analyzing data, it is essential to understand the type of data you are working with and how to safely convert it into a numerical representation for processing by machine learning models. There are three fundamental types:  Categorical Data : Data that represents discrete groups or labels with no inherent order.  Ordinal Data : A type of categorical data with a defined, meaningful order.  Numerical Data : Data that represents quantities and can be measured and compared numerically.       Categorical Data and One-Hot Encoding   Categorical Data represents discrete groups or labels with no inherent order between the values. For example, a variable named \"Color\" can take values from {\"red\", \"blue\", \"green\"}. Another variable may be \"Animal\" with values from {\"cat\", \"dog\", \"parrot\"}. In Python's Pandas library, you can store this as an object type or, for efficiency and clarity, convert it to a category type.   import pandas as pd colors = pd.Series([\"red\", \"blue\", \"red\", \"green\"], dtype=\"category\") print(colors)   0 red 1 blue 2 red 3 green dtype: category Categories (3, object): ['blue', 'green', 'red']  Many machine learning algorithms require a numerical representation of these values. A common method for this is one-hot encoding .  Suppose we have 3 unique colors. We represent each value with a 3-dimensional vector with a \"1\" in one position and \"0\" in the others:    Example of One-Hot Encoding     Sample 1  Sample 2  Sample 3  Sample 4    Original Data  red  blue  red  green    Color_red  1  0  1  0    Color_blue  0  1  0  0    Color_green  0  0  0  1     The main limitation of one-hot encoding is that the number of dimensions equals the number of categories. If there are thousands of unique categories (like words in a language), this becomes inefficient. In such cases, embeddings are preferred.     Ordinal Data and Safe Encoding   Ordinal Data is a type of categorical data with a defined order. For example, a variable for clothing size may have the order small < medium < large, or a satisfaction rating may be low < medium < high. The order matters, but the numeric difference between categories is not meaningful.   sizes = pd.Series([\"medium\", \"small\", \"large\", \"small\"], dtype=pd.CategoricalDtype(categories=[\"small\", \"medium\", \"large\"], ordered=True)) print(sizes)   0 medium 1 small 2 large 3 small dtype: category Categories (3, object): ['small' < 'medium' < 'large']  If you map ordinal values to integers naively, you risk misleading the model. For instance, \"High School\" → 1, \"Bachelor's\" → 2, \"Master's\" → 3, \"PhD\" → 4 implies equal numeric gaps, which isn’t true in reality.  Safer strategies:   Integer Encoding : Map to integers, but use with models (like decision trees) that care about order, not differences.  Binning : Collapse many levels into broader, meaningful groups.  Embeddings : In deep learning, treat them like tokens and let the model learn relationships.  Avoid One-Hot : It removes ordering information completely.      Numerical Data: Discrete vs Continuous   Numerical Data represents measurable quantities. These can be:   Discrete : Countable items (e.g., number of rooms in a house: 3, 4, 5).  Continuous : Measurable values with potentially infinite precision (e.g., height, weight, price).      Data Type Summary  This table summarizes the properties and common encoding methods for categorical, ordinal, and numerical data.   Comparison of Data Types    Type  Meaningful Order  Meaningful Interval  Encoding Method  Examples    Categorical  No  No  One-Hot Encoding  Colors, Animals, Cities    Ordinal  Yes  No  Integer Encoding, Binning  Ratings, Clothing Sizes    Numerical  Yes  Yes  Scaling \/ Normalization  Height, Weight, Price      "
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
  "body": " Basic Probability for Machine Learning   Probability is the backbone of machine learning, helping us model uncertainty in data, predictions, and outcomes. In machine learning, probability underpins tasks like classification (e.g., predicting labels), evaluating model confidence, and handling noisy data. This section introduces probability concepts such as sample spaces, events, and axioms—and connects them to practical machine learning applications using Python.  An event is a specific outcome or set of outcomes from an experiment, represented as a set. For a coin toss, \"heads\" is , \"tails\" is , and \"heads or tails\" is . Each trial answers whether an event occurred (yes\/no). For a die roll yielding , events like or occur if they include . Sets allow combining events via union ( ) or intersection ( ), such as .    Axiomatic View of Probability  In 1933, Andrey Kolmogorov formalized probability with three axioms, providing a mathematical framework. Think of these as rules that ensure probabilities make sense, like ensuring a weather forecast never predicts negative rain or more than chance.   Sample Space : The set of all possible outcomes. For a six-sided die, . For a student passing an exam, .   Event Space : All possible subsets of , including the empty set (impossible event) and (event certain to happen). For a coin toss ( ), . With outcomes, has events.  With every event we can identify its complementary event or complement . The complementary event includes all other possibilities that excluded the event . Thus, in a six-sided die, say . Its complement will be the event . Clearly, their intersection will be the null event.    Probability Measure : Assigns a number to each event , representing its likelihood. For example, for a fair die, . You only need for elementary events, which are the elements of , each taken as one event since you can use the Aditivity law below to get probability of any event in the entire event space .  The probability space is the triplet . Kolmogorov’s axioms are:  Non-negativity : Although, negative probabilities may be taunting science fiction scenarios, they do not make sense in the probailities we deal with everyday. We require that   Normalization : Since the event has all possible elementary events, except the null event, every possible event is inculded in . Therefor, is called event of certainty. Thus, , ensures total certainty for event . This is why although frequencies are proportional to probabilities, we need to divide them by total number of trials to convert them into normalized probabilities.  Additivity : If two events that are disjoint, i.e., there is no situation in which both events can occur together, i.e, ), the probability of either of them occuring, i.e., their union, must be sum of the probabilities of the events occuring separately. . Of course, if there was an overlap, between the two events, then, we would need to subtrat the overlap part since that would have been counted twice, once in and another time in .     Derived results:    illustrated the union formula in which two events (odd numbers), have an overlap. Theoretically, Let's see how it plays out in the rolls simulation shown in . The left region counts rolls of (in only), the right region counts rolls of (in only), the overlap counts rolls of (in ), and a total of counts for . From these counts we estimate probabilities. The counts , , and in total rolls . From these counts we get Now let's check    Venn diagram illustrating events (odd numbers) and for 1,000 simulated rolls of a fair six-sided die. The code of the simulation is given below.   Venn diagram of intersecting events.     Simulating die roll for union probability  # --- DIE ROLL VENN DIAGRAM --- import numpy as np from matplotlib_venn import venn2 import matplotlib.pyplot as plt np.random.seed(42) n_trials = 1000 rolls = np.random.randint(1, 7, n_trials) # Events e1 = np.isin(rolls, [1, 3, 5]) # Odd numbers e2 = np.isin(rolls, [3, 5, 6]) # 3,5,6 e1_only = np.sum(e1 \\amp; ~e2) e2_only = np.sum(e2 \\amp; ~e1) both = np.sum(e1 \\amp; e2) # Venn diagram plt.figure(figsize=(6, 4)) venn2(subsets=(e1_only, e2_only, both), set_labels=('E1 (Odd)', 'E2 (3,5,6)')) plt.title('Venn Diagram of Die Roll Events') plt.savefig('venn-diagram-E1-E2.png', dpi=300) plt.show() # Probabilities p_e1 = np.mean(e1) p_e2 = np.mean(e2) p_inter = np.mean(e1 \\amp; e2) p_union = np.mean(e1 | e2) print(f\"P(E1): {p_e1:.3f}, P(E2): {p_e2:.3f}, P(E1 ∩ E2): {p_inter:.3f}, P(E1 ∪ E2): {p_union:.3f}\") # --- END CODE ---     Sum and Product Rules for Probability  The sum and product rules are two cornerstones of probability theory. They allow us to compute probabilities of unions and intersections of events, which are essential in many machine learning applications, from estimating marginal distributions to building classifiers.   Sum Rule : The probability of the union of two events and is given by: The subtraction of avoids double-counting the overlap between the two events. This can be clearly understood using a Venn diagram.   Illustration of the sum rule. The probability of is the sum of the shaded areas of and , minus the overlapping region that would otherwise be counted twice.   Venn diagram showing the sum rule for two overlapping events.     Product Rule : The joint probability of two events and is: where is the conditional probability of given . This factorization is the basis for probabilistic models such as Naive Bayes.    Conditional Probability and Independence   Conditional Probability : The probability of an event given that event has occurred is written as . Intuitively, this represents updating our belief about once we know that is true. For instance, the probability that a student passes an exam may be different depending on whether we already know they studied more than 20 hours.  Probability of King if the Card is known to be Spade  Imagine we have a standard deck of 52 playing cards. Event = \"the card is a King\". Event = \"the card is a Spade\".  The unconditional probability of drawing a King is . But suppose we are told that the card drawn is a Spade. Now, the sample space is only 13 Spade cards. Out of these, exactly one is a King (the King of Spades).  Thus the conditional probability is , which differs from . This illustrates how new information (that the card is a Spade) changes the probability of .    Independence : Two events and are independent if the occurrence of one does not affect the probability of the other. Formally, the eventa and are independent if . Equivalently, Independence means that knowing whether occurred does not provide any new information about .  Tosses of a fair coin are independent  Suppose we toss a fair coin twice. Let event be \"the first toss is Heads\" and event be \"the second toss is Heads.\" The sample space is .  We have , , and . Therefore, , which means the events are independent.  Now let event be \"at least one toss is Heads.\" Then , , and . But , so . Thus, and are not independent.    Example (Student Study Data):   Suppose we record whether students studied more than 20 hours (High Study) and whether they passed an exam (Pass). Using the dataset below, we will estimate probabilities empirically:    : overall fraction of students who passed,     : fraction who studied more than 20 hours,     : fraction who both passed and studied more than 20 hours.     : Just look at the students who studied more than 20 hours (High Study), what fraction Passed the exam (Pass).   The sum rule gives . The product rule verifies that .   Sum and product rules with student data  import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns np.random.seed(42) data = pd.DataFrame({ \"Hours_Studied\": np.random.normal(20, 5, 200).clip(0, 40), \"Passed\": np.random.binomial(1, 0.7, 200) }) data[\"High_Study\"] = data[\"Hours_Studied\"] > 20 # Probabilities p_pass = data[\"Passed\"].mean() p_high = data[\"High_Study\"].mean() p_both = ((data[\"Passed\"] == 1) \\amp; (data[\"High_Study\"] == True)).mean() p_union = p_pass + p_high - p_both p_conditional = p_both \/ p_high print(f\"P(Pass) = {p_pass:.3f}\") print(f\"P(High Study) = {p_high:.3f}\") print(f\"P(Pass ∩ High Study) = {p_both:.3f}\") print(f\"P(Pass ∪ High Study) = {p_union:.3f}\") print(f\"P(Pass|High Study) * P(High Study) = {p_conditional:.3f} * {p_high:.3f} = {p_conditional * p_high:.3f}\") # Visualize joint distribution joint_table = pd.crosstab(data[\"Passed\"], data[\"High_Study\"], normalize=\"all\") sns.heatmap(joint_table, annot=True, cmap=\"Blues\", fmt=\".3f\") plt.xlabel(\"High Study Hours (>20)\") plt.ylabel(\"Pass (0=No, 1=Yes)\") plt.title(\"Joint Probability of Pass and High Study Hours\") plt.savefig(\"joint-probability-heatmap.png\", dpi=300) plt.show()    Joint probability table of passing and high study hours. Each cell represents . Row and column sums recover marginals ( and ), illustrating the sum rule. The product rule is verified by comparing the joint probability with .   Heatmap of joint probabilities for pass and study hours.    Let's see how we can read the heatmap in . I will read by the rows first. Let us do notation Then the joint probabilities in the heatmap are: First we will check that the total probaility is actually . Great! Now, let us the probability of Passing the exam, whether you studied or not. This is asking for probability of regardless of the values; So, we need to sum ovr the values. That's pretty high chance of passing. The chance of not passing will just be it's complement. Now, what is the chance that a random student has actullay studied hard regardless of whatever happened in the test. That will be , which we can get from the joint probabilities by summing over the values while keeping the values to . That would mean the probability of a random student having not studied excessively is Now, how about the conditional probabilities? From this heatmap, it is easy to get conditional probabilities. For instance, suppose we want to know Had you studied more than 20 hrs , wat would be your change of passing  Wow, this data produced by simulation showing us that had you studied excessively, your chance of passing the exam actually went down from to . Wild! Okay, now your turn of computing sme other conditional probabilities.    Probability Distributions  Probability distributions describe how probabilities are distributed over outcomes. In machine learning, distributions model data or predictions.   Bernoulli Distribution : Models a binary outcome (e.g., pass\/fail) with probability . For passing an exam, , .   Binomial Distribution : Counts successes in independent Bernoulli trials. For 10 students, the number who pass follows a binomial distribution.   Binomial distribution for student passes  # --- BINOMIAL DISTRIBUTION --- import numpy as np import matplotlib.pyplot as plt from scipy.stats import binom n, p = 10, 0.7 # 10 students, P(Pass) = 0.7 k = np.arange(0, 11) pmf = binom.pmf(k, n, p) plt.bar(k, pmf) plt.xlabel('Number of Passes') plt.ylabel('Probability') plt.title('Binomial Distribution (n=10, p=0.7)') plt.grid(True, alpha=0.3) plt.savefig('.\/images\/essential-probability-and-statistics\/binomial-dist.png', dpi=300) plt.show() # --- END CODE ---    Bar plot of the binomial probability mass function (PMF) for the number of students passing an exam out of 10, with a pass probability . Each bar represents the probability of students passing, calculated as . The peak around 7 passes reflects the high likelihood of most students passing given . This distribution is critical in machine learning for modeling binary outcomes, such as predicting the number of successful predictions in a classification task.   Bar plot of binomial distribution.      Three Types of Probabilities  Probability can be approached theoretically, empirically (frequentist), or subjectively (Bayesian).     Theoretical Probability : Uses symmetry. For a fair die, . For even numbers, .     Frequentist Probability : Estimates probability from trial frequencies: .   Frequentist estimation for fair and biased dice  # --- FREQUENTIST SIMULATION --- import numpy as np import matplotlib.pyplot as plt np.random.seed(42) n_trials = 1000 fair_rolls = np.random.randint(1, 7, n_trials) biased_rolls = np.random.choice([1, 2, 3, 4, 5, 6], n_trials, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]) # Cumulative probabilities cum_fair = np.cumsum(fair_rolls == 1) \/ np.arange(1, n_trials + 1) cum_biased = np.cumsum(biased_rolls == 1) \/ np.arange(1, n_trials + 1) plt.plot(cum_fair, label='Fair Die (P=1\/6)') plt.plot(cum_biased, label='Biased Die (P=0.2)') plt.axhline(1\/6, color='red', linestyle='--', label='Theoretical P=1\/6') plt.xlabel('Trials') plt.ylabel('Estimated P(1)') plt.title('Frequentist Estimates: Fair vs. Biased Die') plt.legend() plt.grid(True, alpha=0.3) plt.savefig('frequentist-convergence.png', dpi=300) plt.show() # --- END CODE ---    Plot showing the convergence of frequentist probability estimates for rolling a 1 on a fair die ( ) and a biased die ( ) over 1,000 trials. The fair die’s estimate (blue) fluctuates but approaches 1\/6 (red dashed line), while the biased die’s estimate (orange) converges to 0.2, reflecting the higher probability of rolling a 1. This visualization demonstrates how empirical frequencies approximate true probabilities in large samples, a technique used in machine learning to estimate probabilities from training data.   Convergence plot for frequentist estimates.       Bayesian Probability . This is also an empirical definition of probability. But, rather than give you one number for probability of an event, Bayesian gives you a probability distribution of the values of probability of the event. From that, you can work out the mean value, which you can use as one value for the probability of the event.  It is based on incorporating belief about the probability of an outcome BEFORE we even conduct the experiment and then updated this so-called prior assumption or bias with what we observe in the experiment. The updated belief is the posterior, and improved value of the probability.  Clearly, as we repeat the experiment infinitely many times, the effect of our initial belief would disappear and the answer will match the results of the frequentists' experiments. However, since we can never do infinite number of trials, the Bayesian gives an edge in cases where we have some information about the outcome even before we start the trials.   Example: This example is a little bit ahead of my presentation here as it requres a little bit of math to properly express how th Bayesian probability works. If you feel up to it, you can ahead and read on, but it's okay to skip it for now.  In the case of a six-sided die, suppose we want to estimate the probability for one-dot face up as we illustrated in the frequentist case above. First, we would need to choose a prior belief, i.e., a probability distribution for , i.e., how likely is any value of between its range of values, which will be from to , inclusive, . Since, we do not know which value is right, we might decide that it could be 1\/2 times it will be face up and 1\/2 of the time it will be not face up (I know a fair die will be 1\/6 times face up, but I want to show you how even a very off prior will eventually converge to the proper value). In such cases and our trial each time being either face up true or false ( which is a case of Bernoulli trials ), it is traditional to choose a beta distribution, which has two parameters and , with and . Using symbol for and , probability density of , we will write this as follows where . where is beta function. The mean value of beta distribution is an important result and can be easily found. Here, I have introduced physicists' notation for the mean of a quantity, . Thus, by choosing as the prior distribution, we are assuming that somehow we suspect that is close to . So, we are basically, starting way off in our belief.  Just a side math info: Beta function is usually written in terms of factorial or Gamma function. where, for integer arguments , and in general, To hide all the mathematical details in our work below, we will, as is normally done, just express the probability by a simpler notation and represent Eq.  where instead of lower case variable name , we use the notation of upper case .  Let's get back to our rolling experiment and see how our belief of the true value of evolves with each roll's result. Suppose we roll the die and observe that the up face is not one, then without showing you the calculations here, which will be done later in the chapter, we use Bayes rule, to be discussed later, to show that the probability distribution now shifts to . How did we go from distribution to ? I used Bayesian theorem. We will not show the calculation here but differ to a later section.  Toss second time, let's say the result is a one. Then our belief will be update with this new data to . Toss again, say no one. We keep updating the probability distribution of . At any point, we can take the expectation value of the the variable in the current distribution to give us the \"best current value\" for . Thus, after three trials above, we will say that .  Suppose you continued rolling and you had the following next 7 trials: After these 10 trials in total, the distribution will be This is still far away from that you would expect from a fair die, but you don't know if the die was fair. So, empirical results are all you have to go by.  For the same rolling results, frequentists' probability will give us the following estimate:   They look similar. But, had you expected the die was fair, you would start with a better prior, with say . Then the 10 trials would update to It would have revealed if the die was not a fair die. It's either not a fair die or we have rolled it too few times.      Conclusion  Probability provides the foundation for modeling uncertainty in machine learning. Axioms define the rules, while theoretical, frequentist, and Bayesian approaches offer different perspectives. Conditional probability and distributions are used in machine learning models to make predictions.   "
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
