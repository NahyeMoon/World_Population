# Introduction

This project analyzes a world population [dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/world-population-dataset) of population of 234 countries or regions from 1970 to 2022. In this analysis, python's Pandas library and PowerBI were used to visualize key trends and insights from the dataset. The project begins by addressing null and unique values in each column then exploring the distribution of populations across countries, using summary statistics and boxplots to identify disparities in global population density. Further, the population growth of the 10 most populous countries is visualized and analyzed, offering how the nations' socioeconomic statuses shaped the population trends. A polynomial regression model was built and used to predict the population of India and China in 2030, 2040, and 2050, testing the accuracy of past predictions. Lastly, the project concludes by examining the total population growth of each continent and providing insights from a socioeconomic standpoint to better understand the global demographic shifts.

<br/>

# Data Information

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
import numpy as np
from PIL import Image
```

<br/>
The dataset contains country name and code, capital, continent and populations (1970, 1980, 1990, 2000, 2010, 2015, 2020 and 2022), rank by population, area, population density, growth rate and world population percentage.
<br/><br/>

```python
df = pd.read_csv(r"C:\Users\Nahye\OneDrive - University of Toronto\Documents\Project\World Population\world_population.csv")
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>CCA3</th>
      <th>Country</th>
      <th>Capital</th>
      <th>Continent</th>
      <th>2022 Population</th>
      <th>2020 Population</th>
      <th>2015 Population</th>
      <th>2010 Population</th>
      <th>2000 Population</th>
      <th>1990 Population</th>
      <th>1980 Population</th>
      <th>1970 Population</th>
      <th>Area (km²)</th>
      <th>Density (per km²)</th>
      <th>Growth Rate</th>
      <th>World Population Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36</td>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>Kabul</td>
      <td>Asia</td>
      <td>41128771.0</td>
      <td>38972230.0</td>
      <td>33753499.0</td>
      <td>28189672.0</td>
      <td>19542982.0</td>
      <td>10694796.0</td>
      <td>12486631.0</td>
      <td>10752971.0</td>
      <td>652230.0</td>
      <td>63.0587</td>
      <td>1.0257</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Tirana</td>
      <td>Europe</td>
      <td>2842321.0</td>
      <td>2866849.0</td>
      <td>2882481.0</td>
      <td>2913399.0</td>
      <td>3182021.0</td>
      <td>3295066.0</td>
      <td>2941651.0</td>
      <td>2324731.0</td>
      <td>28748.0</td>
      <td>98.8702</td>
      <td>0.9957</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>DZA</td>
      <td>Algeria</td>
      <td>Algiers</td>
      <td>Africa</td>
      <td>44903225.0</td>
      <td>43451666.0</td>
      <td>39543154.0</td>
      <td>35856344.0</td>
      <td>30774621.0</td>
      <td>25518074.0</td>
      <td>18739378.0</td>
      <td>13795915.0</td>
      <td>2381741.0</td>
      <td>18.8531</td>
      <td>1.0164</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>213</td>
      <td>ASM</td>
      <td>American Samoa</td>
      <td>Pago Pago</td>
      <td>Oceania</td>
      <td>44273.0</td>
      <td>46189.0</td>
      <td>51368.0</td>
      <td>54849.0</td>
      <td>58230.0</td>
      <td>47818.0</td>
      <td>32886.0</td>
      <td>27075.0</td>
      <td>199.0</td>
      <td>222.4774</td>
      <td>0.9831</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>203</td>
      <td>AND</td>
      <td>Andorra</td>
      <td>Andorra la Vella</td>
      <td>Europe</td>
      <td>79824.0</td>
      <td>77700.0</td>
      <td>71746.0</td>
      <td>71519.0</td>
      <td>66097.0</td>
      <td>53569.0</td>
      <td>35611.0</td>
      <td>19860.0</td>
      <td>468.0</td>
      <td>170.5641</td>
      <td>1.0100</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>226</td>
      <td>WLF</td>
      <td>Wallis and Futuna</td>
      <td>Mata-Utu</td>
      <td>Oceania</td>
      <td>11572.0</td>
      <td>11655.0</td>
      <td>12182.0</td>
      <td>13142.0</td>
      <td>14723.0</td>
      <td>13454.0</td>
      <td>11315.0</td>
      <td>9377.0</td>
      <td>142.0</td>
      <td>81.4930</td>
      <td>0.9953</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>230</th>
      <td>172</td>
      <td>ESH</td>
      <td>Western Sahara</td>
      <td>El Aaiún</td>
      <td>Africa</td>
      <td>575986.0</td>
      <td>556048.0</td>
      <td>491824.0</td>
      <td>413296.0</td>
      <td>270375.0</td>
      <td>178529.0</td>
      <td>116775.0</td>
      <td>76371.0</td>
      <td>266000.0</td>
      <td>2.1654</td>
      <td>1.0184</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>231</th>
      <td>46</td>
      <td>YEM</td>
      <td>Yemen</td>
      <td>Sanaa</td>
      <td>Asia</td>
      <td>33696614.0</td>
      <td>32284046.0</td>
      <td>28516545.0</td>
      <td>24743946.0</td>
      <td>18628700.0</td>
      <td>13375121.0</td>
      <td>9204938.0</td>
      <td>6843607.0</td>
      <td>527968.0</td>
      <td>63.8232</td>
      <td>1.0217</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>232</th>
      <td>63</td>
      <td>ZMB</td>
      <td>Zambia</td>
      <td>Lusaka</td>
      <td>Africa</td>
      <td>20017675.0</td>
      <td>18927715.0</td>
      <td>NaN</td>
      <td>13792086.0</td>
      <td>9891136.0</td>
      <td>7686401.0</td>
      <td>5720438.0</td>
      <td>4281671.0</td>
      <td>752612.0</td>
      <td>26.5976</td>
      <td>1.0280</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>233</th>
      <td>74</td>
      <td>ZWE</td>
      <td>Zimbabwe</td>
      <td>Harare</td>
      <td>Africa</td>
      <td>16320537.0</td>
      <td>15669666.0</td>
      <td>14154937.0</td>
      <td>12839771.0</td>
      <td>11834676.0</td>
      <td>10113893.0</td>
      <td>7049926.0</td>
      <td>5202918.0</td>
      <td>390757.0</td>
      <td>41.7665</td>
      <td>1.0204</td>
      <td>0.20</td>
    </tr>
  </tbody>
</table>
<p>234 rows × 17 columns</p>
</div>


<br/><br/>
The dataset is mostly complete, with only a few missing values scattered across some columns. Key columns such as Rank, Country, and Continent contain no missing values, while other columns have only minor data gaps. To ensure accuracy and preserve the original data integrity, these missing values were neither filled nor replaced.
<br/><br/>

```python
# Get the info of df
# # of cols, non-null entries and data type of each col
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 234 entries, 0 to 233
    Data columns (total 17 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   Rank                         234 non-null    int64  
     1   CCA3                         234 non-null    object 
     2   Country                      234 non-null    object 
     3   Capital                      234 non-null    object 
     4   Continent                    234 non-null    object 
     5   2022 Population              230 non-null    float64
     6   2020 Population              233 non-null    float64
     7   2015 Population              230 non-null    float64
     8   2010 Population              227 non-null    float64
     9   2000 Population              227 non-null    float64
     10  1990 Population              229 non-null    float64
     11  1980 Population              229 non-null    float64
     12  1970 Population              230 non-null    float64
     13  Area (km²)                   232 non-null    float64
     14  Density (per km²)            230 non-null    float64
     15  Growth Rate                  232 non-null    float64
     16  World Population Percentage  234 non-null    float64
    dtypes: float64(12), int64(1), object(4)
    memory usage: 31.2+ KB
    


```python
# Number of nulls in each column
df.isnull().sum()
```




    Rank                           0
    CCA3                           0
    Country                        0
    Capital                        0
    Continent                      0
    2022 Population                4
    2020 Population                1
    2015 Population                4
    2010 Population                7
    2000 Population                7
    1990 Population                5
    1980 Population                5
    1970 Population                4
    Area (km²)                     2
    Density (per km²)              4
    Growth Rate                    2
    World Population Percentage    0
    dtype: int64


<br/><br/>
The number of unique values in each column is shown below. The columns Rank, CCA3, Country, and Capital only had unique values and the Continent had 6 different values as they should. The populations in each year recorded, area and area and density were mostly unique with less than 5 non-unique values. However, Growth Rate and World Population Percentage have less unique values than those since those values tend to be small and have small variance.
<br/><br/>

```python
# Number of unique values in each column
df.nunique()
```




    Rank                           234
    CCA3                           234
    Country                        234
    Capital                        234
    Continent                        6
    2022 Population                230
    2020 Population                233
    2015 Population                230
    2010 Population                227
    2000 Population                227
    1990 Population                229
    1980 Population                229
    1970 Population                230
    Area (km²)                     231
    Density (per km²)              230
    Growth Rate                    178
    World Population Percentage     70
    dtype: int64

<br/>

# Data Exploration and Analysis

## Summary Statistic 
According to the summary statistic of each column, Growth Rate ranged from 0.91 to 1.07 with standard deviation 0.01, which is very small compared to standard deivations of populations that go up to tens or hundreds of million. World Population Percentage ranges from 0.00 to 17.88 with a higher standard deviation of 1.71. However, 25%, 50% and 75% percentile of 0.01, 0.07 and 0.28 indicate that the most countries have a small World Population Percentage below 0.28. The boxplot not only supports those observations but visualizes the distributions of all values in each column.


```python
# Format the numbers to two decimal places and place commas every three digits for better legibility.
pd.set_option('display.float_format', lambda x: f"{x:,.2f}")
```


```python
# statistic of each col.
df.describe()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>2022 Population</th>
      <th>2020 Population</th>
      <th>2015 Population</th>
      <th>2010 Population</th>
      <th>2000 Population</th>
      <th>1990 Population</th>
      <th>1980 Population</th>
      <th>1970 Population</th>
      <th>Area (km²)</th>
      <th>Density (per km²)</th>
      <th>Growth Rate</th>
      <th>World Population Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>234.00</td>
      <td>230.00</td>
      <td>233.00</td>
      <td>230.00</td>
      <td>227.00</td>
      <td>227.00</td>
      <td>229.00</td>
      <td>229.00</td>
      <td>230.00</td>
      <td>232.00</td>
      <td>230.00</td>
      <td>232.00</td>
      <td>234.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>117.50</td>
      <td>34,632,250.88</td>
      <td>33,600,710.95</td>
      <td>32,066,004.16</td>
      <td>30,270,164.48</td>
      <td>26,840,495.26</td>
      <td>19,330,463.93</td>
      <td>16,282,884.78</td>
      <td>15,866,499.13</td>
      <td>581,663.75</td>
      <td>456.81</td>
      <td>1.01</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>std</th>
      <td>67.69</td>
      <td>137,889,172.44</td>
      <td>135,873,196.61</td>
      <td>131,507,146.34</td>
      <td>126,074,183.54</td>
      <td>113,352,454.57</td>
      <td>81,309,624.96</td>
      <td>69,345,465.54</td>
      <td>68,355,859.75</td>
      <td>1,769,133.06</td>
      <td>2,083.74</td>
      <td>0.01</td>
      <td>1.71</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>510.00</td>
      <td>520.00</td>
      <td>564.00</td>
      <td>596.00</td>
      <td>651.00</td>
      <td>700.00</td>
      <td>733.00</td>
      <td>752.00</td>
      <td>1.00</td>
      <td>0.03</td>
      <td>0.91</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>59.25</td>
      <td>419,738.50</td>
      <td>406,471.00</td>
      <td>394,295.00</td>
      <td>382,726.50</td>
      <td>329,470.00</td>
      <td>261,928.00</td>
      <td>223,752.00</td>
      <td>145,880.50</td>
      <td>2,567.25</td>
      <td>36.60</td>
      <td>1.00</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>117.50</td>
      <td>5,762,857.00</td>
      <td>5,456,681.00</td>
      <td>5,244,415.00</td>
      <td>4,889,741.00</td>
      <td>4,491,202.00</td>
      <td>3,785,847.00</td>
      <td>3,135,123.00</td>
      <td>2,511,718.00</td>
      <td>77,141.00</td>
      <td>95.35</td>
      <td>1.01</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>175.75</td>
      <td>22,653,719.00</td>
      <td>21,522,626.00</td>
      <td>19,730,853.75</td>
      <td>16,825,852.50</td>
      <td>15,625,467.00</td>
      <td>11,882,762.00</td>
      <td>9,817,257.00</td>
      <td>8,817,329.00</td>
      <td>414,643.25</td>
      <td>236.88</td>
      <td>1.02</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>max</th>
      <td>234.00</td>
      <td>1,425,887,337.00</td>
      <td>1,424,929,781.00</td>
      <td>1,393,715,448.00</td>
      <td>1,348,191,368.00</td>
      <td>1,264,099,069.00</td>
      <td>1,153,704,252.00</td>
      <td>982,372,466.00</td>
      <td>822,534,450.00</td>
      <td>17,098,242.00</td>
      <td>23,172.27</td>
      <td>1.07</td>
      <td>17.88</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Use boxplot to view outliers and distributions
df.boxplot(figsize = (25, 10))
```





    
![png](output_13_1.png)

<br/>
The correlation heat map among columns with numeric datatypes show that the change in population of the countries was not statistically significant especially from 2010 to 2022 as the correlations between popualtions recorded in the time frame were all 1. Rank and population columns were negatively correlated as expected since greater the population the higher the ranking. The polulations and growth rate are negatively correlated by small numbers (> -0.01), and this is reasonable considering that the populations of populous countries can saturate over time.

```python
# Correlation between all numeric values
num_df = df.select_dtypes(include=['number'])
```

```python
# Visualizae the correlation using heat map
sns.heatmap(num_df.corr(), annot = True)

plt.rcParams['figure.figsize'] = (20, 7)

plt.show()
```
![png](output_heat.png)

<br/>


## Population Density 

The bubble map of global population density created on PowerBI is shown below, with the largest bubbles concentrated in Southeast Asia and Southern Europe, and this is supported by the table below listing the 10 most densely populated regions or countries.  5 of those regions or coutnries are in Asia, 3 in Europe, and 2 in the Caribbean region of North America. Notably, the top six regions—Macau, Monaco, Singapore, Hong Kong, Gibraltar, and Bahrain—share key characteristics. Despite their limited land areas, they serve as major financial and economic hubs. This concentration of wealth and business opportunities results in high living costs, an affluent population, and rapid urban development. Each of these regions also holds a unique political status and significant autonomy, especially Hong Kong and Macau, which operate as Special Administrative Regions of China with distinct economic systems.

In contrast, the lower 4 regions or countries - Maldives, Malta, Sint Maarten, Bermuda - are small islands known for their tourism sectors that capitalize on natural beauty and warm climates. Like financial hubs, these territories often have unique political statuses that grant them high autonomy, which aids their specialization in tourism or financial services. Bermuda, for example, is a British Overseas Territory with a high degree of self-governance, allowing it to implement policies that attract tourism.

These results suggest that global population density disparities could be reduced by creating new districts, territories, or cities aimed at attracting tourism or finance hubs in less densely populated regions. This approach would not only reduce pressures in densely populated areas but also foster economic growth and urban development in underdeveloped regions. This creates job opportunities, infrastructure, and increased standards of living for local populations, similar to how Singapore or Dubai developed through strategic positioning as financial and tourism centers ([Harvard International Revies](https://hir.harvard.edu/singapore-dubai-model-opportunities-for-expansion-in-africa-latin-america-and-beyond/#:~:text=Both%20countries%20have%20constructed%20major,passengers%20during%20the%20same%20period.)).


```python
bubble = Image.open("C:/Users/Nahye/OneDrive - University of Toronto/Documents/Project/World Population/bubble map.jpeg")

bubble
```




    
![png](output_15_0.png)
    




```python
df_density = df.sort_values(by = 'Density (per km²)', ascending = False).iloc[0:10]
df_density[['Rank', 'Country', 'Continent', '2022 Population',  'Growth Rate', 'World Population Percentage', 'Density (per km²)', 'Area (km²)']]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Country</th>
      <th>Continent</th>
      <th>2022 Population</th>
      <th>Growth Rate</th>
      <th>World Population Percentage</th>
      <th>Density (per km²)</th>
      <th>Area (km²)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>119</th>
      <td>167</td>
      <td>Macau</td>
      <td>Asia</td>
      <td>695,168.00</td>
      <td>1.01</td>
      <td>0.01</td>
      <td>23,172.27</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>134</th>
      <td>217</td>
      <td>Monaco</td>
      <td>Europe</td>
      <td>36,469.00</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>18,234.50</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>187</th>
      <td>113</td>
      <td>Singapore</td>
      <td>Asia</td>
      <td>5,975,689.00</td>
      <td>1.01</td>
      <td>0.07</td>
      <td>8,416.46</td>
      <td>710.00</td>
    </tr>
    <tr>
      <th>89</th>
      <td>104</td>
      <td>Hong Kong</td>
      <td>Asia</td>
      <td>7,488,865.00</td>
      <td>1.00</td>
      <td>0.09</td>
      <td>6,783.39</td>
      <td>1,104.00</td>
    </tr>
    <tr>
      <th>76</th>
      <td>219</td>
      <td>Gibraltar</td>
      <td>Europe</td>
      <td>32,649.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>5,441.50</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>154</td>
      <td>Bahrain</td>
      <td>Asia</td>
      <td>1,472,233.00</td>
      <td>1.01</td>
      <td>0.02</td>
      <td>1,924.49</td>
      <td>765.00</td>
    </tr>
    <tr>
      <th>123</th>
      <td>174</td>
      <td>Maldives</td>
      <td>Asia</td>
      <td>523,787.00</td>
      <td>1.00</td>
      <td>0.01</td>
      <td>1,745.96</td>
      <td>300.00</td>
    </tr>
    <tr>
      <th>125</th>
      <td>173</td>
      <td>Malta</td>
      <td>Europe</td>
      <td>533,286.00</td>
      <td>1.01</td>
      <td>0.01</td>
      <td>1,687.61</td>
      <td>316.00</td>
    </tr>
    <tr>
      <th>188</th>
      <td>214</td>
      <td>Sint Maarten</td>
      <td>North America</td>
      <td>44,175.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1,299.26</td>
      <td>34.00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>206</td>
      <td>Bermuda</td>
      <td>North America</td>
      <td>64,184.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1,188.59</td>
      <td>54.00</td>
    </tr>
  </tbody>
</table>
</div>


<br/>

## Most Populous Countries

The table below shows the 10 countries with the largest population in 2022 with five of them in Asia, 2 in North America 1 each in South America, Africa and Europe. The top two countries, China and India take up 17.88% and 17.77% of the world population, respectively and United States ranked third take up 4.24% which is significantly low compared to the other two. This explains the left-skewed distribution of values in World Population Percentage that was previously mentioned.


```python
df_2022 = df.sort_values(by = '2022 Population', ascending = False).iloc[0:10]
df_2022[['Rank', 'Country', 'Continent', '2022 Population',  'Growth Rate', 'World Population Percentage']]
# OR
# df.sort_values(by = '2022 Population', ascending = False).head(10)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Country</th>
      <th>Continent</th>
      <th>2022 Population</th>
      <th>Growth Rate</th>
      <th>World Population Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>China</td>
      <td>Asia</td>
      <td>1,425,887,337.00</td>
      <td>1.00</td>
      <td>17.88</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2</td>
      <td>India</td>
      <td>Asia</td>
      <td>1,417,173,173.00</td>
      <td>1.01</td>
      <td>17.77</td>
    </tr>
    <tr>
      <th>221</th>
      <td>3</td>
      <td>United States</td>
      <td>North America</td>
      <td>338,289,857.00</td>
      <td>1.00</td>
      <td>4.24</td>
    </tr>
    <tr>
      <th>93</th>
      <td>4</td>
      <td>Indonesia</td>
      <td>Asia</td>
      <td>275,501,339.00</td>
      <td>1.01</td>
      <td>3.45</td>
    </tr>
    <tr>
      <th>156</th>
      <td>5</td>
      <td>Pakistan</td>
      <td>Asia</td>
      <td>235,824,862.00</td>
      <td>1.02</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th>149</th>
      <td>6</td>
      <td>Nigeria</td>
      <td>Africa</td>
      <td>218,541,212.00</td>
      <td>1.02</td>
      <td>2.74</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7</td>
      <td>Brazil</td>
      <td>South America</td>
      <td>215,313,498.00</td>
      <td>1.00</td>
      <td>2.70</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8</td>
      <td>Bangladesh</td>
      <td>Asia</td>
      <td>171,186,372.00</td>
      <td>1.01</td>
      <td>2.15</td>
    </tr>
    <tr>
      <th>171</th>
      <td>9</td>
      <td>Russia</td>
      <td>Europe</td>
      <td>144,713,314.00</td>
      <td>1.00</td>
      <td>1.81</td>
    </tr>
    <tr>
      <th>131</th>
      <td>10</td>
      <td>Mexico</td>
      <td>North America</td>
      <td>127,504,125.00</td>
      <td>1.01</td>
      <td>1.60</td>
    </tr>
  </tbody>
</table>
</div>



<br/>

## Population Trend of China and India
The line graph below shows the population trends from 1970 to 2022 for the seven most populous countries in 2022. China and India experienced rapid population growth until around 2010, after which the growth began to slow. Similarly, the populations of the other five countries started to stabilize around this time. This trend is likely to be due to low fertility rates driven by urbanization, economic development, the high of living and education.

China and India have populations that far exceed those of other five countries in the graph, with little difference between them (1.426 billion and 1.417 billion, respectively, in 2022) and similar growth rates of 1.00 and 1.01. This supports many prior research few years ago suggesting that India’s population will eventually surpass China’s. According to the [UN](https://population.un.org/wpp/), India is indeed the most populous country in the world in 2024.

[UN](https://population.un.org/wpp/) also expects that India's population will increase to 1.515 billion by 2030 while China's population will decrease slightly to 1.416 billion by 2030. China’s one-child policy, introduced in 1979, directly contributed to its slower growth rate. Although the policy was adjusted to allow two children in 2015 and three in May 2021, China’s population is expected to continue declining due to the low fertility rates impacted by the poilcy and previously mentioned socioeconomic factors that developed countries are facing [(CNN)](https://www.cnn.com/2023/01/18/china/china-population-drop-explainer-intl-hnk/index.html). On the other hand, India's population is expected to grow despite of the drop in the fertility rate of 6 in 1960s to 2 in 2021 due to the considerable number of women entering their fertility years [(CNN)](https://www.cnn.com/2023/04/28/asia/india-population-overtakes-china-graphics-intl-hnk-dst-dg/index.html).


```python
population_columns = ['1970 Population', '1980 Population', '1990 Population', 
                      '2000 Population', '2010 Population', '2015 Population', 
                      '2020 Population', '2022 Population']

countries_of_interest = ['China', 'India', 'United States', 'Indonesia', 'Pakistan', 'Bangladesh', 'Brazil']
df_countries = df[df['Country'].isin(countries_of_interest)]

# Melt the dataframe to plot populations over time for selected countries
df_melted = df_countries.melt(id_vars=["Country"], value_vars=population_columns, 
                              var_name='Year', value_name='Population')

df_melted['Year'] = df_melted['Year'].str.split().str[0]

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Population', hue='Country', data=df_melted)
plt.title('Population Growth Over Time')
plt.show()
```


    
![png](output_23_0.png)
    

<br/>

## Population Prediction of China and India
A polynomial regression model with a degree of 3 was used to predict the populations of China and India in 2030, and its accuracy was evaluated. The two graphs below display the polynomial regression fit to the data and the predicted populations of the countries for 2030 based on the model. According to this simple model, which does not account for socioeconomic status or growth rates of each country, the predicted populations for 2030, 2040, and 2050 were approximately the same: 1.452 billion, 1.451 billion, and 1.417 billion, respectively. The model underestimated India's population in 2030 by 63 million (4%) and overestimated China's population by 36 million (3%), which are significant discrepancies from a demographic perspective. This clearly highlights the limitations of the polynomial regression model in predicting future populations, especially when relying solely on past data, particularly for large numbers. 

This also suggests that socioeconomic factors, such as income levels, education, urbanization rates, fertility rates, healthcare access, and government policies, are crucial in building a more robust and accurate population model. Incorporating these factors would enable the model to account for the dynamic and complex nature of population growth, offering a more realistic projection of future trends. Without considering such variables, the model's predictions remain overly simplistic and may not accurately reflect the true population trajectories, especially in populous countries with rapidly changing social and economic landscapes.


```python
# Prepare x and y variables for the polynomial regression model
years = [int(col.replace(' Population', '')) for col in population_columns]
years_linreg = [[year] for year in years]  # Reshape year data into a 2D array

```


```python
China_df = df[df['Country'] == 'China'][population_columns]
China_population = [[int(pop)] for pop in China_df.values.flatten()]

```


```python
# Population Prediction of China

poly_params = np.polyfit(years, China_population, 3)

# Generate predictions for all years for visualization
predicted_population = np.polyval(poly_params, years)

#Visualize the data and the polynomial fit
plt.scatter(years, China_population, label='Actual Data', color = 'red')
plt.plot(years, predicted_population, label='Polynomial Fit (degree 3)', color='blue')

# 2030 prediction
predicted_population_2030 = np.polyval(poly_params, 2030)
plt.scatter(2030, predicted_population_2030, color= 'red', marker = 'x', label=f'Prediction')
plt.plot([years[-1], 2030], [China_population[-1], predicted_population_2030], 'r--')

# 2040 prediction
predicted_population_2040 = np.polyval(poly_params, 2040)
plt.scatter(2040, predicted_population_2040, color= 'red', marker = 'x')
plt.plot([2030, 2040], [predicted_population_2030, predicted_population_2040], 'r--')

# 2050 prediction
predicted_population_2050 = np.polyval(poly_params, 2050)
plt.scatter(2050, predicted_population_2040, color= 'red', marker = 'x')
plt.plot([2040, 2050], [predicted_population_2040, predicted_population_2050], 'r--')

# label
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Population (Billion)', fontsize = 12)
plt.title('Popuation Predcition of China: 3rd Degree Polynomial Regression', fontsize = 16)
plt.legend()
plt.show()


# Print the predicted population for 2030
print("The poluation of China is predicted to be:")
print(f"{np.round(predicted_population_2030/(10**9), 3)} billion in 2030")
print(f"{np.round(predicted_population_2040/(10**9), 3)} billion in 2040")
print(f"{np.round(predicted_population_2050/(10**9), 3)} billion in 2050")
```


    
![png](output_27_0.png)
    


    The poluation of China is predicted to be:
    [1.452] billion in 2030
    [1.451] billion in 2040
    [1.417] billion in 2050
    


```python
India_df = df[df['Country'] == 'India'][population_columns]
India_df.values
```




    array([[5.57501301e+08,            nan,            nan, 1.05963368e+09,
            1.24061362e+09, 1.32286650e+09, 1.39638713e+09, 1.41717317e+09]])




```python
# Omit NaNs in the model (1980 and 1990)
India_df = df[df['Country'] == 'India'][['1970 Population', '2000 Population', '2010 Population', '2015 Population', 
                                        '2020 Population', '2022 Population']]
India_years = [year for year in years if year != 1980 and year != 1990]
India_years_linreg = [[year] for year in India_years]

India_population = [[int(pop)] for pop in India_df.values.flatten()]

```


```python
# Population Prediction of India

India_poly_params = np.polyfit(India_years, India_population, 3)

# Generate predictions for all years for visualization
India_predicted_population = np.polyval(India_poly_params, India_years)

#Visualize the data and the polynomial fit
plt.scatter(India_years, India_population, label='Actual Data', color = 'red')
plt.plot(India_years, India_predicted_population, label='Polynomial Fit (degree 3)', color='blue')

India_2030 = np.polyval(poly_params, 2030)
plt.plot([India_years[-1], 2030], [India_population[-1], India_2030], 'r--')
plt.scatter(2030, India_2030, color= 'red', marker = 'x', label=f'Prediction')

India_2040 = np.polyval(poly_params, 2040)
plt.scatter(2040, India_2040, color= 'red', marker = 'x')
plt.plot([2030, 2040], [India_2030, India_2040], 'r--')

India_2050 = np.polyval(poly_params, 2050)
plt.scatter(2050, India_2050, color= 'red', marker = 'x')
plt.plot([2040, 2050], [India_2040, India_2050], 'r--')


plt.xlabel('Year', fontsize = 12)
plt.ylabel('Population (Billion)', fontsize = 12)
plt.title('Popuation Predcition of India: 3rd Degree Polynomial Regression', fontsize = 16)
plt.legend()
plt.show()

# Print the predicted population
print("The poluation of India is predicted to be:")
print(f"{np.round(India_2030/(10**9), 3)} billion in 2030")
print(f"{np.round(India_2040/(10**9), 3)} billion in 2040")
print(f"{np.round(India_2050/(10**9), 3)} billion in 2050")
```


    
![png](output_30_0.png)
    


    The poluation of India is predicted to be:
    [1.452] billion in 2030
    [1.451] billion in 2040
    [1.417] billion in 2050
    


```python
# Find total population of each continent and sort by 2022 mean population from the highest to the lowest.
df2 = df.groupby('Continent')[num_df.columns].sum().sort_values(by = '2022 Population', ascending = False)
df3 = df2.filter(items = population_columns)
df3
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1970 Population</th>
      <th>1980 Population</th>
      <th>1990 Population</th>
      <th>2000 Population</th>
      <th>2010 Population</th>
      <th>2015 Population</th>
      <th>2020 Population</th>
      <th>2022 Population</th>
    </tr>
    <tr>
      <th>Continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asia</th>
      <td>2,104,314,136.00</td>
      <td>1,933,360,000.00</td>
      <td>2,334,719,776.00</td>
      <td>3,706,718,415.00</td>
      <td>4,187,125,190.00</td>
      <td>4,458,250,182.00</td>
      <td>4,652,801,584.00</td>
      <td>4,720,041,978.00</td>
    </tr>
    <tr>
      <th>Africa</th>
      <td>361,194,640.00</td>
      <td>480,817,791.00</td>
      <td>637,110,013.00</td>
      <td>817,508,493.00</td>
      <td>1,020,502,655.00</td>
      <td>1,156,663,993.00</td>
      <td>1,360,671,810.00</td>
      <td>1,425,529,262.00</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>655,923,991.00</td>
      <td>681,600,217.00</td>
      <td>709,689,789.00</td>
      <td>726,066,600.00</td>
      <td>735,613,934.00</td>
      <td>736,345,252.00</td>
      <td>745,792,196.00</td>
      <td>737,713,219.00</td>
    </tr>
    <tr>
      <th>North America</th>
      <td>315,434,606.00</td>
      <td>368,293,361.00</td>
      <td>421,266,425.00</td>
      <td>486,069,584.00</td>
      <td>542,720,651.00</td>
      <td>570,383,850.00</td>
      <td>594,236,593.00</td>
      <td>600,296,136.00</td>
    </tr>
    <tr>
      <th>South America</th>
      <td>192,947,156.00</td>
      <td>241,789,006.00</td>
      <td>297,146,415.00</td>
      <td>325,206,553.00</td>
      <td>348,262,142.00</td>
      <td>413,134,396.00</td>
      <td>431,530,043.00</td>
      <td>436,816,608.00</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>19,480,270.00</td>
      <td>22,920,240.00</td>
      <td>26,743,822.00</td>
      <td>31,222,778.00</td>
      <td>37,102,764.00</td>
      <td>40,403,283.00</td>
      <td>43,933,426.00</td>
      <td>45,020,499.00</td>
    </tr>
  </tbody>
</table>
</div>


<br/>

## Continent Population
The graph below represents the populations of the six continents from 1970 to 2022. Asia has consistently been the most populous continent, while Oceania has remained the least populous throughout this period. Asia experienced a dramatic population increase from 1990 to 2000, largely driven by rapidly developing countries such as China and India.

Overall, Asia’s population grew significantly between each time point until 2010, with more volatile changes. In contrast, the populations of the other continents grew more steadily over the same period.

Africa's population surpassed Europe's around 1995, making Africa the second most populous continent by the early 21st century. This shift can be attributed to factors such as a high fertility rate, declining mortality rate, and increasing life expectancy in Africa, as highlighted in [research](https://www.scirp.org/journal/paperinformation?paperid=103878#:~:text=Infant%20mortality%20rates%20in%20African,rapid%20growth%20of%20Africa's%20population.) from 2020.


```python
df3.columns = [col.replace(' Population', '') for col in df3.columns]

# Transpose and plot the data
df3.transpose().plot()

# Add title and labels
plt.title("Population of Continents Over Time", fontsize = 16)
plt.xlabel("Year", fontsize = 14)
plt.ylabel("Population (Billion)", fontsize = 14)

# Display the plot
plt.show()
```


    
![png](output_33_0.png)
    
<br/>

# Conclusion

To conclude, this analysis focused on the disparities in population distribution and density per area across countries. The population data revealed a left-skewed distribution, where the majority of countries have relatively small populations, while a few countries, such as China and India, dominate global population figures. 

By analyzing the 10 most densely populated regions, the project emphasizes that reducing global population density disparity requires a multi-faceted approach. Encouraging migration to less populated areas, improving access to education, and implementing sustainable development strategies could help alleviate pressures in densely populated regions. These actions not only promote more balanced growth but also ensure a fairer distribution of resources and opportunities for all nations.

Furthermore, the polynomial regression model used to predict the future populations of India and China for 2030, 2040, and 2050 highlighted some key limitations. The model does not account for variables such as government policies, economic shifts, or unforeseen global crises that can significantly impact population growth rates. As a result, the model underestimated population of India in 2030 by 63 million and overestimated population of China in 2030 by 36 million, suggesting that a more accurate model should employ more complex model than a polynomia regression model and take account for socioeconomic factors.
