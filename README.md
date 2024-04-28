# EXP-03-FEATURE-ENCODING-AND-TRANSFORMATION-PROCESS
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```python

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
 ![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/d50feeb0-02d1-4d80-988b-73beff9a922e)

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/3b99bb56-9f79-4c19-8073-6acb51e3c7ff)


```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/729b6d4a-54ad-49e3-b375-9a91aed0d8cd)

```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/a204cb14-d8f7-405f-9d74-b899c7475853)


```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/e14d5ed4-9a92-422a-a194-f3279acc29bb)

```python
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/5f4c3d75-0fd9-4c5b-8809-41697ae76a58)


```python
be=BinaryEncoder()
nd=be.fit_transform(df["Ord_2"])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/53805568-c5f6-4e6c-bb81-316490b1cb75)


```python
from category_encoders import TargetEncoder
te=TargetEncoder()
te=TargetEncoder()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/f4a82029-6738-4453-952f-d80605ddb10e)


```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/f958785d-f1c9-471c-b04f-14d89376f25d)


```python
df.skew()
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/eef1f465-295a-469d-914e-7313a046c046)

![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/d046aa73-0223-47cc-a784-dacfc95dddce)

![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/18ba4598-6720-49bc-b83b-4ceb6023f403)

![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/5eabf620-9326-44f0-812b-5f617a360f03)

![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/42c3fb2a-3455-4ef3-9e61-2c92a9859451)

![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/de879772-39d9-4bfb-9b6a-d14dd48f5e93)


```python
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/31a20908-73ec-4ec5-a674-cda800631d2b)

```python
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/a052d2fd-ba04-4347-bb2c-f531649ce468)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal")
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df


```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/6836d5a9-d235-4ff7-b488-4f360a65ede7)

```python
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/df29d423-66d0-4b85-90e1-8dab668768ac)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/9714882b-4b8c-484c-90a7-3e7033ff1db5)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/c0be92c4-93aa-4a26-9410-b4b792fc1315)

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/ca10ff7f-c5dc-49ff-8a56-a2a92254e5b4)

```python
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/varshasharon/EXNO-3-DS/assets/98278161/b9f8b166-1f1c-4d8a-86d9-7bf9db62e8be)

# RESULT:
 The Feature Encoding and Transformation process has been executed successfully.
       
