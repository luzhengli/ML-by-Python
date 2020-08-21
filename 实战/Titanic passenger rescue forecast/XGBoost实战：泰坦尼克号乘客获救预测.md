```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
```

# 数据集介绍

此项目数据集分为2份数据集 train.csv 和 test.csv

train.csv: 训练集，共计891条数据

test.csv: 测试集，共计418条数据


字段|字段说明
-|-
PassengerId| 乘客编号
Survived   | 存活情况（存活：1 ; 死亡：0）
Pclass     | 客舱等级
Name       | 乘客姓名
Sex        | 性别
Age        | 年龄
SibSp      | 同乘的兄弟姐妹/配偶数
Parch      | 同乘的父母/小孩数
Ticket     | 船票编号
Fare       | 船票价格
Cabin      | 客舱号
Embarked   | 登船港口

PassengerId 是数据唯一序号；Survived 是存活情况，为预测标记特征；剩下的10个是原始特征数据。

# 数据初探
查看一下训练集和测试集前10条数据：


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```


```python
train.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>3</td>
      <td>Svensson, Mr. Johan Cervin</td>
      <td>male</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>7538</td>
      <td>9.2250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>3</td>
      <td>Connolly, Miss. Kate</td>
      <td>female</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>330972</td>
      <td>7.6292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>2</td>
      <td>Caldwell, Mr. Albert Francis</td>
      <td>male</td>
      <td>26.0</td>
      <td>1</td>
      <td>1</td>
      <td>248738</td>
      <td>29.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>3</td>
      <td>Abrahim, Mrs. Joseph (Sophie Halaut Easu)</td>
      <td>female</td>
      <td>18.0</td>
      <td>0</td>
      <td>0</td>
      <td>2657</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>3</td>
      <td>Davies, Mr. John Samuel</td>
      <td>male</td>
      <td>21.0</td>
      <td>2</td>
      <td>0</td>
      <td>A/4 48871</td>
      <td>24.1500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



查看训练集是否有缺失数据，以及数据的类型：


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB


统计各属性缺失值数量：


```python
train.isnull().sum()  
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



- 从上述分析可见，**属性 Age、Cabin（大量缺失）、Embarked（极少缺失） 带有缺失值，缺失值数量分别为177、687、2**

查看训练集的一些基本统计信息：


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



- 从以上分析可知，获救人数占总人数的0.383838、乘客平均年龄是29.699118
- **疑问**：上述统计可以发现有乘客的船票价格（Fare）是0，这个难道真是如电影中所说中了彩票？

# 数据清洗

## 缺失值处理
1. 客舱号Cabin列由于存在大量的空值，如果直接对空值进行填空，带来的误差影响会比较大，先不选用Cabin列做特征
2. Age列比较重要，缺失数量还可接受，因此这里使用中位数进行填充（好处：采用中位数可以保证年龄是个整数）


```python
train.Age = train.Age.fillna(train.Age.median())  # 填充Age列的
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.361582</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>13.019697</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



## 特征选择
1. PassengerId是一个连续的序列，对于是否能够存活的判断无关，不选用PassengerId作为特征


```python
feature_name = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']  # 选用的属性/特征名

X = train[feature_name]
y = train.Survived
```

## 数据划分
数据集 test 是不带标签的，对模型的训练和评估都排不上用处。这里选取数据集 train 作为交叉验证集。

# XGBoost模型
## 建立模型


```python
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', 
                  max_depth=3, n_estimators=7, 
                  n_jobs=-1,)
```

## 基于交叉验证的网格搜索


```python
param_grid = {
    'max_depth': range(2, 20),
    'n_estimators': list(range(3, 20, 2)),
}  # 要搜索的参数空间
gsCv = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, refit=True, verbose=1, n_jobs=-1, cv=5)  # 5折交叉验证网格搜索（使用所有CPU所有线程并行）
gsCv.fit(X, y)
```

    Fitting 5 folds for each of 162 candidates, totalling 810 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    2.2s
    [Parallel(n_jobs=-1)]: Done 810 out of 810 | elapsed:    4.3s finished





    GridSearchCV(cv=5,
                 estimator=XGBClassifier(base_score=None, booster='gbtree',
                                         colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=None, gamma=None,
                                         gpu_id=None, importance_type='gain',
                                         interaction_constraints=None,
                                         learning_rate=None, max_delta_step=None,
                                         max_depth=3, min_child_weight=None,
                                         missing=nan, monotone_constraints=None,
                                         n_estimators=7, n_jobs=-1,
                                         num_parallel_tree=None, random_state=None,
                                         reg_alpha=None, reg_lambda=None,
                                         scale_pos_weight=None, subsample=None,
                                         tree_method=None,
                                         validate_parameters=False,
                                         verbosity=None),
                 n_jobs=-1,
                 param_grid={'max_depth': range(2, 20),
                             'n_estimators': [3, 5, 7, 9, 11, 13, 15, 17, 19]},
                 verbose=1)




```python
print(gsCv.best_score_)  # 输出最好的模型的分数（交叉验证集上的平均精准度）
print(gsCv.best_params_)  # 输出最好的模型参数
best_model = gsCv.best_estimator_  # 获取到最优的模型
```

    0.7206327286422698
    {'max_depth': 4, 'n_estimators': 7}


## 优化：增加特征 Sex 和 Embarked 列，查看对模型预测的影响
1. 将性别 Sex 的值映射为0或1
2. 将登船港口 Embarked 的值映射为0、1或2


```python
# 性别属性预处理 male => 0，female => 1
train.loc[train.Sex == "male", "Sex"] = 0  
train.loc[train.Sex == "female", "Sex"] = 1   
```


```python
train.Embarked.value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64




```python
#缺失值用最多的S进行填充
train.Embarked = train.Embarked.fillna('S') 

#地点用0,1,2
train.loc[train["Embarked"] == "S", "Embarked"] = 0    
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
```


```python
# 转换数据类型（这里必须将object类型转为int 否则训练会报错）
train.Sex = train.Sex.astype('int')
train.Embarked = train.Embarked.astype('int')
```


```python
# 重新进行特征选择
feature_name = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train[feature_name]
y = train.Survived
```


```python
# 建立模型
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', 
                  max_depth=3, n_estimators=7, 
                  n_jobs=-1,)

# 要搜索的参数空间
param_grid = {
    'max_depth': range(2, 20),
    'n_estimators': list(range(3, 20, 2)),
}  

# 基于交叉验证的网格搜索
gsCv = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, refit=True, verbose=1, n_jobs=-1, cv=5)  # 5折交叉验证网格搜索（使用所有CPU所有线程并行）
gsCv.fit(X, y)
```

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    Fitting 5 folds for each of 162 candidates, totalling 810 fits


    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=-1)]: Done 810 out of 810 | elapsed:    2.3s finished





    GridSearchCV(cv=5,
                 estimator=XGBClassifier(base_score=None, booster='gbtree',
                                         colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=None, gamma=None,
                                         gpu_id=None, importance_type='gain',
                                         interaction_constraints=None,
                                         learning_rate=None, max_delta_step=None,
                                         max_depth=3, min_child_weight=None,
                                         missing=nan, monotone_constraints=None,
                                         n_estimators=7, n_jobs=-1,
                                         num_parallel_tree=None, random_state=None,
                                         reg_alpha=None, reg_lambda=None,
                                         scale_pos_weight=None, subsample=None,
                                         tree_method=None,
                                         validate_parameters=False,
                                         verbosity=None),
                 n_jobs=-1,
                 param_grid={'max_depth': range(2, 20),
                             'n_estimators': [3, 5, 7, 9, 11, 13, 15, 17, 19]},
                 verbose=1)




```python
print(gsCv.best_score_)  # 输出最好的模型的分数（交叉验证集上的平均精准度）
print(gsCv.best_params_)  # 输出最好的模型参数
best_model = gsCv.best_estimator_  # 获取到最优的模型
```

    0.8417864540832338
    {'max_depth': 6, 'n_estimators': 13}


**小结**：可以看到，添加属性 Sex 和 Embarked 后模型的性能得到了提升，说明好的特征可以提高模型的预测能力。
