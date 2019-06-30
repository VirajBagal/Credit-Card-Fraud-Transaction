#import basic modules
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis,skew
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score,make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import zscore
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#Loading Data 
data=pd.read_csv('../input/creditcard.csv')
data.head()

#Basic Overview of Data
data.shape
data.info()
data.isnull().sum()
data.describe()

# Basic EDA

# Let us plot V1,V2,.. upto V28. V16, V18 and V19 seem to be closer to normal distribution than any other columns. 
# We will dive a bit deeper in the next step.

f,ax=plt.subplots(7,4,figsize=(15,15))
for i in range(28):
    sns.distplot(data['V'+str(i+1)],ax=ax[i//4,i%4])
    

plt.tight_layout()
plt.show()


# Let us create a dataframe with columns mean, standard deviation, max, min, skewness and kurtosis of each of the V columns. 
# Let us plot each of these features.

stats=pd.DataFrame()
cols=[col for col in data.columns[1:29]]
mean=data[cols].mean(axis=0)
std=data[cols].std(axis=0)
max_val=data[cols].max(axis=0)
min_val=data[cols].min(axis=0)
skew=data[cols].skew(axis=0)
kurt=data[cols].kurt(axis=0)
stats['mean']=mean
stats['std']=std
stats['max']=max_val
stats['min']=min_val
stats['skew']=skew
stats['kurt']=kurt
stats.index=cols

x_ticks=np.arange(1,29,1)
f,ax=plt.subplots(2,3,figsize=(15,8))
for i in range(6):
    ax[i//3,i%3].plot(x_ticks,stats.iloc[:,i].values,'b.')
    ax[i//3,i%3].set_title(stats.columns[i])
    
plt.tight_layout()
plt.show()

# There is atleast one outlier in each of the graphs. In 'mean' graph, mean of V3 is quite different from other columns.
# 'std' graph looks fine and clearly shows decreasing trend in standard deviation with increasing value of column. 
# In 'max' graph, max value of V6 and V7 is quite different. In 'min' graph, minimum value of V5 is quite different. 
# In 'skew' aswellas 'kurtosis' graph, skewness and kurtosis of V29 are quite high. Let us understand what skewness and kurtosis actually mean. 
# Skewness is a measure of how unsymmetric are the tails of the distribution. Normal distribution has 0 skewness because it has perfectly symmetric tails. 
# V8 and V29 are skewed to left and right respectively to large extent. Skewed to left means the tails are longer on the left side rather than right side of mean. 
# Kurtosis is a measure of how much probability mass is concentrated on the shoulders of distribution. Normal distribution has a kurtosis of 3 (excess kurtosis=0). 
# As kurtosis increases, probability mass decreases from shoulder and spreads at the center and tails of the distribution. Minimum kurtosis possible is 0 (excess kurtosis= -3) and maximum possible is infinite. 
# 0 kurtosis implies all the probability mass is concentrated on the shoulders. Many columns have very high excess kurtosis. Both skewness and kurtosis are unitless. 
# Outliers in a sample have more effect on kurtosis than on skewness because kurtosis involves fourth moment while skewness involves third moment of distribution.

print(stats.loc[['V16','V18','V19'],:])

# Here we can see that only looking at graphs is deceptive. I had thought that V16,V18 and V19 look close to normal distributions. But kurtosis of V16 says otherwise. 
# V18 is quite close to normal distribution.


plt.figure(figsize=(15,10))
sns.heatmap(data.corr())
plt.show()

# None of the columns is significantly correlated with Class. Amount is kind of significantly correlated with V7 and V20.



