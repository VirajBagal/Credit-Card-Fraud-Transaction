# Now let us look how the Classes are distributed.

sns.countplot(data['Class'])
print((data['Class'].value_counts()/data.shape[0])*100)

# Class 1 corresponds to Fraud Transactions. We see that only 0.172% of transactions are fraud. 
# The data is highly imbalanced. Since just upsampling with replace=True will lead to lot of duplicates. So, I shall use a technique called SMOTE. 
# It is basically oversamplng technique which tweaks just one column a little bit and thus a new sample of minority class is created.
# I shall use SMOTE only on training data so that fraud transactions remain as minority in case of validation as it would be in real world scenario.

#splitting the data

X=data.drop(['Class','Time'],axis=1)
Y=data['Class']
train_X,test_X,train_y,test_y=train_test_split(X,Y,random_state=5,test_size=0.2)

# Scaling the data

sc=StandardScaler()
train_X=sc.fit_transform(train_X)
test_X=sc.transform(test_X)
train_X=pd.DataFrame(train_X,columns=X.columns)
test_X=pd.DataFrame(test_X,columns=X.columns)

#Using SMOTE to generate new data for minority class

sm=SMOTE(random_state=5)
train_X_res,train_y_res=sm.fit_sample(train_X,train_y)
train_X_res=pd.DataFrame(train_X_res,columns=train_X.columns)
train_y_res=pd.Series(train_y_res,name='Class')

# Let us see how much separable are the two classes. If we consider all the V columns then we shall have 28 dimensional space.
# We cannot visualise such high dimensional data. I shall use TSNE to project the points from 28 dimensional space to 2 dimensional space. 
# For faster computation, I shall only take 2500 points from each class.

#Let us use TSNE for projecting data to two dimensional space.

train=pd.concat([train_X_res,train_y_res],axis=1)
fraud=train[train['Class']==1].sample(2500)
non_fraud=train[train['Class']==0].sample(2500)
tsne_data=pd.concat([fraud,non_fraud],axis=0)
tsne_data_1=tsne_data.drop(['Class'],axis=1)

tsne=TSNE(n_components=2,random_state=5,verbose=1)
tsne_trans=tsne.fit_transform(tsne_data_1)

tsne_data['first_tsne']=tsne_trans[:,0]
tsne_data['second_tsne']=tsne_trans[:,1]
plt.figure(figsize=(15,10))
sns.scatterplot(tsne_data['first_tsne'],tsne_data['second_tsne'],hue='Class',data=tsne_data)

