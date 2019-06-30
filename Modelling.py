# Let us use different models

models=[SVC(probability=True),LogisticRegression(),LinearDiscriminantAnalysis(),DecisionTreeClassifier(),
       ExtraTreesClassifier(n_estimators=100),AdaBoostClassifier(n_estimators=100),RandomForestClassifier(n_estimators=100)]

model_names=['SVC','LR','LDA','DTC','ETC','ABC','RFC']
train_score=[]
score_1=[]
test_score=[]

#Defining function to train models and predict probabilities.

skf=StratifiedKFold(n_splits=5,random_state=5)
def get_model(train_X,train_y,test_X,test_y,model):
    for train_index,val_index in skf.split(train_X,train_y):
        train_X_skf,val_X_skf=train_X.iloc[train_index,:],train_X.iloc[val_index,:]
        train_y_skf,val_y_skf=train_y.iloc[train_index],train_y.iloc[val_index]
        clf=model
        clf.fit(train_X_skf,train_y_skf)
        pred=clf.predict_proba(val_X_skf)[:,1]
        score=average_precision_score(val_y_skf,pred)
        score_1.append(score)
        
    train_score.append(np.mean(score_1))
    clf.fit(train_X,train_y)
    pred_prob=clf.predict_proba(test_X)[:,1]
    score_test=average_precision_score(test_y,pred_prob)
    test_score.append(score_test)

# To increase computational speed, I sampled only 50000 points from train_X_res and 10000 points from test_X

train_X_sam=train_X_res.sample(10000)
train_X_index=train_X_sam.index
train_y_sam=train_y_res[train_X_index]
train_X_sam.reset_index(drop=True,inplace=True)
train_y_sam.reset_index(drop=True,inplace=True)
test_X_sam=test_X.sample(1000)
test_X_index=test_X_sam.index
test_y_sam=test_y[test_X_index]
test_X_sam.reset_index(drop=True,inplace=True)
test_y_sam.reset_index(drop=True,inplace=True)

for model in models:
    get_model(train_X_sam,train_y_sam,test_X,test_y,model)

result=pd.DataFrame({'models':model_names,'train_score':train_score,
                    'test_score':test_score},index=model_names)


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
result['train_score'].plot.bar()
plt.title('Train Score')
plt.subplot(1,2,2)
result['test_score'].plot.bar()
plt.title('Test Score')
plt.tight_layout()
plt.show()

#All models are overfitting. But Logistic Regression, ETC, ABC and RFC provide decent test scores.

#Let us try Blending the top 4 performers with XGBClassifier

clf=LogisticRegression()
clf.fit(train_X_sam,train_y_sam)
lr_pred=clf.predict_proba(test_X)[:,1]

clf_2=AdaBoostClassifier()
clf_2.fit(train_X_sam,train_y_sam)
abc_pred=clf_2.predict_proba(test_X)[:,1]

clf_3=ExtraTreesClassifier()
clf_3.fit(train_X_sam,train_y_sam)
etc_pred=clf_3.predict_proba(test_X)[:,1]

clf_4=RandomForestClassifier()
clf_4.fit(train_X_sam,train_y_sam)
rfc_pred=clf_4.predict_proba(test_X)[:,1]

xgb=XGBClassifier()
xgb.fit(train_X_sam,train_y_sam)
xgb_pred=xgb.predict_proba(test_X)[:,1]


#This blending score is better than each of the models alone.

blending_pred=0.20*(lr_pred+etc_pred+rfc_pred+abc_pred+xgb_pred)
blending_score=average_precision_score(test_y,blending_pred)
print(blending_score)

# Things which you can further try :

# 1. Using different scaling of variables
# 2. Using different algorithm for upsampling
# 3. Using different models
# 4. Hyperparamter tuning of mutiple models and then blending
# 5. Ensembling