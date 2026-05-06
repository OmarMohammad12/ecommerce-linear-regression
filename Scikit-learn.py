#%%
#basic imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
#load and read the head of the data
df=pd.read_csv("Ecommerce.csv")
df.head()
#%%
#checking for nulls and understand the data more
df.info()
#%%
df.describe()

#%%
#starting EDA
sns.jointplot(x="Time on Website",y="Yearly Amount Spent",data=df,alpha=0.5)
#%%
sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=df,alpha=0.5)

#%%
#exploer relations at once
sns.pairplot(df,kind="scatter",plot_kws={"alpha": 0.4})
#%%
#membership length looks strongly related to spending
sns.lmplot(x="Length of Membership",
           y="Yearly Amount Spent",
           data=df,
           scatter_kws={"alpha": 0.4})

#%%
from sklearn.model_selection import train_test_split
#%%
# features and target
x=df[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]
y=df["Yearly Amount Spent"]
#%%
# 70% train, 30% test
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
#%%
#imported the model
from sklearn.linear_model import LinearRegression
#%%
lm = LinearRegression()
lm.fit(X_train,y_train)
#%%
lm.coef_
#%%
#how much does each feature affect spending?
cdf=pd.DataFrame(lm.coef_,x.columns,columns=["Coefficient"])
print(cdf)
#%%
#predictions on test data
predictions=lm.predict(X_test)
predictions
#%%
#predicted vs actual - should look like a straight line if proper model was used
sns.scatterplot(x=predictions,y=y_test)
plt.xlabel("Predictions")
plt.title("Evaluation of my LM model")
#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import math

#%%
# how good is the model?

print("Mean Absolute Error",mean_absolute_error(y_test,predictions))
print("Mean squared Error",mean_squared_error(y_test,predictions))
print("RMSE",math.sqrt(mean_squared_error(y_test,predictions)))
print("R2 Score", r2_score(y_test, predictions))
#%%
# checking residuals
residuals=y_test-predictions

#%%
# if points follow the line, the model is solid
import pylab
import scipy.stats as stats
stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()