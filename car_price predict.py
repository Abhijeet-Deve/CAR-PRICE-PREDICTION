#!/usr/bin/env python
# coding: utf-8

# # **PROJECT NAME: CAR PRICE PREDICTIONS**
# 
# ## **Project Type:** <span style="font-size:20px">REGRESSION</span>
# 
# ## **Contribution:** <span style="font-size:20px">INDIVIDUAL</span>
# 
# ## **Name:** <span style="font-size:20px">ABHIJEET JOSHI</span>
# 
# ---
# 
# ### Project Summary:
# **Project Title:** Car Price Prediction Model
# 
# **Objective:** The objective of this project is to develop a machine learning model that accurately predicts the price of a car based on various features such as 'Make', 'Model', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Engine', 'Max Power', 'Max Torque', 'Drivetrain', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity'.
# 
# **Data:** The project will utilize a dataset containing information about past car sales, including features such as 'Make', 'Model', 'Price', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Engine', 'Max Power', 'Max Torque', 'Drivetrain', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity'.
# 
# **Methodology:**
# 1. **Data Preprocessing:** Cleaning the dataset, handling missing values, encoding categorical variables, and scaling numerical features.
# 2. **Feature Selection/Engineering:** Identifying important features that contribute most to the prediction and possibly creating new features that capture relevant information.
# 3. **Model Selection:** Experimenting with various machine learning algorithms such as linear regression, decision trees, random forest, AdaBoost, gradient boost, and SVM to determine the best-performing model.
# 4. **Model Training and Evaluation:** Splitting the dataset into training and testing sets, training the selected models on the training data, and evaluating their performance using metrics such as mean squared error, mean absolute error, and R²-Score.
# 5. **Hyperparameter Tuning:** Fine-tuning the parameters of the chosen model(s) using techniques such as grid search or random search to optimize performance.
# 6. **Model Deployment:** Deploying the final trained model in a production environment, making it accessible for users to input car features and receive price predictions.
# 
# **Expected Outcome:** The expected outcome of this project is a robust and accurate car price prediction model that can assist car buyers, sellers, and motorcar professionals in estimating car values more reliably.
# 
# **Benefits:**
# 1. **For Car Buyers/Sellers:** Provide information about the car value, aiding in decision-making regarding buying, selling, or renting properties.
# 2. **For Investors:** Assists in identifying potential and future returns.
# 

# # **Problem Statement**
# 
# ## **Car Price Prediction**
# 
# **Objective:** Develop a prediction model to estimate the selling prices of residential cars based on various features.
# 
# **Background:** Car prices are influenced by numerous factors including 'Make', 'Model', 'Price', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Location', 'Color', 'Owner', 'Seller Type', 'Engine', 'Max Power', 'Max Torque', 'Drivetrain', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity', and investors.
# 
# **Data:** The dataset typically includes historical data on car prices along with various attributes for each car, such as:
# 1. **Location:** ZIP code, neighborhood, proximity to schools and public transportation.
# 2. **Physical Attributes:** Car color, car safety, build quality, average, etc.
# 
# ### Methodology:
# 1. **Data Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features.
# 2. **Exploratory Data Analysis (EDA):** Identify patterns and correlations between features and the target variable (car prices).
# 3. **Feature Engineering:** Create new features or modify existing ones to improve model performance.
# 4. **Model Selection:** Compare various regression models (e.g., linear regression, decision trees, random forest, gradient boosting, AdaBoost) to identify the best-performing model.
# 5. **Model Training and Evaluation:** Train the chosen model on the training dataset and evaluate its performance using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) on the validation dataset.
# 6. **Hyperparameter Tuning:** Optimize the model's parameters to improve predictions on new data.
# 

# # Let's Begin !
# 

# # Import Libries

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt #for simple graphs
import seaborn as sns #for coplex graphs
from statsmodels.stats.outliers_influence import variance_inflation_factor

#### for  trai test split
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV

### liner rigration
from sklearn.linear_model import LinearRegression

##### hypothesis testing
from scipy.stats import shapiro

####  evalution matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor


from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso,Ridge
from sklearn.svm import SVR


# In[4]:


df = pd.read_csv(r'C:\Users\rocky\OneDrive\Desktop\Data Science work\machine learing\supervised ML\car details v4.csv')


# In[5]:


df


# # Understanding the more about data

# # Dataset first viwe

# In[24]:


df.head()             #### return first five rows of data default 


# In[25]:


df.tail()        ###3 return the bottom five rows of data


# In[26]:


df.info()           #### return the all about inforomation of the dataset


# In[27]:


df.describe()             ### return the statestical inforomation of the dataset


# In[19]:


df.shape      ## return shape of the dataset


# In[20]:


df.size           ## return all the size of the dataset


# In[21]:


df.index          ### return the index of a rows and steps


# In[22]:


df.columns                #### return all the labal of the columns


# In[23]:


df.axes             #### return rows and columns label


# In[30]:


df.dtypes        #### return the data type of data columns


# In[31]:


df.rename(columns={'Fuel Type':'Fuel_Type'},inplace = True)
df


# In[32]:


df.rename(columns={' Seller Type':' Seller_Type'},inplace = True)
df


# In[33]:


df.rename(columns={'Max Power':'Max_Power'},inplace = True)
df


# In[34]:


df.rename(columns={'Max Torque':'Max_Torque'},inplace = True)
df


# # **Encoding**
# **change all object into the numarical format by using encoding**

# In[35]:


Make = LabelEncoder()
Model = LabelEncoder()
Fuel_Type = LabelEncoder()
Transmission = LabelEncoder()
Location = LabelEncoder()
Color = LabelEncoder()
Owner = LabelEncoder()

Max_Power = LabelEncoder()
Max_Torque = LabelEncoder()
Engine = LabelEncoder()
Max = LabelEncoder()
Drivetrain = LabelEncoder()


# In[36]:


df['Make']= Make.fit_transform(df['Make'])


# In[37]:


df['Model']= Model.fit_transform(df['Model'])
df['Fuel_Type']= Fuel_Type.fit_transform(df['Fuel_Type'])
df['Transmission']= Transmission.fit_transform(df['Transmission'])
df['Location']= Location.fit_transform(df['Location'])
df['Color']= Color.fit_transform(df['Color'])
df['Owner']= Owner.fit_transform(df['Owner'])
df['Max_Power']= Max_Power.fit_transform(df['Max_Power'])
df['Max_Torque']= Max_Torque.fit_transform(df['Max_Torque'])
df['Engine']= Engine.fit_transform(df['Engine'])

df['Drivetrain']= Drivetrain.fit_transform(df['Drivetrain'])


# In[38]:


df


# In[41]:


df['Seller Type'].unique()


# In[42]:


df = df.replace({'Corporate':0,'Individual':1,'Commercial Registration':2})


# In[43]:


df


# # Check Duplicate value

# In[48]:


print(f"Data is duplicated ? {df.duplicated().value_counts()},unique values with {len(df[df.duplicated()])} duplication")


# # Check uniqe value of each Variable 

# In[49]:


for i in df.columns.tolist():
  print(f"No. of unique values in {i} is {df[i].nunique()}.")


# # Check missing values

# In[53]:


df.isna()        ### this funtion is return the all null values in a dataset but true and false format


# In[56]:


df.isna().sum()


# # Visualizing the null values

# In[57]:


sns.heatmap(df.isna())


# In[58]:


for column in ['Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)


# In[59]:


df.isna().sum()


# ## Preprocessing the Dataset
# 
# ### Why Do We Need to Handle Missing Values?
# 
# In real-world scenarios, datasets often contain a significant amount of missing values. These missing values can occur due to various reasons, such as data corruption or failure to record data. Handling missing data is crucial during the preprocessing stage of a dataset for the following reasons:
# 
# 1. **Algorithm Compatibility**: Many machine learning algorithms do not support missing values. If these missing values are not addressed, it can lead to errors or inaccurate model training.
# 2. **Data Integrity**: Missing values can distort statistical properties of the dataset, affecting the analysis and insights derived from it.
# 3. **Model Performance**: Properly handling missing values can improve the performance and reliability of machine learning models.
# 
# To ensure the dataset is ready for analysis and modeling, it is important to first check for missing values and handle them appropriately.
# 

# # Using Funtion to Check and Handle Outliers for
# Each Independent Columns By Using
# Boxplot

# In[60]:


def check_and_remove_outliers(df, columns):
    # Visualize the original data
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[columns], palette="Set3")
    plt.title('Box Plot Before Removing Outliers')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.show()

    # Remove outliers
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df_cleaned = df.copy()
    for column in columns:
        df_cleaned = remove_outliers(df_cleaned, column)

    # Visualize the cleaned data
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_cleaned[columns], palette="Set3")
    plt.title('Box Plot After Removing Outliers')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.show()

    return df_cleaned

# Example usage with your dataframe 'df' and columns list
# cleaned_df = check_and_remove_outliers(df, ["Make", "Model", "Year", "Kilometer", "Fuel_Type", "Transmission"])

# Use the function to check for outliers, remove them, and visualize the results
cleaned_df = check_and_remove_outliers(df, ["Year", "Kilometer","Fuel_Type","Owner","Seller Type","Drivetrain","Width","Seating Capacity","Fuel Tank Capacity"])


# In[61]:


cleaned_df = check_and_remove_outliers(df, [ "Kilometer"])


# In[62]:


sns.boxplot(df['Make'])


# In[63]:


sns.histplot(df['Make'])


# In[64]:


sns.boxplot(df['Model'])


# In[65]:


sns.kdeplot(df['Model'],fill = True)


# In[66]:


sns.kdeplot(df['Price'],fill = True)


# In[67]:


plt.bar(df["Year"],df["Price"])


# In[68]:


plt.plot(df['Kilometer'],linestyle = 'dotted')


# In[69]:


sns.boxplot(df['Fuel_Type'])


# In[70]:


sns.kdeplot(df['Fuel_Type'],fill = True,color = "r")


# In[71]:


sns.boxplot(df['Transmission'])


# In[72]:


sns.boxplot(df['Location'])


# In[73]:


sns.kdeplot(df['Location'],linestyle = 'dotted',color = 'g', fill = True)


# In[74]:


sns.boxplot(df['Color'])


# In[78]:


sns.barplot(df['Color'],color = "g")


# In[79]:


pd.plotting.lag_plot(df['Owner'])


# In[80]:


import seaborn as sns

# Bar plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x="Seller Type", y="Price", data=df)
plt.xlabel("Seller Type")
plt.ylabel("Price")
plt.title("Price by Seller Type")
plt.show()


# In[81]:


sns.boxplot(df['Engine'])


# In[82]:


plt.hist(df['Engine'])


# In[83]:


sns.boxplot(df['Max_Power'])


# In[84]:


sns.distplot(df['Max_Power'])


# In[85]:


sns.boxplot(df['Max_Torque'])


# In[86]:


sns.distplot(df['Max_Torque'],color = "g")


# In[87]:


plt.hist(df['Drivetrain'],color = "r")


# In[88]:


sns.boxplot(df['Length'])


# In[89]:


plt.figure(figsize=(40,40))
sns.displot(df["Length"])


# In[90]:


sns.scatterplot(x=df["Width"],y = df["Price"])


# In[91]:


plt.fill_between(df["Width"],df["Price"],color = "r")


# In[92]:


sns.kdeplot(df['Seating Capacity'],fill = True, color = "g")


# In[93]:


sns.histplot(df['Fuel Tank Capacity'])


# In[94]:


# Histogram Graph

plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='blue', bins=30)
plt.title('Histogram of Column Price')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# # Assumption - 1 Linearity
# 1. Before Model Training
# 2. Correlation : Strenght Between Two Features
# 3. Range : -1 To +1

# In[95]:


df.corr()


# # Checking the corralation between two variables

# In[96]:


sns.heatmap(df.corr())


# # Heatmap
# A correlation Heatmap is a type of graphical representation that displays the correlation
# matrix which to determine the correlation between different variables.

# In[100]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[101]:


sns.pairplot(df.corr())


# In[102]:


sns.pairplot(df)


# # Assumption - 2 Multicolinearity

# # 1. Before Model Training
# 2. Two Independent features should not be strongly correlated with each other
# 3. VIF(Variance Inflation Error)
# 4. Range : 1 To Infinity
# 5. 1 to 10

# In[103]:


df.dtypes


# In[104]:


df


# In[105]:


df.columns


# In[108]:


# Deleting the 'Unnamed: 2' column from the DataFrame
if 'Unnamed: 2' in df.columns:
    df.drop(columns=['Unnamed: 2'], inplace=True)
    print("'Unnamed: 2' column has been deleted.")
else:
    print("'Unnamed: 2' column does not exist in the DataFrame.")

# Display the DataFrame to confirm the deletion
print(df)


# In[109]:


df


# In[110]:


df1 =df.iloc[:,:-1]


# In[111]:


df1


# In[112]:


y = df["Price"]


# # Scaling 

# In[115]:


std = StandardScaler()
std_feat = std.fit_transform(df)
df = pd.DataFrame(std_feat)
df


# In[116]:


vif_ind = pd.DataFrame()         #### convert the datafreame vif_df
vif_ind["Features"] = df1.columns         ##### create the column of all independat feture 

vif_ind


# In[117]:


vif_list = []

for i in range(df1.shape[1]):
    vif = variance_inflation_factor(df1.to_numpy(),i)
    vif_list.append(vif)
   
vif_ind["VIF"] = vif_list

vif_ind


# In[118]:



x = df1


# In[119]:


x


# In[120]:


y


# In[121]:


xtrain,xtest, ytrain,ytest = train_test_split(x,y, test_size=0.2, random_state=10) # test_size >> 20% testing data, random_state >> taking samples randomly for training as well as testing


# In[122]:


xtrain.shape


# In[123]:


xtest.shape


# In[124]:


ytrain.shape


# In[125]:


ytest.shape


# # Algoritham - 01 Linear Regression Algoritham
# instantiate Linear Regrassion

# In[126]:


lin_reg = LinearRegression()
lin_reg


# In[127]:


lin_reg_model = lin_reg.fit(xtrain,ytrain)  # linear regression algorithm works, finds best fit line, SGD algo works(optimal values f m, c, mse)
lin_reg_model


# In[128]:


xtrain.head()


# In[129]:


ytrain.head()


# # prediction training and testing data

# In[130]:


ytrain_predict = lin_reg_model.predict(xtrain)
ytrain_predict


# # Loss Function
# calculate Between two data point  ie. Error/ Resudual

# In[131]:


error = ytrain - ytrain_predict
error


# # Model Evaluation on Training data and Testing data

# In[132]:


mse = mean_squared_error(ytrain,ytrain_predict)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,ytrain_predict)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,ytrain_predict)
print(f"r2scor is : {r2scor}")

# **Looks like our train set's R2 Score value is 0.66 that means our model is
able to capture most of the data variance . lets save it in a dataframe for
later comparisons **
# In[134]:


ytest_predict = lin_reg_model.predict(xtest)
ytest_predict


# In[135]:


#### for testing 

mse = mean_squared_error(ytest,ytest_predict)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,ytest_predict)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,ytest_predict)
print(f"r2scor is : {r2scor}")

 ** The test set's R2 Score is 0.73. this means our linear model is performing not well on the data. Let us try visualize our residuals and see if there is heteroscedasticity (unequal variance or scatter).**
# # Assumtion 3 Normality of resudual**
# 1. After Model Training.
# 2. Residual Means the diffrence between observed and predicted values in a
# regression analysis it is a measure of how well a model fits the data.
# 3. Residual should be normally distributed
# 4. Check resudual by using kdplot and histogram
error should be normally distributed

check by :
visualization : kdeplot
or
hypothesis testing : shapiro, kstest
# In[137]:


sns.kdeplot(error)


# # Hypothesis testing

# In[138]:


stats, pval = shapiro(error)

if pval > 0.05:
    print("Data is normally distributed")
    print("Null hypothesis H0 is True")
else: 
    print("Data is not normally distributed")
    print("Alternative hypothesis H1 is True")


# # Assumption - 4 Homoscadasticity
Error should have constant variable
# In[140]:


sns.scatterplot(x = ytrain, y = error)


# # Regularization (used to reduce overfitting of liner model)
# 
# 1. Regularization techniques like Lasso(L1), Ridge(L2), and ElasticNet(combination of
# L1 and L2) add penalties to the model for having large coefficients, which helps to
# reducing overfitting for linear models.
# 2. Ridge and Lasso Regression are types of regularization techniques
# 3. Regularization techniques are used to deal with overfitting and when the dataset is
# large
# 4. Ridge and Lasso Regression involve adding penalties to the regression funtion

# # Applying Lasso Avoiding model overfitting
# **Lasso Regression**
# 1. Lasso regression analysis is a shrinkage and variable selection method for linear
# regression models. the goal of lasso regression is to obtain the subset of prediction
# that minimizes prediction error for a quantitative response variable. it uses the linear
# regression model with L1 regularization.

# # instantiat Lasso Rgrssion

# In[141]:


lasso_fun = Lasso()
lasso_fun


# In[142]:


lasso_model = lasso_fun.fit(xtrain,ytrain)
lasso_model


# In[143]:


y_pridition = lasso_model.predict(xtrain)


# In[144]:


### for training

mse = mean_squared_error(ytrain,y_pridition)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,y_pridition)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,y_pridition)
print(f"r2scor is : {r2scor}")


# In[145]:


ytest_pridiction = lasso_model.predict(xtest)


# In[146]:


#### for testing 

mse = mean_squared_error(ytest,ytest_pridiction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,ytest_pridiction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,ytest_pridiction)
print(f"r2scor is : {r2scor}")


# # Ridge Regression**
# 1. Ridge regression is a method of estimating the coefficients of regression model in
# scenarios where the independent variables are highly correlated. it uses the linear
# regression model with the L2 regularization method.

# # Instantiat Ridge Regression

# In[147]:


ridge_re = Ridge()
ridge_re


# In[148]:


ridge_re_model = ridge_re.fit(xtrain,ytrain)
ridge_re_model


# In[149]:


ytrain_ridge_prediction = ridge_re_model.predict(xtrain)


# In[150]:


### for training

mse = mean_squared_error(ytrain,ytrain_ridge_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,ytrain_ridge_prediction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,ytrain_ridge_prediction)
print(f"r2scor is : {r2scor}")


# In[151]:


ytest_ridge_predction = ridge_re_model.predict(xtest)


# In[152]:


#### for testing 

mse = mean_squared_error(ytest,ytest_ridge_predction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,ytest_ridge_predction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,ytest_ridge_predction)
print(f"r2scor is : {r2scor}")


# # Algoritham  -02  Decion tree Regressor

# In[154]:


dt_reg = DecisionTreeRegressor()     #### DecisiontreeRegressor insteciate
dt_reg


# In[155]:


dt_reg_model = dt_reg.fit(xtrain,ytrain)                 ###### train the model 
dt_reg_model


# In[156]:


dt_ytrain_predict = dt_reg_model.predict(xtrain)     #### predict model for training data 


# # Evaluation Matrics for Decission tree Regressior

# In[157]:


### for training

mse = mean_squared_error(ytrain,dt_ytrain_predict)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,dt_ytrain_predict)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,dt_ytrain_predict)
print(f"r2scor is : {r2scor}")


# In[158]:


dt_ytest_predition = dt_reg_model.predict(xtest)         #### prediction of model for testing data 


# In[159]:


#### for testing 

mse = mean_squared_error(ytest,dt_ytest_predition)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,dt_ytest_predition)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,dt_ytest_predition)
print(f"r2scor is : {r2scor}")


# In[160]:


plt.figure(figsize=(10,10))
plot_tree(dt_reg_model, feature_names=df.columns, class_names=["No","Yes"],filled=True)
plt.savefig("dt_clf")


# In[161]:


dt_reg_model.feature_importances_ 


# # Decision Tree with Hyperparamiters Tunning

# In[162]:


hyperparameters = {
                   "criterion" : ['squared_error','absolute_error','poisson','friedman_mse'],
                   "min_samples_split" : np.arange(1,15),
                   "min_samples_leaf" : np.arange(1,15),
                   "max_depth" : np.arange(1,10)
}


# In[163]:


gscv  = GridSearchCV(dt_reg_model, hyperparameters, cv = 5)           ####3 crossvalidation with GridSearchcv
gscv


# In[164]:


gscv.estimator    


# # Model Training after cross validation

# In[165]:


gscv_dt_reg_model = gscv.fit(xtrain,ytrain)            ##### model training after coss validatin 
gscv_dt_reg_model


# In[166]:


gscv_dt_reg_model.best_estimator_    


# In[167]:


dt_reg_hyp = DecisionTreeRegressor(criterion='absolute_error', max_depth=9,
                      min_samples_split=8)                                    #### hyperparameter best estimetor 
dt_reg_hyp


# In[168]:


dt_reg_hyp_model = dt_reg_hyp.fit(xtrain,ytrain)      ##### train model with best estimators
dt_reg_hyp_model


# # prediction on training

# In[169]:


dt_hyp_ytrain_pred = dt_reg_hyp_model.predict(xtrain)           ##### predict model 


# # Model Evaluation

# In[170]:


### for training

mse = mean_squared_error(ytrain,dt_hyp_ytrain_pred)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,dt_hyp_ytrain_pred)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,dt_hyp_ytrain_pred)
print(f"r2scor is : {r2scor}")


# # For Testing

# In[173]:


dt_hyp_ytest_pred = dt_reg_hyp_model.predict(xtest)


# In[174]:


#### for testing 

mse = mean_squared_error(ytest,dt_hyp_ytest_pred)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,dt_hyp_ytest_pred)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,dt_hyp_ytest_pred)
print(f"r2scor is : {r2scor}")


# In[175]:


s1= pd.Series(data= dt_reg_model.feature_importances_, index=x.columns)
s1


# In[176]:


s1.plot(kind = "bar")


# In[177]:


df1


# # Algoritham - 3 RandomForest regressior

# In[179]:


rafo_reg = RandomForestRegressor()
rafo_reg


# In[180]:


rafo_reg_model = rafo_reg.fit(xtrain,ytrain)
rafo_reg_model


# In[181]:


rafo_reg_predict = rafo_reg_model.predict(xtrain)      #### prediction on train data in randomforest algoritham


# In[182]:


### for training

mse = mean_squared_error(ytrain,rafo_reg_predict)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,rafo_reg_predict)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,rafo_reg_predict)
print(f"r2scor is : {r2scor}")


# In[183]:


rafo_reg_test_predict = rafo_reg_model.predict(xtest)        ##### prdiction on testing data 


# In[184]:


#### for testing 

mse = mean_squared_error(ytest,rafo_reg_test_predict)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,rafo_reg_test_predict)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,rafo_reg_test_predict)
print(f"r2scor is : {r2scor}")


# # RandomFForest with Hyperparameter Tunning

# In[185]:


hyperparamters = {                                           ##### hyperparamitores for random forest 
                    "criterion" : ["mse","mae"],              # for regression : "mse","mae"
                    "min_samples_split" : np.arange(1,15),
                    "min_samples_leaf" : np.arange(1,15),
                    "max_depth" : np.arange(1,10),
                    "n_estimators" : np.arange(10,50) #number of decision treses should get form
                 }


# In[186]:


rscv_rf_reg = RandomizedSearchCV(rafo_reg_model,hyperparamters,cv = 5)
rscv_rf_reg


# In[187]:


rscv_rf_model = rscv_rf_reg.fit(xtrain,ytrain)
rscv_rf_model


# In[240]:


rscv_rf_train_pred = rscv_rf_model.predict(xtrain)


# In[241]:


### for training

mse = mean_squared_error(ytrain,rscv_rf_train_pred )
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,rscv_rf_train_pred )
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,rscv_rf_train_pred )
print(f"r2scor is : {r2scor}")


# In[243]:


rscv_rf_test_pred = rscv_rf_model.predict(xtest)


# In[244]:


#### for testing 

mse = mean_squared_error(ytest,rscv_rf_test_pred)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,rscv_rf_test_pred)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,rscv_rf_test_pred)
print(f"r2scor is : {r2scor}")


# In[188]:


rscv_rf_model.best_estimator_


# In[189]:


rscv_rf_reg = RandomForestRegressor(criterion='mse', max_depth=12, min_samples_leaf=11,
                      min_samples_split=5, n_estimators=42)


# In[190]:


rscv_rf_reg_model = rscv_rf_reg.fit(xtrain,ytrain)
rscv_rf_reg_model


# In[191]:


ytrain_pred_rf = rscv_rf_reg_model.predict(xtrain)
ytrain_pred_rf


# In[192]:


### for training

mse = mean_squared_error(ytrain,ytrain_pred_rf)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,ytrain_pred_rf)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,ytrain_pred_rf)
print(f"r2scor is : {r2scor}")


# In[238]:



ytest_pred_rf = rscv_rf_reg_model.predict(xtest)
ytest_pred_rf


# In[239]:


#### for testing 

mse = mean_squared_error(ytest,ytest_pred_rf)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,ytest_pred_rf)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,ytest_pred_rf)
print(f"r2scor is : {r2scor}")


# # Algoritham - 4 Adaboost Regressor

# In[196]:


adb_reg = AdaBoostRegressor()
adb_reg   


# In[197]:


adb_reg_model = adb_reg.fit(xtrain, ytrain)
adb_reg_model                                           ###### train the model with xtrain and ytrain


# In[198]:


adb_ytrain_reg_prediction = adb_reg_model.predict(xtrain)            #### prediction of a model


# # Model Evaluation for Adaboost Regressor algoritham

# In[199]:


### for training

mse = mean_squared_error(ytrain,adb_ytrain_reg_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,adb_ytrain_reg_prediction )
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,adb_ytrain_reg_prediction)
print(f"r2scor is : {r2scor}")


# In[200]:


adb_ytest_reg_prediction = adb_reg_model.predict(xtest)


# In[201]:


#### for testing 

mse = mean_squared_error(ytest,adb_ytest_reg_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,adb_ytest_reg_prediction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,adb_ytest_reg_prediction)
print(f"r2scor is : {r2scor}")


# # Adaboost with Hyperparamiters Tunning

# In[203]:


hyp = {
    "n_estimators" : np.arange(10,100),
    "learning_rate" : [0.1,0.01,0.001,1]
   
}


# In[204]:


rscv_adb_reg = RandomizedSearchCV(adb_reg_model, hyp, cv = 5)
rscv_adb_reg 


# In[205]:


rscv_adb_reg = rscv_adb_reg.fit(xtrain,ytrain)
rscv_adb_reg


# In[206]:


rscv_adb_reg.best_estimator_


# In[207]:


rscv_adb_reg = AdaBoostRegressor(learning_rate=0.1, n_estimators=88)
rscv_adb_reg


# In[208]:


rscv_adb_reg_model = rscv_adb_reg.fit(xtrain,ytrain)
rscv_adb_reg_model


# In[209]:


rscv_ytrain_adb = rscv_adb_reg_model.predict(xtrain)
rscv_ytrain_adb


# In[210]:


### for training

mse = mean_squared_error(ytrain,rscv_ytrain_adb)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,rscv_ytrain_adb)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,rscv_ytrain_adb)
print(f"r2scor is : {r2scor}")


# In[211]:


rscv_ytest_adb = rscv_adb_reg_model.predict(xtest)


# In[212]:


#### for testing 

mse = mean_squared_error(ytest,rscv_ytest_adb)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,rscv_ytest_adb)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,rscv_ytest_adb)
print(f"r2scor is : {r2scor}")


# # Algoritham - 5 GradiantBoosting Regressor algoritam

# In[213]:


gra_reg = GradientBoostingRegressor()
gra_reg


# In[214]:


gra_reg_model = gra_reg.fit(xtrain,ytrain)
gra_reg_model


# In[215]:


gra_ytrain_reg_prediction = gra_reg_model.predict(xtrain)


# In[216]:


### for training

mse = mean_squared_error(ytrain,gra_ytrain_reg_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,gra_ytrain_reg_prediction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,gra_ytrain_reg_prediction)
print(f"r2scor is : {r2scor}")


# In[217]:


gra_ytest_reg_prediction = gra_reg_model.predict(xtest)


# In[218]:


#### for testing 

mse = mean_squared_error(ytest,gra_ytest_reg_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,gra_ytest_reg_prediction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,gra_ytest_reg_prediction)
print(f"r2scor is : {r2scor}")


# In[ ]:





# # GradientBoosting Regressor with Hyperparameter Tunning

# In[219]:


hyperparamiters = {
    "n_estimators" : np.arange(10,100),
    "learning_rate" : [0.1,0.01,0.001,1]
}


# In[223]:


rscv_reg_gra = rscv_gra_reg.fit(xtrain,ytrain)
rscv_reg_gra


# In[224]:


rscv_reg_gra.best_estimator_


# In[225]:


rscv_reg_gra = GradientBoostingRegressor(learning_rate=0.1, n_estimators=10)
rscv_reg_gra


# In[226]:


rscv_reg_gra_model = rscv_reg_gra.fit(xtrain,ytrain)
rscv_reg_gra_model


# In[227]:


ytrain_gra_reg = rscv_reg_gra_model.predict(xtrain)


# In[228]:


### for training

mse = mean_squared_error(ytrain,ytrain_gra_reg)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,ytrain_gra_reg)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,ytrain_gra_reg)
print(f"r2scor is : {r2scor}")


# In[229]:


ytest_gra_reg = rscv_reg_gra_model.predict(xtest)


# In[230]:


#### for testing 

mse = mean_squared_error(ytest,ytest_gra_reg)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,ytest_gra_reg)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,ytest_gra_reg)
print(f"r2scor is : {r2scor}")


# # Algoritham - 6 Support Vactor Regressor Algoritham

# In[231]:


svr_al = SVR()
svr_al


# In[232]:


svr_al_model = svr_al.fit(xtrain,ytrain)
svr_al_model


# In[233]:


svr_ytrain_prediction = svr_al_model.predict(xtrain)


# In[234]:



mse = mean_squared_error(ytrain,svr_ytrain_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytrain,svr_ytrain_prediction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytrain,svr_ytrain_prediction)
print(f"r2scor is : {r2scor}")


# In[235]:


svr_ytest_prediction = svr_al_model.predict(xtest)


# In[236]:


### for testing 

mse = mean_squared_error(ytest,svr_ytest_prediction)
print(f"mean squard error is: {mse}")


mae = mean_absolute_error(ytest,svr_ytest_prediction)
print(f"mean absolute error is : {mae}")

rmse = np.sqrt(mse)
print(f"RMSE Root Mean Squared Error : {rmse}")

r2scor = r2_score(ytest,svr_ytest_prediction)
print(f"r2scor is : {r2scor}")


# In[237]:


import pickle


# In[249]:


with open("gradent_reg_model.pkl","wb") as f:
    pickle.dump(gra_reg_model,f)


# In[250]:


df.tail(1)


# In[251]:


test_data = df1.tail(1)
test_data


# In[252]:


with open("gradent_reg_model.pkl","rb") as f:
    final_model = pickle.load(f)


# In[253]:


final_model.predict(test_data)[0]


# In[254]:


def predict_price(Make, Model, Year, Kilometer, Fuel_Type, Transmission, Location, Color, Owner, Seller_Type, Engine, Max_Power, Max_Torque, Drivetrain, Length, Width, Height, Seating_Capacity, Fuel_Tank_Capacity):
    test_data = pd.DataFrame({
        "Make": [Make],
        "Model": [Model],
        "Year": [Year],
        "Kilometer": [Kilometer],
        "Fuel_Type": [Fuel_Type],
        "Transmission": [Transmission],
        "Location": [Location],
        "Color": [Color],
        "Owner": [Owner],
        "Seller_Type": [Seller_Type],
        "Engine": [Engine],
        "Max_Power": [Max_Power],
        "Max_Torque": [Max_Torque],
        "Drivetrain": [Drivetrain],
        "Length": [Length],
        "Width": [Width],
        "Height": [Height],
        "Seating_Capacity": [Seating_Capacity],
        "Fuel_Tank_Capacity": [Fuel_Tank_Capacity],
    })

    with open("lin_reg.pkl", "rb") as f:
        final_model = pickle.load(f)

    print(f"Predicted Price: {final_model.predict(test_data)[0]}")


# In[259]:





# In[ ]:




