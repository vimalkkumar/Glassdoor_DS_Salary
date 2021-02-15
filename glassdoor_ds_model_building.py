import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def main():
    # Importing the whole dataset
    whole_df = pd.read_csv('Data/gd_ds_salary_cleaned_data.csv')
    
    # Coping the dataset into variable for accessing the relevent features 
    df_salary = copy.deepcopy(whole_df)
    
    # selecting the required features for predicting the average salary
    """
    Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Hourly', 'Employer Provided', 
    'Average Salary', 'State', 'Same State', 'Age', 'Python', 'R', 'AWS', 'Azure', 'Spark', 'Excel', 
    'Job Sample', 'Seniority', 'Desc Length', 'Comp Num'
    """
    # droping rest of the features from the selected dataset    
    df_salary = df_salary.drop(['Job Title', 'Salary Estimate', 'Job Description', 
                                'Company Name', 'Location', 'Headquarters', 'Founded', 
                                'Revenue', 'Competitors', 'Min Salary', 'Max Salary',
                                'Company Name Text'], axis = 1)
    
    # Renaming the columns name for standaridzing the naming convention
    df_salary.rename(columns = {
        'Rating' : 'rating', 
        'Size' : 'size', 
        'Type of ownership' : 'ownership_type', 
        'Industry' : 'industry', 
        'Sector': 'sector', 
        'Hourly' : 'hourly', 
        'Employer Provided' : 'emp_provided', 
        'Average Salary' : 'avg_salary', 
        'State': 'state', 
        'Same State': 'same_state', 
        'Age': 'age', 
        'Python': 'python', 
        'R': 'r', 
        'AWS': 'aws', 
        'Azure': 'azure', 
        'Spark': 'spark', 
        'Excel': 'excel', 
        'Job Sample': 'job_sample', 
        'Seniority': 'seniority', 
        'Desc Length': 'desc_len', 
        'Comp Num': 'comp_num'
        }, inplace = True)
    
    # Geting and Handling the dummy Data
    df_salary = pd.get_dummies(df_salary)
    
    # Spliting the dataset into train and test split
    X = df_salary.drop('avg_salary', axis = 1)
    y = df_salary.avg_salary.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Fitting the Multiple Linear Regression to the Training Set
    ml_regressor = LinearRegression()
    ml_regressor.fit(X_train, y_train)
     
    # Validating using Cross Validation 
    cvs = cross_val_score(estimator = ml_regressor, X = X_train, y = y_train, scoring = 'neg_mean_absolute_error', cv = 5)
    np.mean(cvs)
    
    # Predicting the Test Set Result 
    y_pred = ml_regressor.predict(X_test)
    
    # Building the optimal model (Ordinary Least Squares)
    X_sm = X = sm.add_constant(X)
    regressor_OLS = sm.OLS(y, X_sm).fit()   
    regressor_OLS.summary()
    
    # Implementing the lasso regression 
    ls_regressor = Lasso()
    cvs_lsr = cross_val_score(estimator = ls_regressor, X = X_train, y = y_train, scoring = 'neg_mean_absolute_error', cv = 5)
    np.mean(cvs_lsr)
    
    alpha = []
    error = []
    for i in range(1, 100):
        alpha.append(i/100)
        ls_reg = Lasso(alpha = (i/100))
        error.append(cross_val_score(estimator = ls_reg, X = X_train, y = y_train, scoring = 'neg_mean_absolute_error', cv = 5).mean())
        
    # Ploting the result
    plt.plot(alpha, error)
    plt.xlabel("Alpha")
    plt.ylabel('Error')
    plt.show()
    
    

if __name__ == "__main__":
    main()

