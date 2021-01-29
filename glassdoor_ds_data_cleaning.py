import numpy as np
import pandas as pd

dataset = pd.read_csv('Data/gd_ds_salary_uncleaned_data.csv')

# Removing the first column 
dataset = dataset.drop(['Unnamed: 0'], axis = 1)
"""Salary Adjustment"""
# Removing the rows where salary estime is -1
dataset = dataset[dataset['Salary Estimate'] != '-1']

# Spliting the Salary Estimate into Hourly Provided and Employer Provided Salary features
dataset['Hourly'] = dataset['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
dataset['Employer Provided'] = dataset['Salary Estimate'].apply(lambda x:1 if 'employer provided salary' in x.lower() else 0)

# Removing the Galssdoor est. from the salary estimate feature
salary = dataset['Salary Estimate'].apply(lambda x: x.split('(')[0])

# Removing the K and $ sign from the salary feature
remove_knd = salary.apply(lambda x: x.replace('K', '').replace('$', ''))

# Removing the Hourly provided and Employer provide salary from the salary Estimate feature
remove_hneps = remove_knd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

# Adding the features Min Salary, Max Salary and Average Salary Features
dataset['Min Salary'] = remove_hneps.apply(lambda x: int(x.split('-')[0]))
dataset['Max Salary'] = remove_hneps.apply(lambda x: int(x.split('-')[1]))
dataset['Average Salary'] = (dataset['Min Salary'] + dataset['Max Salary'])/2

# Coverting the hourly wages to annual wages
dataset['Min Salary'] = dataset.apply(lambda x: x['Min Salary']*2 if x['Hourly'] == 1 else x['Min Salary'], axis = 1)
dataset['Max Salary'] = dataset.apply(lambda x: x['Max Salary']*2 if x['Hourly'] == 1 else x['Max Salary'], axis = 1)

# Company Name : We can see that there are ratings are given with the company name so we will remove the ratings from their
dataset['Company Name Text'] = dataset.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else str(x['Company Name'])[:-3], axis = 1)
dataset['Company Name Text'] = dataset['Company Name Text'].apply(lambda x: x.replace('\n', '')) 

# Spliting the state from the location feature also handling the  Los Angeles to CA
dataset['State'] = dataset['Location'].apply(lambda x: x.split(',')[1])
dataset['State'] = dataset['State'].apply(lambda x: x.replace(' Los Angeles', 'CA'))

# Checking the Company headquarteras are same state or not
dataset['Same State'] = dataset.apply(lambda x: 1 if x['Location'] == x['Headquarters'] else 0, axis = 1)

# Company Age at 2021
dataset['Age'] = dataset['Founded'].apply(lambda x: x if x < 1 else 2021 - x)

# Parsing the joc Description like Python, R studio, AWS, Azure, Spark, Excel etc
dataset['Python'] = dataset['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
dataset['Python'].value_counts()

dataset['R'] = dataset['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
dataset['R'].value_counts()

dataset['AWS'] = dataset['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
dataset['AWS'].value_counts()

dataset['Azure'] = dataset['Job Description'].apply(lambda x: 1 if 'azure' in x.lower() else 0)
dataset['Azure'].value_counts()

dataset['Spark'] = dataset['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
dataset['Spark'].value_counts()

dataset['Excel'] = dataset['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
dataset['Excel'].value_counts()

# Job Title and Seniority
def title_extractor(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data emgineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'sr.' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'senior'
    elif 'junior' in title.lower() or 'jr' in title.lower() or 'jr.' in title.lower():
        return 'junior'
    else:
        return 'na'

dataset['Job Sample'] = dataset['Job Title'].apply(title_extractor)
dataset['Job Sample'].value_counts()

dataset['Seniority'] = dataset['Job Title'].apply(seniority)
dataset['Seniority'].value_counts()

# Finding out the job description length also
dataset['Desc Length'] = dataset['Job Description'].apply(lambda x: len(x))

# Checking any competitors are exist or not
dataset['Comp Num'] = dataset['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)


# Exporting the latest CSV File
dataset.to_csv('Data/gd_ds_salary_cleaned_data.csv', index = False)
pd.read_csv('Data/gd_ds_salary_cleaned_data.csv')