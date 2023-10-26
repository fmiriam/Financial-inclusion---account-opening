#!/usr/bin/env python
# coding: utf-8

# # KaggleX Final Project
# 
# 
# 
# ## Financial Inclusion in Africa
# 
# 
# ### Background
# 
# Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and make payments while also helping businesses build up their credit-worthiness and improve their access to loans, insurance, and related services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.
# 
# The objective of this competition is to create a machine learning model to predict which individuals are most likely to have or use a bank account. The models and solutions developed can provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania and Uganda, while providing insights into some of the key factors driving individuals’ financial security.
# 
# ### Data
# 
# Data is sourced from Zindi https://zindi.africa/competitions/financial-inclusion-in-africa/data
# 
# * **Country:** Country of citizenship. 
# * **Year:** Year data was collected
# * **Uniqueid:** Unique ID
# * **Bank_account:** Does respondent have a bank account or not.
# * **Location_type:** Whether the respondent resides in rural or urban area
# * **Cellphone_access:** Whether the respondent has a cellphone or not.
# * **Household_size:** Number of people in a household
# * **Age_of_respondent:** Age of respondant
# * **Gender_of_respondent:** Gender
# * **Relationship_with_head:** How does the respondent relate to the head of house.
# * **Marital_status:** Marital status
# * **Education_level:** The level of education of the respondent.
# * **Job_type:** The type of job the respondent is doing.
# 
# The data contains two CSV files, training and testing set.
# 

# In[1]:


## Load Required libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




# In[2]:


# Load data

train_df = pd.read_csv("Train.csv")
test_df =pd.read_csv("Test.csv")


# In[3]:


# Check the top rows of the dataset
train_df.head()


# In[4]:


# Check the bottom rows of the dataset
train_df.head()


# In[5]:


# Check data size
train_df.shape


# In[6]:


train_df.columns


# The training set contains 23524 records and 13 columns

# In[7]:


train_df.info()


# ### Exploratory data analysis 
# 
# #### Data overview

# In[8]:


# Count of Bank Accounts by Country
country_counts = train_df['country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

fig1 = px.bar(country_counts, x='Country', y='Count', title='Count of Bank Accounts by Country')
fig1.show()


# Rwanda has the highest record of account holders and Uganda the lowest number as per the dataset.
# 
# Rwanda: 8735
# Tanzania: 6620
# Kenya: 6068
# Uganda:2101

# In[9]:


# Count of Bank Accounts by Year
year_counts = train_df['year'].value_counts().reset_index()
year_counts.columns = ['Year', 'Count']

fig2 = px.bar(year_counts, x='Year', y='Count', title='Count of Bank Accounts by Year')
fig2.show()


# #### Demographic Analysis

# In[10]:


# Age Distribution (Histogram)
fig_age = px.histogram(train_df, x='age_of_respondent', title='Age Distribution')
fig_age.show()


# In[11]:


# Gender Distribution (Pie Chart)
gender_counts = train_df['gender_of_respondent'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
fig_gender = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution')
fig_gender.show()


# In[12]:


# Marital Status Distribution (Bar Chart)
marital_status_counts = train_df['marital_status'].value_counts().reset_index()
marital_status_counts.columns = ['Marital Status', 'Count']

fig_marital_status = px.bar(marital_status_counts, x='Marital Status', y='Count', 
                            title='Marital Status Distribution')
fig_marital_status.show()


# In[13]:


# Education Level Distribution (Bar Chart)
education_counts = train_df['education_level'].value_counts().reset_index()
education_counts.columns = ['Education Level', 'Count']

fig_education = px.bar(education_counts, x='Education Level', y='Count', 
                       title='Education Level Distribution')
fig_education.show()


# In[14]:


# Job Type Distribution (Bar Chart)
job_type_counts = train_df['job_type'].value_counts().reset_index()
job_type_counts.columns = ['Job Type', 'Count']

fig_job_type = px.bar(job_type_counts, x='Job Type', y='Count', 
                      title='Job Type Distribution')
fig_job_type.show()


# #### Household Analysis

# In[15]:


# Household Size Distribution (Histogram)
fig_household_size = px.histogram(train_df, x='household_size', title='Household Size Distribution')
fig_household_size.show()


# In[16]:


# Relationship with Head of Household Distribution 
relationship_counts = train_df['relationship_with_head'].value_counts().reset_index()
relationship_counts.columns = ['Relationship Type', 'Count']

fig_relationship = px.bar(relationship_counts, x='Relationship Type', y='Count', 
                           title='Relationship with Head of Household Distribution')
fig_relationship.show()


# #### Financial Inclusion Factors
# 

# In[17]:


# Cellphone Access vs. Bank Accounts (Stacked Bar Chart)
cellphone_bank_counts = train_df.groupby(['cellphone_access', 'bank_account']).size().reset_index(name='Count')

fig_cellphone_access = px.bar(cellphone_bank_counts, 
                              x='cellphone_access', 
                              y='Count', 
                              color='bank_account', 
                              barmode='stack',
                              labels={'cellphone_access': 'Cellphone Access', 'bank_account': 'Bank Account'},
                              title='Cellphone Access vs. Bank Accounts')
fig_cellphone_access.show()


# In[18]:


# Location Type vs. Bank Accounts (Stacked Bar Chart)
location_bank_counts = train_df.groupby(['location_type', 'bank_account']).size().reset_index(name='Count')

fig_location_type = px.bar(location_bank_counts, 
                           x='location_type', 
                           y='Count', 
                           color='bank_account', 
                           barmode='stack',
                           labels={'location_type': 'Location Type', 'bank_account': 'Bank Account'},
                           title='Location Type vs. Bank Accounts')
fig_location_type.show()


# #### Trends and patterns 

# In[19]:


# Age vs. Bank Accounts (Box Plot)
fig_age_vs_bank_account = px.box(train_df, x='bank_account', y='age_of_respondent', 
                                  labels={'bank_account': 'Bank Account Presence', 'age_of_respondent': 'Age'},
                                  title='Age vs. Bank Accounts')
fig_age_vs_bank_account.show()


# In[20]:


# Yearly Trends in Bank Accounts (Line Chart)
yearly_trends = train_df.groupby('year')['bank_account'].value_counts().reset_index(name='Count')

fig_yearly_trends = px.line(yearly_trends, x='year', y='Count', color='bank_account',
                            labels={'year': 'Year', 'Count': 'Count of Bank Accounts'},
                            title='Yearly Trends in Bank Accounts')
fig_yearly_trends.show()


# In[21]:


# Selecting numerical variables
numerical_vars = ['year', 'household_size', 'age_of_respondent']

# Creating a correlation matrix
correlation_matrix = train_df[numerical_vars].corr()

# Creating a correlation heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap for Numerical Variables')
plt.show()


# In[22]:


# Bank Account Distribution (Bar Chart)
plt.figure(figsize=(6, 4))
sns.countplot(x='bank_account', data=train_df)
plt.xlabel('Bank Account Presence')
plt.ylabel('Count')
plt.title('Bank Account Distribution ')
plt.show()


# In[23]:


# Bank Account Distribution (Pie Chart)
bank_account_counts = train_df['bank_account'].value_counts().reset_index()
bank_account_counts.columns = ['Bank Account', 'Count']

fig_bank_account_pie = px.pie(bank_account_counts, names='Bank Account', values='Count', 
                               title='Bank Account Distribution ')
fig_bank_account_pie.show()


# In[24]:


# Selecting features and target variable
features = ['country', 'year', 'location_type', 'cellphone_access', 'household_size',
            'age_of_respondent', 'gender_of_respondent', 'relationship_with_head',
            'marital_status', 'education_level', 'job_type']

target = 'bank_account'

# Splitting data into features and target variable
X = train_df[features]
y = train_df[target]

# One-hot encoding categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))
encoded_feature_names = encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

# Concatenating one-hot encoded features with numerical features
X_final = pd.concat([X_encoded_df, X.select_dtypes(include=['int64'])], axis=1)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)




# ## Modeling

# ### Logistic Regression

# In[25]:


# Training and evaluating Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)

# Confusion Matrix and Plot for Logistic Regression
logreg_cm = confusion_matrix(y_test, logreg_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(logreg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# ### Random Forest

# In[26]:


# Training and evaluating Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Confusion Matrix and Plot for Random Forest Classifier
rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()


# ### Gradient Boosting Classifier

# In[27]:


# Training and evaluating Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

# Confusion Matrix and Plot for Gradient Boosting Classifier
gb_cm = confusion_matrix(y_test, gb_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Gradient Boosting Confusion Matrix')
plt.show()


# In[28]:


# Print accuracies
print("Logistic Regression Accuracy:", logreg_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)

# Print classification reports
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))


# *  Logistic Regression model has the high precision for the positive class (Yes) with a value of 0.67, meaning that when it predicts a positive case, it is correct 67% of the time. However, its recall for the positive class is quite low at 0.29, indicating that it misses a substantial number of actual positive cases.
# 
# * Random Forest model has a higher recall for the positive class (0.39) compared to the Logistic Regression model, meaning that it is better at identifying actual positive cases. However, its precision for the positive class is lower at 0.48, indicating that there are more false positives.
# 
# * Gradient Boosting model strikes a balance between precision (0.71) and recall (0.34) for the positive class, making it better at correctly identifying positive cases while also having a reasonable precision.
# 
# Considering the trade-off between precision and recall, the Gradient Boosting Classifier appears to be the best model among the three for predicting the likelihood of opening a bank account in this scenario. It achieves a good balance between identifying positive cases and minimizing false positives. Additionally, it has the highest overall accuracy (0.89) among the three models.

# ## Test Validation Using the testing Set of data

# Fit Gradient boosting model on the test data

# In[29]:


# Selecting features for the test data
test_features = ['country', 'year', 'location_type', 'cellphone_access', 'household_size',
                 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head',
                 'marital_status', 'education_level', 'job_type']

# Extracting test features
X_test_data = test_df[test_features]

# One-hot encoding categorical features for the test data
X_test_encoded = encoder.transform(X_test_data.select_dtypes(include=['object']))
encoded_test_feature_names = encoder.get_feature_names_out(X_test_data.select_dtypes(include=['object']).columns)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_test_feature_names)

# Concatenating one-hot encoded features with numerical features for the test data
X_test_final = pd.concat([X_test_encoded_df, X_test_data.select_dtypes(include=['int64'])], axis=1)

# Making predictions using the trained Gradient Boosting Classifier
gb_predictions = gb_clf.predict(X_test_final)

# Adding predictions to the test data
test_df['predicted_bank_account'] = gb_predictions

# Displaying the test data with predictions
print(test_df[['uniqueid', 'predicted_bank_account']])


# In[30]:


test_df.predicted_bank_account.value_counts()


# In[ ]:




