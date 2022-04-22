#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import re
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import FreqDist

from wordcloud import WordCloud

warnings.filterwarnings("ignore")
sb.set()


# In[2]:


'''
plot_cloud()
This function is used to display the word cloud of the words used in a given WordCloud object. 

@param word_cloud is the WordCloud object that we intend to display.
'''

def plot_cloud(word_cloud):
    plt.figure(figsize = (40,30))
    plt.imshow(word_cloud)
    plt.axis("off")


# In[3]:


'''
generateTopWords()

This function is used to generate and display the most common words in ascending order.
Nested within uses plot_cloud(). Refer to plot_cloud() for more information.

@param k is the Category that we intend to analyse.
@param v is the DataFrame associated with k.
@param num is the maximum limit for the most common words that we want displayed.
'''

def generateTopWords(k, v, num):
    
    # Define a TitleDF for NLP Analysis
    titleDF = pd.DataFrame(v[['Title', 'Views']])
    print("This is the first 5 entries of the dataframe generated for the category " + k)
    display(titleDF.head())
    
    # Creating a SnowballStemmer object to stem the different words in the title for each video in the dataframe.
    j = 0
    Snow_Stemmer = SnowballStemmer(language = 'english')
    for i in titleDF['Title']:
        title = re.findall(r'[^\W_]+', i.lower())
        wordList = [w for w in title if w not in stopwords.words("english")]
        
        stemmedList = []
        for word in wordList:
            stemmedWord = Snow_Stemmer.stem(word)
            stemmedList.append(stemmedWord)
        
        titleDF['Title'].iloc[j] = stemmedList
        j += 1
        
    
    # Creating a list and appending all the words that have been stemmed prior.
    WordList = []
    for i in titleDF['Title']:
        for word in range(len(i)):
            WordList.append(i[word])

    # Concatenating all the words in the list into one long chained string, separated by one whitespace.
    string = ' '.join(WordList)
    
    # Generating the wordcloud of the words in the list.
    wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, background_color = 'salmon', colormap = 'Pastel1', collocations = False).generate(string)
    plot_cloud(wordcloud)
    
    # Generating the frequency distribution of the Top "Num" words
    data = FreqDist(WordList)
    topWords = pd.DataFrame(data.most_common(num), columns = ["Word", "Occurrence"])
    display(topWords.head(20))
    
    return topWords


# In[4]:


'''
calUpperLowerBound()
This function is used to calculate the upper (1.5 * IQR + Q3) and lower (Q1 - 1.5 * IQR) bound of a numerical dataset.

@param DF is the numerical dataset that we want to calculate.

@return upper is the upper bound of the dataset.
@return lower is the lower bound of the dataset.
'''

def calUpperLowerBound(DF):
    IQR = DF.quantile(0.75) - DF.quantile(0.25)
    
    upper = DF.quantile(0.75) + (1.5 * IQR)
    lower = DF.quantile(0.25) - (1.5 * IQR)
    
    return upper, lower


# In[5]:


'''
calFPR()
This function is used to calculate the False Positive Rate of a Prediction.

@param y is the dataset of the response.
@param y_pred is the predicted dataset of the response.

@return this is the calculated False Positive Rate.
'''

def calFPR(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    
    return matrix[0][1] / (matrix[0][0] + matrix[0][1])


# In[6]:


'''
calTPR()
This function is used to calculate the True Positive Rate of a Prediction.

@param y is the dataset of the response.
@param y_pred is the predicted dataset of the response.

@return this is the calculated True Positive Rate.
'''

def calTPR(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    
    return matrix[1][1] / (matrix[1][0] + matrix[1][1])


# In[7]:


'''
calFNR()
This function is used to calculate the False Negative Rate of a Prediction.

@param y is the dataset of the response.
@param y_pred is the predicted dataset of the response.

@return this is the calculated False Negative Rate.
'''

def calFNR(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    
    return matrix[1][0] / (matrix[1][0] + matrix[1][1])


# In[8]:


'''
calTNR()
This function is used to calculate the True Negative Rate of a Prediction.

@param y is the dataset of the response.
@param y_pred is the predicted dataset of the response.

@return this is the calculated True Negative Rate.
'''

def calTNR(y, y_pred):
    matrix = confusion_matrix(y, y_pred)
    
    return matrix[0][0] / (matrix[0][0] + matrix[0][1])


# In[9]:


"""
LRFuncByCat()
This function is used to generate an entire Linear Regression Model specific to each category. It includes the 
train-test splitting of the dataset, the summary statistics of the data, plots out the associated figures, 
and prints out the goodness of fit of the given model.

@param k is the Category that we intend to analyse.
@param v is the DataFrame associated with k.
"""

def LRFuncByCat(k, v, random_state = 42):
    x = pd.DataFrame(v[['Subscribers', 'Length', 'LengthOfTitle']])
    y = pd.DataFrame(v['Views'])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = random_state)
    
    print("This is the Train-Test Split for the data points categorised under " + k)
    print("Train Set :", y_train.shape, x_train.shape)
    print("Test Set  :", y_test.shape, x_test.shape)
    
    print()
    print("This is the Summary Statistics of our Train Dataset:")
    display(y_train.describe())
    display(x_train.describe())
    
    print("\nBelow plots the Boxplot, Histogram, and Violinplot of Views")
    plt.figure()
    f, axes = plt.subplots(1, 3, figsize=(24, 6))
    sb.boxplot(data = y_train, orient = "h", ax = axes[0])
    sb.histplot(data = y_train, ax = axes[1])
    sb.violinplot(data = y_train, orient = "h", ax = axes[2])
    plt.show()
    
    print("\nBelow plots the Boxplot, Histogram, and Violinplot of the following variables (in order): ")
    print("(1) Subscribers\n(2) Released\n(3) Length\n(4) LengthOfTitle")
    plt.figure()
    f, axes = plt.subplots(3, 3, figsize = (18, 12))
    f.subplots_adjust(hspace = 0.5, wspace = 0.125)
    count = 0
    for var in x_train:
        sb.boxplot(data = pd.DataFrame(x_train[var]), orient = "h", ax = axes[count, 0])
        sb.histplot(data = pd.DataFrame(x_train[var]), ax = axes[count, 1])
        sb.violinplot(data = pd.DataFrame(x_train[var]), orient = "h", ax = axes[count, 2])
        count += 1
    plt.show()
    
    print("\nHere displays the heatmap and the Pairplot of all the variables in our Training Data")
    trainDF = pd.concat([y_train, x_train], axis = 1).reindex(y_train.index)
    
    plt.figure()
    f = plt.figure(figsize = (12, 8))
    sb.heatmap(trainDF.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f")
    plt.show()
    
    sb.pairplot(data = trainDF)
    plt.show()
    
    print("\nThis is the Linear Regression Model based on videos that are categorised under " + k)
    linreg = LinearRegression()         # create the linear regression object
    linreg.fit(x_train, y_train)        # train the linear regression model

    # Coefficients of the Linear Regression line
    print('Intercept of Regression \t: b = ', linreg.intercept_)
    print('Coefficients of Regression \t: a = ', linreg.coef_)
    print()

    # Print the Coefficients against Predictors
    pd.DataFrame(list(zip(x_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"])

    y_train_pred = linreg.predict(x_train)
    y_test_pred = linreg.predict(x_test)

    # Plot the Predictions vs the True values
    plt.figure()
    f, axes = plt.subplots(1, 2, figsize = (24, 12))
    axes[0].scatter(y_train, y_train_pred, color = "blue")
    axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
    axes[0].set_xlabel("True values of the Response Variable (Train)")
    axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
    axes[1].scatter(y_test, y_test_pred, color = "green")
    axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
    axes[1].set_xlabel("True values of the Response Variable (Test)")
    axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
    plt.show()
    
    print()
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(x_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print("Root Mean Squared Error (RMSE) \t:", mean_squared_error(y_train, y_train_pred, squared = False))
    print()

    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(x_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
    print("Root Mean Squared Error (RMSE) \t:", mean_squared_error(y_test, y_test_pred, squared = False))
    print()


# In[10]:


"""
LRFuncInGen()
This function is used to generate an entire Linear Regression Model in general, narrowing to one predictor. 
It includes the train-test splitting of the dataset, the summary statistics of the data, plots out the associated 
figures, and prints out the goodness of fit of the given model.

@param DF is the dataframe that we intend to analyse.
@param predictor is the predictor variable used.
@param response is the response variable that will be predicted.
@param random_state is used to change the outcome of the train test split.
"""

def LRFuncInGen(DF, predictor, response, random_state):
    subscribers = pd.DataFrame(DF[predictor])
    views = pd.DataFrame(DF[response])
    
    X_train, X_test, y_train, y_test = train_test_split(subscribers, views, test_size = 0.20, random_state = random_state)
    print("This is the Train-Test Split for the dataset:")
    print("Train Set :", y_train.shape, X_train.shape)
    print("Test Set  :", y_test.shape, X_test.shape)
    
    print()
    print("This is the Summary Statistics of our Train Dataset:")
    display(y_train.describe())
    display(X_train.describe())
    
    print("\nBelow plots the Boxplot, Histogram, and Violinplot of " + response)
    plt.figure()
    f, axes = plt.subplots(1, 3, figsize = (24, 6))
    sb.boxplot(data = y_train, orient = "h", ax = axes[0])
    sb.histplot(data = y_train, ax = axes[1])
    sb.violinplot(data = y_train, orient = "h", ax = axes[2])
    plt.show()
    
    print("\nBelow plots the Boxplot, Histogram, and Violinplot of " + predictor)
    plt.figure()
    f, axes = plt.subplots(1, 3, figsize = (24, 6))
    sb.boxplot(data = X_train, orient = "h", ax = axes[0])
    sb.histplot(data = X_train, ax = axes[1])
    sb.violinplot(data = X_train, orient = "h", ax = axes[2])
    plt.show()
    
    print("\nHere displays the heatmap and the Pairplot of all the variables in our Training Data")
    trainDF = pd.concat([y_train, X_train], axis = 1).reindex(y_train.index)
    
    plt.figure()
    f = plt.figure(figsize = (12, 8))
    sb.heatmap(trainDF.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f")
    plt.show()
    
    sb.pairplot(data = trainDF)
    plt.show()
    
    print("\nThis is the Linear Regression Model predicting Views based on Subsribers in general")
    linreg = LinearRegression()         # create the linear regression object
    linreg.fit(X_train, y_train)        # train the linear regression model
    
    # Coefficients of the Linear Regression line
    print('Intercept of Regression \t: b = ', linreg.intercept_)
    print('Coefficients of Regression \t: a = ', linreg.coef_)
    print()

    # Print the Coefficients against Predictors
    pd.DataFrame(list(zip(X_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"])

    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    # Plot the Predictions vs the True values
    plt.figure()
    f, axes = plt.subplots(1, 2, figsize = (24, 12))
    axes[0].scatter(y_train, y_train_pred, color = "blue")
    axes[0].plot(y_train, y_train, 'w-', linewidth = 1)
    axes[0].set_xlabel("True values of the Response Variable (Train)")
    axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
    axes[1].scatter(y_test, y_test_pred, color = "green")
    axes[1].plot(y_test, y_test, 'w-', linewidth = 1)
    axes[1].set_xlabel("True values of the Response Variable (Test)")
    axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
    plt.show()
    
    print()
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print("Root Mean Squared Error (RMSE) \t:", mean_squared_error(y_train, y_train_pred, squared = False))
    print()

    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
    print("Root Mean Squared Error (RMSE) \t:", mean_squared_error(y_test, y_test_pred, squared = False))
    print()


# In[11]:


"""
DTFunc()
This function is used to generate an entire Decision Tree Classification Model. It includes the train-test
splitting of the dataset, plots out the associated figures, and prints out the goodness of fit of the given model.

Nested within uses calFPR(), calFNR(), calTPR(), calTNR(). Refer to their respective cells for more information.

@param k is the Category that we intend to analyse.
@param v is the DataFrame associated with k.
@param topWords is the list of most frequent words that appear in Titles.
"""

def DTFunc(k, v, topWords):
    
    # Creating a new column that includes the data generated from generateTopWords().
    v['hasTop20'] = 0
    for i in range(len(v)):
        v['hasTop20'].iloc[i] = 0
        for j in topWords['Word']:
            if j in v['Title'].iloc[i]:
                v['hasTop20'].iloc[i] = 1
                break
    
    # Splitting the dataset into a Training dataset and a Test dataset.
    X = pd.DataFrame(v['Views'])
    y = pd.DataFrame(v['hasTop20'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
    print("This is the Train-Test Split for the data points categorised under " + k)
    print("Train Set :", y_train.shape, X_train.shape)
    print("Test Set  :", y_test.shape, X_test.shape)

    # Decision Tree using Train Data
    dectree = DecisionTreeClassifier(max_depth = 3)  # create the decision tree object with max depth 3
    dectree.fit(X_train, y_train)                    # train the decision tree model

    # Predict hasTop20 values corresponding to Views
    y_train_pred = dectree.predict(X_train)
    y_test_pred = dectree.predict(X_test)
    
    # Create a joint dataframe by concatenating Views and hasTop20
    trainDF = pd.concat([X, y], axis = 1).reindex(X.index)

    # Joint Swarmplot of Views Train against hasTop20 Train
    print("\nThis is the swarmplot of Views Train against hasTop20 Train.")
    f = plt.figure(figsize=(24, 6))
    sb.swarmplot(x = "Views", y = "hasTop20", data = trainDF, orient = "h")
    plt.show()
    
    # Plotting the Decision Tree
    print("\nThis is the decision tree after training.")
    f = plt.figure(figsize = (12,12))
    plot_tree(dectree, filled = True, rounded = True, feature_names = X_train.columns, class_names = ["no top 20","has top 20"])
    plt.show()
    
    # Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Classification Accuracy \t:", dectree.score(X_train, y_train))
    print("False Negative Rate\t\t:", calFNR(y_train, y_train_pred))
    print("True Negative Rate\t\t:", calTNR(y_train, y_train_pred))
    print()
    print("False Positive Rate\t\t:", calFPR(y_train, y_train_pred))
    print("True Positive Rate\t\t:", calTPR(y_train, y_train_pred))
    print()

    # Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Classification Accuracy \t:", dectree.score(X_test, y_test))
    print("False Negative Rate\t\t:", calFNR(y_test, y_test_pred))
    print("True Negative Rate\t\t:", calTNR(y_test, y_test_pred))
    print()
    print("False Positive Rate\t\t:", calFPR(y_test, y_test_pred))
    print("True Positive Rate\t\t:", calTPR(y_test, y_test_pred))
    print()

    # Plot the Confusion Matrix for Train and Test
    print("\nThis is the corresponding Confusion Matrix for both Train and Test.")
    print("Train is on the left. Test is on the right.")
    plt.figure()
    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    sb.heatmap(confusion_matrix(y_train, y_train_pred),
               annot = True, fmt = ".0f", annot_kws = {"size": 18}, ax = axes[0])
    sb.heatmap(confusion_matrix(y_test, y_test_pred), 
               annot = True, fmt = ".0f", annot_kws = {"size": 18}, ax = axes[1])
    plt.show()
    


# In[12]:


"""
RFFunc()
This function is used to generate an entire Random Forest Classification Model. It includes the train-test
splitting of the dataset, plots out the associated figures, and prints out the goodness of fit of the given model.

@param DF is the dataframe that we intend to analyse.
@param categoryList is the list of categories that we are analysing.
"""

def RFFunc(DF, categoryList):
    # Extract Response and Predictors
    y = pd.DataFrame(DF['Category'].astype('category'))
    X = pd.DataFrame(DF['Subscribers']) 
    
    # Split the Dataset into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 14)
    print("This is the Train-Test Split for the data points:")
    print("Train Set :", y_train.shape, X_train.shape)
    print("Test Set  :", y_test.shape, X_test.shape)
    
    # Draw the distribution of Response
    plt.figure()
    sb.catplot(y = "Category", data = y_train, kind = "count")
    plt.show()
    
    # Relationship between Response and the Predictors
    trainDF = pd.concat([y_train, X_train], axis = 1).reindex(y_train.index)

    f = plt.figure(figsize=(18, 42))
    sb.boxplot(x = "Subscribers", y = "Category", data = trainDF, orient = "h")
    plt.show()
    
    # Random Forest using Train Data
    rforest = RandomForestClassifier(n_estimators = 100, max_depth = 7)  # create the object
    rforest.fit(X_train, y_train.values.ravel())                         # train the model

    # Predict Response corresponding to Predictors
    y_train_pred = rforest.predict(X_train)
    y_test_pred = rforest.predict(X_test)
    
    # Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Classification Accuracy \t:", rforest.score(X_train, y_train))
    print()

    # Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Classification Accuracy \t:", rforest.score(X_test, y_test))
    print()

    # Plot the Confusion Matrix for Train and Test
    f, axes = plt.subplots(2, 1, figsize=(12, 24))
    sb.heatmap(confusion_matrix(y_train, y_train_pred), annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])
    sb.heatmap(confusion_matrix(y_test, y_test_pred), annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])
    
    matrix = confusion_matrix(y_test, y_test_pred)
    matrix = matrix.astype('float') / matrix.sum(axis = 1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize = (16,7))
    sb.set(font_scale = 1.4)
    sb.heatmap(matrix, annot = True, annot_kws = {'size':10}, cmap = plt.cm.Greens, linewidths = 0.2)

    # Add labels to the plot
    class_names = categoryList
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation = 25)
    plt.yticks(tick_marks2, class_names, rotation = 0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()

    print(classification_report(y_test, y_test_pred))


# In[ ]:




