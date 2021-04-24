# AI-106394
# GROUP MEMBERS
Student Id | Name
------------ | -------------
63229| **Bushra Liaquat Khan** (Group Leader) 
63178| Kanza Ahmad
63191| Riba Zainab

# How we achieved each task:
Our group was unable to understand what the errors were but we researched and then were able to work on the project.
We took the code from GCR,the datasets from kaggle and used it to apply the techniques on the three given matrices i.e 5x5 , 7x7 and 9x9 along with two different filters which are same for the other two sizes of matrices.
# Filter #1:
          ([[1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]])

# Filter #2:
          ([[1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]])

# Contribution of each member:
We had 6 techniques so we divided two each techniques to one member and assigned all three matrices with the two filters aswell.

# 1:Bushra Liaquat Khan :
I've applied **MultinomialNB** and **K-Nearest Neighbours** on 5x5, 7x7, 9x9 matrices with two filters which are mentioned above.
I have attached all of the outputs of the code down below. I guided my team mates about research on the techniques they have used and along the documentation work I've tried the best to make our project self explainatory.

# 2: Riba Zainab:
I've worked on **NaiveBayes Algorithms** which are **Bernoulli** and **GaussianNB** on 5x5,7x7 and 9x9 matrices with two filters which are mentioned above.
I uploaded all the test models of my group members on kaggle and due to internet failure I'm having issues on attaching .png files.

# 3: Kanza Ahmad: 
I've applied **Random Forest** and **Support Vector Classifier SVC** on 5x5,7x7 and 9x9 matrices with two filters which are mentioned above.


# Description of techniques:
All of following machine learning algorithms are supervised machine learning algorithm and now you would ask “**what exactly is supervised machine learning**?” and to answer that I will give you a real life example,
Suppose we have child and we are it’s supervisor i.e. parent and we want the child to learn how to identify different animals.We want our child to learn how a cat looks like. What we will do inorder to make the child learn what a cat looks like, we will show the child several pictures of the cat and few others of the different animals so the child can be able to identify the difference between the cat. While showing the child pictures of the cat we will shout “**cat**!” (when actually a picture of a cat appears)and when it’s some another animal we shout “**it’s not a cat!**” so after showing several pictures then we ask the the child  and they will correctly tell and will be able to identify what the picture is. This is **Supervised Learning.**

# K-Nearest Neighbours:
**K-Nearest Neighbours (KNN)** algorithm is an easy to implement **supervised** machine learning algorithm that can be used to solve both regression and classification problems. What this technique do is that it collects similar kind of data that exist in a dataset and combines them. 
**"Birds of a feather flock together"**
We can take the example of **Netflix** as if we have watched a movie whose genre was "animated" so from next time the suggestion box will show us the movies or TV series of the same genre.


**Naive Bayes** are a group of **supervised** machine learning classification algorithms based on the Bayes theorem. It is a simple classification technique, but has high functionality. Why they are called **naive** becausue two assumptions are made in this algorithm.
1)First assumption is characterized in that between the mutually independent.
2)Second assumption is that each feature is equally important.
So, when we are using this algorithm for text classification,this looks very ungenuous with the assumptions but it can give good results.

# MultinomialNB:
One of the most famous applications of machine laerning is the analysis of categorical data, specially text data. Multinomial Naive Bayes uses term frequency i.e. the number of times a given term appears in a document.After normalization, term frequency can be used to compute maximum likelihood estimates based on the training data to estimate the conditional probability.

# Bernoulli:
Bernoulli model considers only text words appear regardless of whether the number of occurrences, which brings great convenience in the preparation of the final naive Bayes classifier function, to get a few lines of output.

# GaussianNB:
**Gaussian Naive Bayes** is a form of Naive Bayes that follows Gaussian normal distribution and supports continuous data.
It is used with continuous data,an assumption often taken is that the continuous values associated with each class are distributed according to a Gaussian distribution. Sometimes we assume:
1) Y is independent 
2) X is independent 
3) or both.
Gaussian Naive Bayes supports continuous valued features and models each as conforming to a Gaussian distribution.


# SVC:

# Random Forest:

# How you achieved each task and 6 techniques:

# Cross validation score of each 6 techniques: 

# **5x5 Filter 1:**



# **5x5 Filter 2:**

![WhatsApp Image 2021-04-24 at 9 56 51 PM (1)](https://user-images.githubusercontent.com/74513063/115974748-52dff680-a578-11eb-9feb-91f7e97c4d66.jpeg)

# **7x7 Filter 1:**



# **7x7 Filter 2:**

![WhatsApp Image 2021-04-25 at 2 25 13 AM (1)](https://user-images.githubusercontent.com/74513063/115974752-68552080-a578-11eb-9b09-95e795c16e27.jpeg)

# **9x9 Filter 1:**



# **9x9 Filter 2:**

![WhatsApp Image 2021-04-25 at 1 04 33 AM (1)](https://user-images.githubusercontent.com/74513063/115974753-6be8a780-a578-11eb-859e-689372218f10.jpeg)

# Discription of important parts of your .py file:

# How you tweaked the parameters of the classifiers you used from Scikit learn and their descriptions in your own words. 
# Highest kaggle score:
