 <p align="left"><img src="/fig/uu_ml.png" width="800" heigth="140"></p>

 # Elements of Machine Learning in Geosciences

 ### A introductory session for machine learning basics. First designed for Utrecht University, The Netherlands.

 ##### This session is designed for supporting the machine learning components of the education project **Beyond Maps**: ***teaching modules to stimulate interactivity with Open Spatial data and data science in Geosciences MSc and the Applied Data Science MSc***. The teaching modules involve all Geosciences departments and from faculties involved in the Applied Data Science MSc, Utrecht University, Netherlands.

 ##### Dr. Jiong (Jon) Wang,  j.wang2@uu.nl


 ## Foreword
 -------------------
 There are two major problems you will encounter in data science: (1) regression and (2) classification. Dealing with data in GeoSciences is without exception. In this course, you will build knowledge of powerful artificial intelligence techniques in dealing with data, especially in the context of Geosciences. In fact, you are going to focus on the core of artificial intelligence, which is machine learning to solve regression and classification problems.

 <p align="center"><img src="/fig/ds.png" width="800" heigth="450"></p>

 Given this purpose, you are assumed to already have basic concepts about machine learning, including supervised and unsupervised learning, model complexity, curse of high dimensionality and variance-bias tradeoff.

 You will start with the simplest regression scenario: the ordinary least squares (OLS), from which we will generalize to one of the most fundamental group of regression models: linear basis function models. A more advanced setup is that you do not have to specify the form of basis function but only capture the covariance in the input space, where Gaussian Process regression is introduced. A short recap of the Kriging will be provided as a special case of Gaussian Process regression in geoscience/geostatistics. So far, you can consider all methods are closely connected with incremental improvements among them. A transition is then introduced from these methods to additive models, which paves the way to more advanced ensemble learning such as Random Forest.

 You will again start with simple scenarios of linear methods in classification, such as the Support Vector Machine (SVM). Then non-linear scenario will be introduced by adding kernel components on top of the SVM. Combination of linear units leads to advanced neural networks model. You will be able to apply different neural network architecture to geospatial datasets, where convolutional neural networks will be introduced as one of the most widely adopted model architecture in geoscience.

 To solve [**regression problems**](https://github.com/jonwangio/uu_ml/blob/main/ml_geo_uu.ipynb), you will build your knowledge and techniques by following:

 - 0. [Getting started: Linear model fitting in 1-dimension](https://github.com/jonwangio/uu_ml/blob/main/REG_0_getting_started.ipynb)
 - 1. [Polynomial curve fitting and regularization](https://github.com/jonwangio/uu_ml/blob/main/REG_1_polynomial_regularization.ipynb)
 - 2. [Model evaluation](https://github.com/jonwangio/uu_ml/blob/main/REG_2_model_evaluation.ipynb)
 - 3. [Bayesian method](https://github.com/jonwangio/uu_ml/blob/main/REG_3_bayes.ipynb)
 - 4. [Additive model, tree-based regression and ensemble learning](https://github.com/jonwangio/uu_ml/blob/main/REG_4_additive_trees.ipynb)
 - 5. [Regression on real dataset: air pollution mapping](https://github.com/jonwangio/uu_ml/blob/main/REG_5_regression_airPollution.ipynb)

 For classification problems, you obtain the following steps:

 - 0. [Getting started: Unsupervised classifications](https://github.com/jonwangio/uu_ml/blob/main/CLA_0_unsupervised_kmeans_EM_mixtureModels.ipynb)
 - 1. [Supervised classification: simple linear models](https://github.com/jonwangio/uu_ml/blob/main/CLA_1_supervised_linearDecisionBoundary.ipynb)
 - 2. [The Support Vector Machine](https://github.com/jonwangio/uu_ml/blob/main/CLA_2_SVM_highDimensionality.ipynb)
 - 3. [The neural networks]

 There can be quite a steep learning curve while you walk through the content, so please always keep in mind that it is very important and efficient to get hands on experience while learning. ***DO NOT*** just sit back and read the codes, ***DO*** modify and rewrite, and check the difference.

 ##### Credit
 -------------------
 You're very welcome to use the content of this page for teaching and learning. Credit to this work can be given as:
 ~~~
 J. Wang, Elements of Machine Learning: A introductory session for machine learning basics (2021), GitHub repository,
 https://github.com/jonwangio/uu_ml
 ~~~

 All materials are prepared based upon three *influential* works in data science and machine learning:
 ~~~
 Bishop, C.M., 2006. Pattern recognition and machine learning. springer.
 ~~~
 An elegant sample implementation of this book can be found at: https://github.com/ctgk/PRML

 ~~~
 Hastie, T., Tibshirani, R. and Friedman, J., 2009. The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
 ~~~
 which is publicly accessible as an open material at: https://web.stanford.edu/~hastie/ElemStatLearn/.

 ~~~
 Rasmussen, C.E., 2003, February. Gaussian processes in machine learning. In Summer School on Machine Learning (pp. 63-71). Springer, Berlin, Heidelberg.
 ~~~
 which is again publicly available online at: http://www.gaussianprocess.org/gpml/

 The data used in the **Challenges** is from research project operated by Dr. Lu, Meng with details at:
 ~~~
 Lu, M., Schmitz, O., de Hoogh, K., Kai, Q. and Karssenberg, D., 2020. Evaluation of different methods and data sources to optimise modelling of NO2 at a global scale. Environment international, 142, p.105856.
 ~~~

 ## Getting started
 -------------------
 We will be deploy this course by taking advantage of cloud based resources. The *Python* codes written in *Jupyter Notebook* will be loaded into *Google Colab*, which is an online programming interface working exactly like *Jupyter Notebook* in your browser. So you can simply treat it as an online version of Jupyter Notebook, the only difference is that your code will be executed on the server provided by *Google* in the cloud.

 Before going to *Colab*, you need a *Google/Gmail* account. Once you create and login in *Colab* with your account, you will immediately see an interface below in your browser:

 <p align="center"><img src="/fig/0.png" width="800" heigth="450"></p>

 On this page, you can familiarize yourself with the panels of the main menu, the table of contents and programming/coding cells. You will be mainly working in the panel of programming/coding cells, where you can write either *Python* codes or texts in each cell.

 In order to work with codes I prepared for you, you can simply **open** the notebook in this *GitHub* repository by providing the link to *Colab*. Click **File** > **Open notebook**, you will see dialogue box below:

 <p align="center"><img src="/fig/1.png" width="800" heigth="450"></p>

 Navigate to the **GitHub** tab in the dialogue box, you can provide the link of *GitHub* repositories that you would like to work with. Copy and paste the address of this *GitHub* repository at *https://github.com/jonwangio/Programming-Basics*, and hit **enter**, you will find the *Jupyter Notebook* listed. **Double click** to select the listed notebook, you are now totally set with your programming environment! As you can see sections prepared for you shown below:

 <p align="center"><img src="/fig/2.png" width="800" heigth="450"></p>

 <p align="center"><img src="/fig/3.png" width="800" heigth="450"></p>

 <p align="center"><img src="/fig/4.png" width="800" heigth="450"></p>

 Follow the instructions in the notebook and play with the sample codes. As been said, please ***DO*** feel free to modify, delete, and add the programs to get the most out of your hands-on experience. Enjoy Machine Learning!
