# Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification

## Description

Aims to predict the severity of wildfires using machine learning techniques, to provide first responders with valuable information to help mitigate damages. The project involves collecting and processing data on various environmental factors and training multiple machine learning models, including k-NN, SVM, Random Forest, XGBoost, AdaBoost, and ANN. The models were evaluated using metrics such as F1-score, Matthews Correlation Coefficient, and Macro AUC-ROC. Among the models tested, the ANN demonstrated the best performance, achieving a Matthews Correlation Coefficient of 0.08 and an AUC-ROC score of 0.57.

## Dataset API

Weather Data - Visual Crossing (https://www.visualcrossing.com/) and The National Oceanic and Atmospheric Administration (NOAA) (https://www.ncei.noaa.gov/access/)

Drought Index - U.S. Drought Monitor (https://droughtmonitor.unl.edu/Data.aspx) and NOAA (https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/divisional/time-series/0401/pdsi/1/2/2013-2023?base_prd=true&begbaseyear=1901&endbaseyear=2000#) and Drought.gov (https://www.drought.gov/about/partners/university-california-merced)

gridMET (https://www.climatologylab.org/gridmet.html) (download pdsi.nc) 

Vegetation Index - National Land Cover Database (NLCD) (https://www.mrlc.gov/data) 

Google Earth Engine API using the NASA “MOD13A1.061 Terra Vegetation Indices 16-Day Global 500m” dataset (https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A1_

Elevation and Slope - Google Earth Engine API, 'USGS/SRTMGL1_003' dataset

Wildfire Data - NASA Fire Information for Resource Management System (FIRMS) (https://firms.modaps.eosdis.nasa.gov/download/) 

Cal Fire (https://www.fire.ca.gov/incidents/)

# Machine learning models

## k-Nearest Neighbors (KNN) 
A non-parametric algorithm that is commonly used for classification and regression tasks. In KNN, the output is determined by the majority of its k nearest neighbors in the feature space.

<img width="634" alt="Screen Shot 2023-11-03 at 12 43 31 AM" src="https://github.com/ProBag/Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification/assets/143302669/9d652a2e-0250-40b0-bcfa-fa2b9398bbeb">


## Support Vector Machine (SVM)  
A supervised learning algorithm that is used for classification and regression analysis. It creates a hyperplane or set of hyperplanes in a high-dimensional space to perform classification, regression, and other tasks.

<img width="843" alt="Screen Shot 2023-11-03 at 12 48 14 AM" src="https://github.com/ProBag/Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification/assets/143302669/bf5b3423-6818-45b3-ae7c-8c32abd3196d">


## Random Forest  
An ensemble learning method that builds multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

<img width="486" alt="Screen Shot 2023-11-03 at 12 49 12 AM" src="https://github.com/ProBag/Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification/assets/143302669/5e7549a3-a17d-423c-b365-64580ad29edd">


## XGBoost 
An optimized distributed gradient boosting library that is designed to be highly efficient, flexible, and portable. It is an implementation of the gradient-boosting decision tree algorithm.

![bagging-sample](https://github.com/ProBag/Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification/assets/143302669/26717d51-9e8f-468c-b2e7-10451507f156)


## AdaBoost 
An ensemble learning method that combines multiple weak learners to create a strong classifier. It is particularly useful for large datasets and can be used for both binary and multiclass classification problems.

<img width="664" alt="Screen Shot 2023-11-03 at 12 51 49 AM" src="https://github.com/ProBag/Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification/assets/143302669/fa158531-119d-407c-890e-ed2c0ad65f7e">


## Artificial Neural Network (ANN) 
A computational model inspired by the structure and functions of biological neural networks. It consists of interconnected nodes, similar to neurons in a biological brain, and is used for various tasks such as classification, regression, and pattern recognition.

<img width="482" alt="Screen Shot 2023-11-03 at 12 52 53 AM" src="https://github.com/ProBag/Wildfire-Damage-Mitigation-Predictive-Machine-Learning-Models-for-Wildfire-Severity-Classification/assets/143302669/b66b2ee1-8e6d-4d90-9e3f-2cd7d826a018">

## Reference 
A. S. Mahdi and S. A. Mahmood, "Analysis of Deep Learning Methods for Early Wildfire Detection Systems: Review," 2022 5th International Conference on Engineering Technology and its Applications (IICETA), Al-Najaf, Iraq, 2022, pp. 271-276, doi: 10.1109/IICETA54559.2022.9888515.

Brownlee, J. (2016, August 17). A gentle introduction to XGBoost for applied machine learning.  	MachineLearningMastery.com. Retrieved April 28, 2023, from 	https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-	learning/ 

Freund, Y., & Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line 
Learning and an Application to Boosting. Journal of Computer and System Sciences, 55(1), 119–139. https://doi.org/10.1006/jcss.1997.1504

Grandini, M., Bagli, E., & Visani, G. (2020, August 13). Metrics for multi-class classification: An overview. arXiv.org. Retrieved April 28, 2023, from https://arxiv.org/abs/2008.05756 

Girtsou, S., Apostolakis, A., Giannopoulos, G., & Kontoes, C. (2021). A machine learning methodology for next-day wildfire prediction. 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS. https://doi.org/10.1109/igarss47720.2021.9554301 

Harrison, M. (2019). Machine learning pocket reference: Working with structured data in 	Python. O'Reilly Media, Inc.

Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class AdaBoost. Statistics and Its 
Interface, 2(3), 349–360. https://doi.org/10.4310/SII.2009.v2.n3.a8

Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y.-F. (2022). Next day wildfire spread: A machine learning dataset to predict wildfire spreading from remote-sensing data. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–13. https://doi.org/10.1109/tgrs.2022.3192974 

