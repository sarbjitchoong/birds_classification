# Predictive Modelling Competition (Sarb, Amir and Sabina)

## Introduction
For this project, we were aiming to classify bird species as accurately as possible with the Bird Species dataset from Kaggle and push our image classification skills to their limits! This was done through the medium of Tensorflow neural networks.

## Packages used
We additionally used the following packages for assessment and plotting purposes: 
- Matplotlib
- Numpy
- OS
- PIL

## Strategy 
In order to not overwhelm our computers, we opted to work with the final 25 bird classes for our image classification. We began by importing and visualizing our dataset to be sure that images would load correctly once we began to work with the neural network:

![](https://github.com/cb-ds-9/ds-predictive-modelling-project-4/blob/group5-sab-amir-sarb/Screen%20Shot%202022-11-22%20at%209.47.01%20AM.png)

Once we could feel confident that our data was successfully imported, we got to work training our models. In this project, we trained 2 models:

> ### Model 1:
![](https://github.com/cb-ds-9/ds-predictive-modelling-project-4/blob/group5-sab-amir-sarb/Screen%20Shot%202022-11-22%20at%209.47.39%20AM.png)

> ### Model 2:
![](https://github.com/cb-ds-9/ds-predictive-modelling-project-4/blob/group5-sab-amir-sarb/Screen%20Shot%202022-11-22%20at%209.48.10%20AM.png)

> We would like to note, we did not standard scale our data. This is because RGB channel values are in the [0,255] range versus [0,1], and in order to maintain colour range we did not opt to standardize our data by compressing the range. Instead, we opted to rescale with tf.keras.layers.Rescaling. 

## Testing
To test our models, we ran our train, test and validation sets through an accuracy score function. On model 1 we see a train score of ~99%, and a test score of ~85%, potentially indicating some problems with overfitting. On our second model, we see a train score of ~99% and a test score of ~76%. In addition to this, we assessed our models through Training and Validation plots to see how the learning performance may change over the number of epochs and help us diagnose any problems with learning which would give us an over or underfit model:

> ### Model 1:
![](https://github.com/cb-ds-9/ds-predictive-modelling-project-4/blob/group5-sab-amir-sarb/Screen%20Shot%202022-11-22%20at%209.47.54%20AM.png)

> ### Model 2:
![](https://github.com/cb-ds-9/ds-predictive-modelling-project-4/blob/group5-sab-amir-sarb/Screen%20Shot%202022-11-22%20at%209.48.20%20AM.png)

## Conclusion
In the first model we got an accuracy of approximately 77% on the unseen data. However, we faced overfitting as there was a gap between the performance of the model on the train and test data. To remove this issue, therfore, we included the dropout and data augmentation in the model 2. Doing this could increase the accuracy by about 2%, and removed the overfitting problem. If we had more time, we could probably increase the accuracy by modifying the number of hidden layers, units in each layer, epochs and dropout proportion.
