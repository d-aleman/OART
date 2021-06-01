# OART
One-class classifier based on adaptive resonance theory, implemented in MATLAB. Sample function call:
```
[anomalyLikelihood,LTM_values] = oart(T,trainRows)
```
The function returns the likelihood of each sample in dataset in matrix `T` being anomalous in `anomalyLikelihood` and the long-term memory (LTM) values found for each feature in `LTM_values`. Training is performed on dataset row indices in vector `trainRows`. Specific parameters to tune the OART classifier are defined in the function, and should be tuned by the user for best performance on the particular dataset `T`.

Training is done in function `oart_train.m` and testing is done in function `oart_test.m`.
