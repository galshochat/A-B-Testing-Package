# A-B-Testing-Package

Library to ease the planning and correct execution of online randomized trials in business environment. It´s primarily directed at data analysts with limited acquaintance with advanced statistics but also to data scientists and product managers. Some methods syntax follows model R´s logic as R statistical modelling might be easier to implement occasionally. The library also includes some more complex testing scenarios involving clustered design and variance reduction methods.

## Standalone Modules:

### preparation.py

* calculation of sample size / power / false positive rate /  effect / standard deviation / point estimate given other parameters. Follows power.t.test/power.prop.test logic in R. Estimated parameters are corrected in case of multiple comparisons (Bonferroni).
* calculation of MDE as a function of experiment duration 
* clustered experiments power analysis simulation using GEE.


### variance_reduction.py
* CUPED - covariate columns creation by using the metric pre-experiment data
* CUPAC - covariate column creation leveraging predictions by LightGBM gradient boosted model trained on pre-experiment data (predictors of the metric which are not     affected by the treatment in experiment period).


