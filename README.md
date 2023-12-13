# A Prediction on the Severity of Power Outages

## Problem Identification

Understanding and predicting the severity of power outages is vital for ensuring public safety, minimizing economic losses, protecting infrastructure, preparing communities, and allocating emergency resources efficiently. Accurate predictions enable effective response and recovery efforts, helping to mitigate the overall impact of such incidents.

This project mainly centered around a question "how to predict the severity of a major power outage". The project is based on our previous project, our exploratory data analysis on this dataset can be found [here](https://lr580.github.io/power_outages_stats/), where we've made comprehensive investigation of the power outage data. The data we use can be downloaded [here](https://engineering.purdue.edu/LASCI/research-data/outages).

This dataset includes 56 column variables and 1534 row variables in total. In researching the question stated above, not all 56 columns variables will be included in the analysis. Thus, relevant information will be included for the analysis such as: 

| Column               | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `OUTAGE.DURATION`    | The minutes an outage lasts                                 |
| `CLIMATE.CATEGORY`   | The climate type(warm, normal, cold) where an outage occurs |
| `CAUSE.CATEGORY`     | The reason for an outage.                                   |
| `CUSTOMERS.AFFECTED` | The number of customers affected by the outage.             |
| `DEMAND.LOSS.MW`     | The power loss(megawatt) in the outage.                     |
| `NERC.REGION`        | The NERC region code where an outage occurs.                |

In our predealing process, we removing all the rows if the `CUSTOMERS.AFFECTED` or `OUTAGE.DURATION` is missing, since we consider with such missing, it's hard to tell the severity. And then, we fill in missing values of another row `DEMAND.LOSS.MW`. 

The prediction problem we'd focus on is predicting the severity of a major power outage.

There may be many columns which can measure the severity, such as number of affected customers, duration, or demand loss. Here, we use the number of affected customers as the only measurement. That is to say, the number of affected customers in an outage is our prediction target.

The reason we use outage rather than other columns(like the number of customers, demand loss, etc.) is that:

1. Choosing "number of customers affected" as the primary factor for predicting power outage severity is effective because it directly reflects the impact's extent and is a clear indicator of socio-economic effects. This measure is typically more reliable and accessible than others.
2. Choosing other factors like "duration" or "demand loss" might not always proportionately reflect the outage's severity and could complicate the model.
3. Also, there's too many missing values of the `DEMAND.LOSS.MW`, which makes it difficult to use.
4. Additionally, integrating multiple factors could increase complexity and risk of collinearity, detracting from the model's manageability and predictive accuracy.

We use the formula {% raw %} $$severity = \log_2 (number\_of\_customers+1)$$ {% endraw %} to measure the severity by experience. The reason for the transformation is that, by observing the data, we found that there're a large difference of the order of magnitude, if we directly use the `CUSTOMERS.AFFECTED` feature, it's both hard to measure and train the model, since in large numbers, any "slight" difference will be great.

Clearly, it's a regression model. The response variable which the model is going to predict is the logarithmic value of `CUSTOMERS.AFFECTED`, the number of people affected by the outage, which can roughly measure the severity of an outage.

$$R^2$$

We use $$R^2$$ as metric to measure our model. The reasons are that:

1. Since it's not a classification model, so we won't use classification metrics like precision or recall.
2. The two metrics RMSE and \(R^2\) are classic for regression model. But we only need one of them to determine which model better. So we compare them as below:
3. RMSE is preferred when the absolute size of errors is crucial, as it directly reflects the average difference between the predicted and actual values and is more sensitive to larger errors.
4. $R^2$ is better suited for assessing a model's explanatory power, as it measures how well the model explains the variability of the target variable, and is useful in standardized performance evaluation across different datasets.
5. We consider the explanatory power and standardized performance more important in our problem, so we use the $R^2$.

## Baseline Model

We try analyzed many features manually(due to space constraints, the process is omitted here), i.e. `U.S._STATE`, `POSTAL.CODE`, `CLIMATE.REGION`, `ANOMALY.LEVEL`, `OUTAGE.START.DATE`, `OUTAGE.START.TIME`, `OUTAGE.RESTORATION.DATE`, `OUTAGE.RESTORATION.TIME`, `TOTAL.PRICE`, `TOTAL.SALES`, `TOTAL.CUSTOMERS`, `POPULATION`, `POPDEN_URBAN`, we try using them singularily and together, but little effect is found. So we think them as irrelevant features. But fortunately we try out a crucial feature `CAUSE.CATEGORY`.

It is worth noting that, though `CAUSE.CATEGORY` is useful in baseline model, we've tried adding `CAUSE.CATEGORY.DETAIL` and `HURRICANE.NAMES`, two features that explain more about the cause category, but they're also make no contribution on improving our model.

Consequently, we adopt the single feature `CAUSE.CATEGORY` for baseline model and state the possible reasons why this feature is useful.

More features and fine adjustments will be added later in the final model.

We adopt the classic scheme that using 75% of the data as training set, 25% of the data as validation set.

We've try several different classic model(due to space constraints, the process is omitted here), and we figure out that the `LinearRegression`, `KNeighborsRegressor` and `SVR` models cannot work well. While the `DecisionTreeRegressor` and `RandomForestRegressor` works well.

We adopt the classic `DecisionTreeRegressor` as baseline model, and we will try compare it later with `RandomForestRegressor` and choose the best one as the final model.

To make our reported result stable and reproducible, we set the random seed manually.

First, we define a helper class to convert the cause category strings into ordinal values, which will be used in the `ColumnTransformer` later.

We observe all the different values of cause category, and find that there's only 7 different values. So we use the mapping below to convert the feature `CAUSE.CATEGORY` into ordinal encoding.

```python
{'severe weather': 0, 'intentional attack': 1, 'public appeal': 2, 'system operability disruption': 3, 'islanding': 4, 'equipment failure': 5, 'fuel supply emergency': 6}
```

We use the baseline model to predict values in both train set and validation set, and calculate the metric selected above.

We find that our model work well on both train and test data. The $R^2$ are both approximately $0.76\sim 0.81$, and by looking at some real examples of the prediction, we find it gets a near value. This means that the $7$ different types of cause category can roughly related to $7$ different order of magnitude in the number of affected customers.

```
train evaluate: 0.8131982175490576
test evaluate: 0.7675446684209979
```

Samples of train prediction:

| Prediction Severity | Real Severity |
| ------------------- | ------------- |
| 11.557              | 12.41         |
| 0.736               | 0.0           |
| 10.084              | 11.444        |
| 11.557              | 11.744        |
| 10.553              | 11.142        |

Samples of test prediction:

| Prediction Severity | Real Severity |
| ------------------- | ------------- |
| 11.557              | 13.108        |
| 0.736               | 0.0           |
| 0.736               | 0.0           |
| 11.557              | 12.553        |
| 11.557              | 12.832        |

## Final Model

### Hyperparameter Searching

The first improvement may lies in hypermarameter selection.

We first present a hyperparameter searching helper function using `GridSearchCV`. The searching range is `[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,50]`。

we find that the `max_depth` parameter is important for the `DecisionTreeRegression`, so we use it to search the tree depth.

The best tree depth is 6. As we expected, no significant improvement found, because the `CAUSE.CATEGORY` is too simple(only 7 different values), there may have little improvement by changing tree depth.

```
Before: 
train evaluate: 0.8131982175490576
test evaluate: 0.7675446684209979
best param: {'model__max_depth': 6}
After: 
train evaluate: 0.8121172035410144
test evaluate: 0.7748218273592309
```

### Different Encodings

We then try using `OneHotEncoder` converting it into nominal encoding to replace the ordinal encoding.

```
Before: 
train evaluate: 0.8131982175490576
test evaluate: 0.7675446684209979
best param: {'model__max_depth': 6}
After: 
train evaluate: 0.8121172035410144
test evaluate: 0.774821827359231
```

No improvement found.

### Adding New Feature

We adopt `NERC.REGION` now, since we consider different regions of NERC have different ability to deal with outage, thus making severity different.

```
Before: 
train evaluate: 0.8422277429766968
test evaluate: 0.7823223717827422
best param: {'model__max_depth': 9}
After: 
train evaluate: 0.8403517560962385
test evaluate: 0.8209825624674565
```

We find that adding `NERC.REGION` can improve a little. So we adopt it.

We then try many other features to add into the model, but almost no more valid improvement can be seen(due to space constraints, the process is omitted here). 

We find that another useful feature is `DEMAND.LOSS.MW`. However, the `DEMAND.LOSS.MW` feature may belong to the feature we would not know at the "time of prediction", the improvement is shown below, but we won't add it into our final model.

So in conclusion, we try as many as near 20 features and their combinations, but only find three features useful, which are `CAUSE.CATEGORY`, `NERC.REGION`, `DEMAND.LOSS.MW`, while the first two are the information we would know at the "time of prediction", so we only use two features `CAUSE.CATEGORY` as well as `NERC.REGION`.

### Different Models

Finally, we try changing it into `RandomForestRegressor` and perform hyperparameter searching again. We search max depth in `[1,2,3,4,5,6,7,8,20,50,100]` and number of estimators in `[1,10,25,50,100]`.

```
Before: 
train evaluate: 0.8412494230284535
test evaluate: 0.7786905744712731
best param: {'model__max_depth': 6, 'model__n_estimators': 50}
After: 
train evaluate: 0.8370717045460225
test evaluate: 0.8161925455663533
```

We find that the two models, `DecisionTreeRegressor` and `RandomForestRegressor`, are almost the same. Also, we've performed the `LinearRegression`, `KNeighborsRegressor` and `SVR`, the three models all work terribly(due to space constraints, the process is omitted here). So we simply adopt the `DecisionTreeRegressor`.

Therefore, our final model is shown below. The visualization that describes our mode's performance is shown below.

```
best param: {'model__max_depth': 9}
train evaluate: 0.8403517560962385
test evaluate: 0.8209825624674565
```

<iframe src="assets/perform_pipeline.html" width=800 height=600 frameBorder=0></iframe>

We'd perform it in the whole data, compared with baseline model. The $R^2$ is shown below:

```
baseline model's R2: 0.801458743651225
final model's R2: 0.8353712022533668
```

There's improvement on $R^2$ in the final model, which means that our improvement methods are useful.



## Fairness Analysis

To answer the question that whether our model is fair, that is, if it work worse for individuals in some groups than it does in others, we'd perform a fairness analysis below.

The quantitative attribute(evaluation metric) we adopt is $R^2$, so we use $R^2$ across two groups to perform the analysis, that is, absolute difference between the $R^2$ values: $|R^2_{groupX} - R^2_{groupY}|$.

We simply define: 

1. group X as the outage where `CLIMATE.CATEGORY` is `cold`
2. group Y as the outage where `CLIMATE.CATEGORY` is not `cold`. 

Obviously, it's a binary groups.

we use permutation test to perform it.

Null hypothesis: Our model is fair. Its precision for the outage where the climate is cold and not cold are roughly the same, and any differences are due to random chance.

Alternative hypothesis: Our model is unfair. Its precision for the outage where the climate is cold is lower than that of the outage where the climate is not cold, or otherwise.

Significance level: 0.05.

Since p-value measures the probability of a extreme case happens if null hypothesis is true, and if it's not the same(which means extreme), the evaluation metric will be greater, so we adds up p-value when simulated value is greater than observed value.

To make our result reproducible, we use a static seed list(`SEED+_`) to random shuffle permutation.

<iframe src="assets/permutation_test.html" width=800 height=600 frameBorder=0></iframe>

The result shows that p-value is 0.278. 

We use a significance level of 0.05. Since p-value is greater than 0.05, we fail to reject the null hypothesis, which means that it's more possible that our model is fair, its precision for different groups are roughly the same.

It also imply that `CLIMATE.CATEGORY` have no effect on predicting the severity, which proves our conclusion that `CLIMATE.CATEGORY` is useless feature to predicting the severity is correct.
