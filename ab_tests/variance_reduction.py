import pandas as pd
import lightgbm as lgb
from typing import List
import numpy as np


class VarianceReduction:
    def __init__(
        self,
        data: pd.DataFrame,
        time_col: str = None,
        exp_start_date: str = None,
        metric: str = None,
        distribution: str = "Gaussian",
        predictors: List[str] = None,
        cat_predictors: List[str] = None,
        user_col: List[str] = None,
        covariate_prefix: str = "covariate_",
        **kwargs
    ):

        self.data = data
        self.time_col = time_col
        self.data[time_col] = pd.to_datetime(self.data[time_col])
        self.start_date = pd.to_datetime(exp_start_date)
        self.metric = metric
        self.distribution = distribution
        self.predictors = predictors
        self.kwargs = kwargs
        self.prefix = covariate_prefix
        self.user_col = user_col

        if cat_predictors:
            for cat_predictor in cat_predictors:
                if cat_predictor not in predictors:
                    raise ValueError(
                        "cat_predictors must be included in the argument to 'predictors' parameter"
                    )
        self.cat_predictors = cat_predictors

    def split_data(self):

        X_train = self.data[self.data[self.time_col] < self.start_date].drop(
            columns=self.metric
        )
        y_train = self.data[self.data[self.time_col] < self.start_date][self.metric]
        X_test = self.data[self.data[self.time_col] >= self.start_date].drop(
            columns=self.metric
        )
        y_test = self.data[self.data[self.time_col] >= self.start_date][self.metric]

        return X_train, y_train, X_test, y_test

    def cupac(self):

        X_train, y_train, X_test, y_test = self.split_data()

        if self.distribution in ["Gaussian", "Normal"]:

            lgbm = lgb.LGBMRegressor(**self.kwargs, verbose=0)
            model = lgbm.fit(
                X_train[self.predictors],
                y_train,
                feature_name=self.predictors,
                categorical_feature=self.cat_predictors,
            )
            pred_test = model.predict(X_test[self.predictors])

        elif self.distribution == "Binomial":

            lgbm = lgb.LGBMClassifier(**self.kwargs, verbose=0)
            model = lgbm.fit(
                X_train[self.predictors],
                y_train,
                feature_name=self.predictors,
                categorical_feature=self.cat_predictors,
            )
            pred_test = model.predict_proba(X_test[self.predictors])

        else:
            raise ValueError(
                "The metric distribution should be one of 'Binomial','Gaussian','Normal'"
            )

        X_test[self.metric] = y_test
        X_test[self.prefix + self.metric] = pred_test

        return X_test

    def cuped(self):

        pre_data = self.data[self.data[self.time_col] < self.start_date]
        post_data = self.data[self.data[self.time_col] >= self.start_date]

        common_users = set(pre_data[self.user_col]).intersection(
            post_data[self.user_col]
        )
        X = pre_data[pre_data[self.user_col].isin(common_users)]
        X_users_means = (
            X.groupby(self.user_col)[self.metric]
            .mean()
            .reset_index()
            .rename(columns={self.metric: "pre_cluster_mean"})
        )
        X_overall_mean = X_users_means["pre_cluster_mean"].mean()

        Y = post_data[post_data[self.user_col].isin(common_users)]
        Y_users_means = (
            Y.groupby(self.user_col)[self.metric]
            .mean()
            .reset_index()
            .rename(columns={self.metric: "post_cluster_mean"})
        )

        Y = Y_users_means.merge(X_users_means, how="inner", on=self.user_col)

        Theta = (
            np.cov(Y["post_cluster_mean"], Y["pre_cluster_mean"])
            / np.var(X_users_means["pre_cluster_mean"])
        )[0][1]
        post_data = post_data.merge(
            Y[[self.user_col, "pre_cluster_mean"]], how="left", on=self.user_col
        )
        post_data[self.prefix + self.metric] = [
            i[self.metric] - Theta * i["pre_cluster_mean"] + Theta * X_overall_mean
            if not np.isnan(i["pre_cluster_mean"])
            else i[self.metric]
            for idx, i in post_data.iterrows()
        ]
        post_data.drop(columns="pre_cluster_mean", inplace=True)

        return post_data
