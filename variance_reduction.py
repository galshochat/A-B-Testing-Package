import pandas as pd
import lightgbm as lgb
from typing import List
import numpy as np


class VarianceReduction:
    def __init__(
        self,
        method: str = "cuped",
        data=pd.DataFrame,
        time_col: str = None,
        exp_start_date: str = None,
        metric: str = None,
        distribution: str = "Gaussian",
        predictors: List[str] = None,
        cat_predictors: List[str] = None,
        cluster_cols: List[str] = None,
        covariate_prefix: str = "covariate_",
        **kwargs
    ):

        self.method = method
        self.data = data
        self.time_col = time_col
        self.data[time_col] = pd.to_datetime(self.data[time_col])
        self.start_date = pd.to_datetime(exp_start_date)
        self.metric = metric
        self.distribution = distribution
        self.predictors = predictors
        self.kwargs = kwargs
        self.prefix = covariate_prefix
        self.cluster_cols = cluster_cols

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

        X_train, y_train, X_test, y_test = self.split_data(self.data)

        if self.distribution in ["Gaussian", "Normal"]:

            lgbm = lgb.LGBMRegressor(**self.kwargs)

        elif self.distribution == "Binomial":

            lgbm = lgb.LGBMClassifier(**self.kwargs)

        else:
            raise ValueError(
                "The metric distribution should be one of 'Binomial','Gaussian','Normal'"
            )

        model = lgbm.fit(
            X_train,
            y_train,
            feature_name=self.predictors,
            categorical_feature=self.cat_predictors,
        )

        pred_test = model.predict_proba(X_test)

        exp_data = pd.concat([X_test, y_test])
        exp_data[self.prefix + self.metric] = pred_test

        return exp_data

    def cuped(self):

        pre_data = self.data[self.data[self.time_col] < self.start_date]
        post_data = self.data[self.data[self.time_col] >= self.start_date]

        common_users = set(pre_data[self.cluster_cols]).intersection(
            post_data[self.cluster_cols]
        )
        X = pre_data[pre_data[self.cluster_cols].isin(common_users)]
        X_users_means = (
            X.groupby(self.cluster_cols)[self.metric]
            .mean()
            .reset_index(drop=True)
            .rename("pre_cluster_mean")
        )
        X_overall_mean = X_users_means.mean()
        Y = post_data[post_data[self.cluster_cols].isin(common_users)]
        Y_users_means = (
            Y.groupby(self.cluster_cols)[self.metric]
            .mean()
            .reset_index(drop=True)
            .rename("post_cluster_mean")
        )
        print(Y_users_means)
        Y = Y_users_means.merge(X_users_means, how="inner", on=self.cluster_cols)

        Theta = np.cov(Y["post_cluster_mean"], Y["pre_cluster_mean"]) / np.var(
            X_users_means["pre_cluster_mean"]
        )

        Y[self.prefix + self.metric] = (
            Y[self.metric]
            - Theta * post_data["pre_cluster_mean"]
            + Theta * X_overall_mean
        )
        post_data = post_data.merge(
            Y[[self.cluster_cols, "pre_cluster_mean"]], how="left", on=self.cluster_cols
        )
        post_data[self.prefix + self.metric] = (
            post_data[self.metric]
            - Theta * post_data["pre_cluster_mean"]
            + Theta * X_overall_mean
            if post_data["pre_cluster_mean"]
            else post_data[self.metric]
        )

        return post_data


if __name__ == "__main__":
    n = VarianceReduction(predictors=["a", "b"], cat_predictors=["b", "a"])
