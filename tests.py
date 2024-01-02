from typing import List, Optional, Literal, Dict
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


class Tests:
    @classmethod
    def validate_selected_arguments(cls, params_dict):
        def validate(func):
            def wrapper(*args, **kwargs):
                obj = func(*args, **kwargs)
                for key, value in params_dict.items():
                    if key in kwargs:
                        if not kwargs[key] in value:
                            raise ValueError(
                                f"""{kwargs[key]} is not an allowed argument for the parameter {
                                    key}. Valid values are {', '.join(value)}"""
                            )
                return obj

            return wrapper

        return validate

    def __init__(
        self,
        df: pd.DataFrame = None,
        metrics: List[str] = None,
        distribution: Literal["Gaussian", "Normal", "Binomial"] = "Normal",
        method: str = "t-test",
        treatment_col: str = None,
        control_variant_name: str = "control",
        cluster_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        correction: Optional[Literal["bonferroni", "fdr_bh"]] = None,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        covariates: Optional[List[str]] = None,
    ):

        self.data = df
        self.metrics = metrics
        self.distribution = distribution
        self.method = method
        self.treatment_col = treatment_col
        self.control_variant_name = control_variant_name
        self.cluster_cols = cluster_cols
        self.alpha = alpha
        self.correction = correction
        self.alternative = alternative
        self.covariates = covariates
        self.variants = np.setdiff1d(
            self.data[self.treatment_col].unique(), [self.control_variant_name]
        )
        self.results = pd.DataFrame()

    def encode_columns(self, data: pd.DataFrame, cat_col: List[str]) -> pd.DataFrame:
        """one hot encoded columns prepended to the dataset"""
        encoded = pd.get_dummies(
            data[cat_col], columns=cat_col, drop_first=True, prefix="", prefix_sep=""
        )

        data.drop(columns=self.treatment_col, inplace=True)
        data = pd.concat([encoded, data], axis=1)
        return data

    def extract_regression_results(
        self, reg_results: sm.regression.linear_model.OLSResults, param_index: int = 1
    ) -> Dict:
        deg_f = reg_results.df_resid

        ate = reg_results.params[1 + param_index]

        if self.alternative == "two-sided":
            p_value = t.sf(abs(reg_results.tvalues[1 + param_index]), deg_f) * 2

            conf_intervals = tuple(
                reg_results.conf_int(alpha=self.alpha).iloc[1 + param_index].values
            )
        else:
            if self.alternative == "greater":
                p_value = t.sf(reg_results.tvalues[1 + param_index], deg_f)
            else:
                p_value = t.cdf(reg_results.tvalues[1 + param_index], deg_f)

            conf_intervals = tuple(
                reg_results.conf_int(alpha=self.alpha * 2).iloc[1 + param_index].values
            )

        return {
            "ate": ate,
            "p_value": p_value,
            "lower confidence boundary": conf_intervals[0],
            "upper confidence boundary": conf_intervals[1],
            "Significant": p_value < self.alpha,
        }

    def ttest(self, data: pd.DataFrame, metric: str) -> Dict:
        if len(self.variants) == 1:
            test = ttest_ind(
                data[data[self.treatment_col] == self.variants[0]][metric],
                data[data[self.treatment_col] == self.control_variant_name][metric],
                alternative=self.alternative,
            )
            ate = data.groupby(self.treatment_col)[metric].mean().diff()[1]
            p_value = test[1]

            conf_intervals = test.confidence_interval(confidence_level=1 - self.alpha)[
                0:2
            ]

            yield {
                "metric": metric,
                "control": self.control_variant_name,
                "variant": self.variants[0],
                "ate": ate,
                "p_value": p_value,
                "lower confidence boundary": conf_intervals[0],
                "upper confidence boundary": conf_intervals[1],
                "Significant": p_value < self.alpha,
            }
        else:
            raise ValueError(
                f"""You have {len(self.variants)} variants but t-test is not
                  a valid test for multivarint scenario."""
            )

    def ols(self, data: pd.DataFrame, metric: str):

        data = self.encode_columns(data, cat_col=[self.treatment_col])

        if self.cluster_cols:
            regression_results = smf.ols(
                f"{metric} ~ {' + '.join(data.drop([metric] + self.cluster_cols, axis=1))}",
                data=data,
            ).fit(cov_type="cluster", cov_kwds={"groups": data[self.cluster_cols]})
        else:
            regression_results = smf.ols(
                f"{metric} ~ {' + '.join(data.drop(metric, axis=1))}", data=data
            ).fit()
        for idx, variant in enumerate(self.variants):
            results_dict = {
                "metric": metric,
                "control": self.control_variant_name,
                "variant": variant,
            }

            results_dict.update(
                self.extract_regression_results(regression_results, idx)
            )
            yield results_dict

    def apply_correction(self, results: pd.DataFrame):

        adjusted = multipletests(
            pvals=results["p_value"],
            alpha=self.alpha,
            method=self.correction,
            # maxiter = -1,
        )
        print(adjusted)
        results["Significant"] = adjusted[0]
        results["adjusted_pvalues"] = adjusted[1]

        return results
