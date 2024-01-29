import pandas as pd
import numpy as np
from typing import List, Optional, Literal, Dict
from scipy.stats import ttest_ind, t, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


class StatTests:

    """class implementing various statistical tests

    methods can be added in the following fashion:
    - in this class. It should be a Generator function yielding
      a results dict as in other methods.
    - in config.py in PARAMS_DICT under 'method' key (string method name)
    - in analysis.py in 'select_method_func' (mapping between name
      string and the created func)
    """

    def validate_one_test_version(func):
        def wrapper(self, *args, **kwargs):

            if len(self.variants) > 1:
                raise ValueError(
                    f"""You have {len(self.variants) + 1} variants but {self.method} is not
                  a valid test for a multivariant scenario."""
                )
            obj = func(self, *args, **kwargs)
            return obj

        return wrapper

    def validate_only_binomial(func):
        def wrapper(self, *args, **kwargs):
            if kwargs["data"][self.metric].nunique() > 2:
                raise ValueError(
                    f"""{self.method} is a test modelling a binomial distribution.
                Make sure your metric values are binary"""
                )
            obj = func(self, *args, **kwargs)
            return obj

        return wrapper

    def enforce_inf_one_sided_tests(func):
        """corrects confidence interval bounds for one-sided tests"""

        def wrapper(self, *args, **kwargs):
            obj = func(self, *args, **kwargs)
            if self.alternative == "greater":
                obj["upper confidence boundary"] = np.inf
            elif self.alternative == "less":
                obj["lower confidence boundary"] = -np.inf
            else:
                pass
            return obj

        return wrapper

    def __init__(
        self,
        data: pd.DataFrame,
        metric: str,
        method: str = None,
        cluster_cols: Optional[List[str]] = None,
        covariates: Optional[List[str]] = None,
        treatment_col: str = None,
        control_variant_name: str = "control",
        alpha: float = 0.05,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        tails: int = 2,
    ):
        self.data = data
        self.metric = metric
        self.method = method
        self.treatment_col = treatment_col
        self.control_variant_name = control_variant_name
        self.cluster_cols = cluster_cols
        self.alpha = alpha
        self.alternative = alternative
        self.tails = tails
        self.covariates = covariates
        self.variants = np.setdiff1d(
            self.data[self.treatment_col].unique(), [self.control_variant_name]
        )

        self.data = self.drop_null_rows()
        self.data = self.filter_unnecessary_columns()
        self.result = self.results()

    @enforce_inf_one_sided_tests
    def results(self):
        df = pd.DataFrame()
        for idx, _ in enumerate(
            self.select_method_func(data=self.data, metric=self.metric)
        ):
            df = pd.concat([df, pd.DataFrame(_, index=[idx])], axis=0)
        return df

    def select_method_func(self, **kwargs):
        """selects the appropriate function according to stat method chosen"""

        methods_funcs_dict = {
            "t-test": self.ttest,
            "ols": self.ols,
            "logistic-regression": self.logit,
            "z-test": self.ztest,
            "gee": self.GEE,
            "delta": self.delta,
        }
        return methods_funcs_dict[self.method](**kwargs)

    def encode_columns(self, data: pd.DataFrame, cat_col: List[str]) -> pd.DataFrame:
        """one hot encoded columns prepended to the dataset"""
        encoded = pd.get_dummies(
            data[cat_col], columns=cat_col, drop_first=True, prefix="", prefix_sep=""
        )

        data.drop(columns=self.treatment_col, inplace=True)
        data = pd.concat([encoded, data], axis=1)
        return data

    def drop_null_rows(self):
        data = self.data[self.data[self.metric].notnull()].reset_index(drop=True)
        return data

    def filter_unnecessary_columns(self) -> pd.DataFrame:

        columns = [self.metric] + [self.treatment_col]
        if self.covariates:
            columns.extend(self.covariates)
        if self.cluster_cols:
            columns.extend(self.cluster_cols)
        return self.data[columns]

    def extract_regression_results(
        self,
        reg_results: sm.regression.linear_model.OLSResults,
        data: pd.DataFrame,
        metric: str,
        variant: str,
        param_index: int = 0,
    ) -> Dict:

        deg_f = reg_results.df_resid
        data = data[data[self.variants] != 1].drop(columns=metric, inplace=True)
        control = np.mean(reg_results.predict(data))

        ate = reg_results.params[1 + param_index]
        variation = control + ate

        if self.alternative == "two-sided":
            p_value = t.sf(abs(reg_results.tvalues[1 + param_index]), deg_f) * 2

            conf_intervals = tuple(
                reg_results.conf_int(alpha=self.alpha).iloc[1 + param_index].values
            )

        else:
            if self.alternative == "greater":
                p_value = t.sf(reg_results.tvalues[1 + param_index], deg_f)
                conf_intervals = (
                    reg_results.conf_int(alpha=self.alpha * 2)
                    .iloc[1 + param_index]
                    .values[0],
                    np.inf,
                )
            else:
                p_value = t.cdf(reg_results.tvalues[1 + param_index], deg_f)
                conf_intervals = (
                    -np.inf,
                    reg_results.conf_int(alpha=self.alpha * 2)
                    .iloc[1 + param_index]
                    .values[1],
                )

        return {
            "metric": metric,
            "control name": self.control_variant_name,
            "variant name": variant,
            "control": control,
            "variant": variation,
            "ate": ate,
            "p_value": p_value,
            "lower confidence boundary": conf_intervals[0],
            "upper confidence boundary": conf_intervals[1],
            "Significant": p_value < self.alpha,
            "deg_f": deg_f,
        }

    @validate_one_test_version
    def ttest(self, data: pd.DataFrame, metric: str) -> Dict:

        t_test = ttest_ind(
            data[data[self.treatment_col] == self.variants[0]][metric],
            data[data[self.treatment_col] == self.control_variant_name][metric],
            alternative=self.alternative,
        )

        p_value = t_test[1]

        conf_intervals = t_test.confidence_interval(confidence_level=1 - self.alpha)[
            0:2
        ]

        deg_f = t_test.df

        variation, control = (
            data[data[self.treatment_col] == self.variants[0]][metric].mean(),
            data[data[self.treatment_col] == self.control_variant_name][metric].mean(),
        )
        ate = variation - control

        yield {
            "metric": metric,
            "control name": self.control_variant_name,
            "variant name": self.variants[0],
            "control": control,
            "variant": variation,
            "ate": ate,
            "p_value": p_value,
            "lower confidence boundary": conf_intervals[0],
            "upper confidence boundary": conf_intervals[1],
            "Significant": p_value < self.alpha,
            "deg_f": deg_f,
        }

    @validate_only_binomial
    @validate_one_test_version
    def ztest(self, data: pd.DataFrame, metric: str):

        control_count = data[data["variant"] == self.control_variant_name][metric].sum()
        variant_count = data[data["variant"] == self.variants[0]][metric].sum()
        control_nobs = len(data[data["variant"] == self.control_variant_name])
        variant_nobs = len(data[data["variant"] == self.variants[0]])
        control, variation = control_count / control_nobs, variant_count / variant_nobs
        ate = variation - control

        z_alternative = {
            "greater": "larger",
            "less": "smaller",
            "two-sided": "two-sided",
        }[self.alternative]
        z_test = proportions_ztest(
            count=[control_count, variant_count],
            nobs=[control_nobs, variant_nobs],
            alternative=z_alternative,
        )

        p_value = z_test[1]
        z_critical = norm.ppf(1 - self.alpha / self.tails)

        margin = abs(
            z_critical
            * np.sqrt(
                variation * (1 - variation) / variant_nobs
                + control * (1 - control) / control_nobs
            )
        )

        conf_intervals = ate - margin, ate + margin

        yield {
            "metric": metric,
            "control name": self.control_variant_name,
            "variant name": self.variants[0],
            "control": control,
            "variant": variation,
            "ate": ate,
            "p_value": p_value,
            "lower confidence boundary": conf_intervals[0],
            "upper confidence boundary": conf_intervals[1],
            "Significant": p_value < self.alpha,
            "deg_f": None,
        }

    def ols(self, data: pd.DataFrame, metric: str):

        data = self.encode_columns(data, cat_col=[self.treatment_col])

        if self.cluster_cols:
            regression_results = smf.ols(
                f"{metric} ~ {' + '.join(data.drop([metric] + self.cluster_cols, axis=1))}",
                data=data,
            ).fit(
                cov_type="cluster", cov_kwds={"groups": data[self.cluster_cols]}, disp=0
            )
        else:
            regression_results = smf.ols(
                f"{metric} ~ {' + '.join(data.drop(metric, axis=1))}", data=data
            ).fit(disp=0)

        for idx, variant in enumerate(self.variants):
            yield self.extract_regression_results(
                regression_results, data, metric, variant, idx
            )

    def GEE(self, data: pd.DataFrame, metric: str):

        family = sm.families.Gaussian()
        cov_struct = sm.cov_struct.Exchangeable()
        data = self.encode_columns(data, cat_col=[self.treatment_col])

        if self.cluster_cols:
            regression_results = sm.GEE.from_formula(
                f"{metric} ~ {' + '.join(data.drop([metric] + self.cluster_cols, axis=1))}",
                data=data,
                groups=data[self.cluster_cols],
                family=family,
                cov_struct=cov_struct,
            ).fit()
        else:
            raise ValueError("'cluster_cols' param should be defined when using GEE")

        for idx, variant in enumerate(self.variants):
            yield self.extract_regression_results(
                regression_results, data, metric, variant, idx
            )

    @validate_only_binomial
    def logit(self, data: pd.DataFrame, metric: str):

        data = self.encode_columns(data, cat_col=[self.treatment_col])

        if self.cluster_cols:
            regression_results = smf.logit(
                f"{metric} ~ {' + '.join(data.drop([metric] + self.cluster_cols, axis=1))}",
                data=data,
            ).fit(
                cov_type="cluster", cov_kwds={"groups": data[self.cluster_cols]}, disp=0
            )
        else:
            regression_results = smf.logit(
                f"{metric} ~ {' + '.join(data.drop(metric, axis=1))}", data=data
            ).fit(disp=0)
        for idx, variant in enumerate(self.variants):

            results_dict = self.extract_regression_results(
                regression_results, data, metric, variant, idx
            )
            results_dict = self.transform_log_odds(results_dict)

            yield results_dict

    @validate_only_binomial
    @validate_one_test_version
    def delta(self, data: pd.DataFrame, metric: str):

        if not self.cluster_cols:
            raise ValueError("'cluster_cols' has to be specified in delta method")

        control = data[data[self.treatment_col] == self.control_variant_name]
        variant = data[data[self.treatment_col] == self.variants[0]]

        variant_prop = variant[metric].mean()
        control_prop = control[metric].mean()
        delta = variant_prop - control_prop

        NiT = variant.groupby(self.cluster_cols).size().mean()
        Var_NiT = variant.groupby(self.cluster_cols).size().var()
        SiT = (
            variant.groupby(self.cluster_cols)[metric]
            .apply(lambda x: (x == 1).sum())
            .mean()
        )
        Var_SiT = (
            variant.groupby(self.cluster_cols)[metric]
            .apply(lambda x: (x == 1).sum())
            .var()
        )
        COV_SiT_NiT = np.cov(
            variant.groupby(self.cluster_cols).size().values,
            variant.groupby(self.cluster_cols)[metric]
            .apply(lambda x: (x == 1).sum())
            .values,
        )[0][1]
        NiC = control.groupby(self.cluster_cols).size().mean()
        Var_NiC = control.groupby(self.cluster_cols).size().var()
        SiC = (
            control.groupby(self.cluster_cols)[metric]
            .apply(lambda x: (x == 1).sum())
            .mean()
        )
        Var_SiC = (
            control.groupby(self.cluster_cols)[metric]
            .apply(lambda x: (x == 1).sum())
            .var()
        )
        COV_SiC_NiC = np.cov(
            control.groupby(self.cluster_cols).size().values,
            control.groupby(self.cluster_cols)[metric]
            .apply(lambda x: (x == 1).sum())
            .values,
        )[0][1]
        n_t = variant[self.cluster_cols].nunique().values[0]
        n_c = control[self.cluster_cols].nunique().values[0]
        var_pt = (
            1 / (NiT**2) * Var_SiT
            + (SiT**2) / (NiT**4) * Var_NiT
            - 2 * SiT / (NiT**3) * COV_SiT_NiT
        )
        var_pc = (
            1 / (NiC**2) * Var_SiC
            + (SiC**2) / (NiC**4) * Var_NiC
            - 2 * SiC / (NiC**3) * COV_SiC_NiC
        )

        p_value = norm.sf(delta / np.sqrt(var_pt / n_t + var_pc / n_c))
        conf_intervals = (
            round(
                delta
                + norm.ppf(self.alpha / self.tails)
                * np.sqrt(var_pt / n_t + var_pc / n_c),
                4,
            ),
            round(
                delta
                - norm.ppf(self.alpha / self.tails)
                * np.sqrt(var_pt / n_t + var_pc / n_c),
                4,
            ),
        )

        yield {
            "metric": metric,
            "control name": self.control_variant_name,
            "variant name": self.variants[0],
            "control": control_prop,
            "variant": variant_prop,
            "ate": delta,
            "p_value": p_value,
            "lower confidence boundary": conf_intervals[0],
            "upper confidence boundary": conf_intervals[1],
            "Significant": p_value < self.alpha,
            "deg_f": None,
        }

    def transform_log_odds(self, results_dict: Dict) -> Dict:
        """transforms log odds to probabilities"""

        for i in ["lower confidence boundary", "upper confidence boundary"]:
            results_dict[i] = results_dict[i] + results_dict["variant"]

        for i in [
            "control",
            "variant",
            "lower confidence boundary",
            "upper confidence boundary",
        ]:
            results_dict[i] = np.exp(results_dict[i]) / (1 + np.exp(results_dict[i]))

        for i in ["lower confidence boundary", "upper confidence boundary"]:
            results_dict[i] = results_dict[i] - results_dict["variant"]

        results_dict["ate"] = results_dict["variant"] - results_dict["control"]

        return results_dict
