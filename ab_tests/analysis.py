import pandas as pd
import numpy as np
from ab_tests.config import PARAMS_DICT
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Literal, Optional, Union
from ab_tests.stattests import StatTests
from scipy.stats import norm, t
from ab_tests.visualization import Visualization


class Analysis:

    """class for analysis of experiment results"""

    def validate_selected_arguments(params_dict):
        def validate(func):
            def wrapper(self, *args, **kwargs):
                for key, value in params_dict.items():
                    if key in kwargs:
                        if not kwargs[key] in value:
                            raise ValueError(
                                f"""{kwargs[key]} is not an allowed argument for the parameter {
                                    key}. Valid values are {', '.join(value)}"""
                            )
                    else:
                        for idx, exp in enumerate(kwargs["tests"]):
                            if key in exp:
                                if not exp[key] in value:
                                    raise ValueError(
                                        f"""{kwargs["tests"][idx][key]}
                                            is not an allowed argument for the parameter {
                                            key}. Valid values are {', '.join(value)}"""
                                    )
                obj = func(self, *args, **kwargs)
                return obj

            return wrapper

        return validate

    @validate_selected_arguments(params_dict=PARAMS_DICT)
    def __init__(
        self,
        df: pd.DataFrame = None,
        time_col: Optional[str] = None,
        tests: List[Dict[str, Union[str, list]]] = None,
        treatment_col: str = None,
        control_variant_name: str = "control",
        alpha: float = 0.05,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        correction: Optional[str] = None,
        early_stopping_proportion: Optional[float] = None,
        visualize: bool = True,
    ):
        """
        Params:

        - df: dataframe
        - time_col: str - optional timestamps/dates column used in visualization
        - tests: dictionary containing:
            - metric: str - name of the target column
            - method: str - statistical test to apply (one of those in config.py)
            - cluster_cols- list of strings. column names to cluster by.
            - covariates  - list of strings. additional confounders affecting the result.
            - treatment_col -string. Name of the treatment variable (can contain strings)
            - control_variant_name - string or number. Name of the control level in treatment
              variable
            - alpha: significance level
            - alternative: one of "two-sided", "greater", "less"
            - correction: string representing the correction method for multiple comparisons.
              Full list in config.py
            - early_stopping_proportion: between 0 and 1. Ratio of data seen to planned when
              the experiment stopped or peeked. Based on
              Lan-DeMets sequential boundaries based on approximation of O'Brian-Fleming GST func
            - visualize: boolean. Whether do draw a lineplot of the experiment.
        """

        self.data = df
        self.time_col = time_col
        self.tests = tests
        self.treatment_col = treatment_col
        self.control_variant_name = control_variant_name
        self.alpha = alpha
        self.correction = correction
        self.alternative = alternative
        self.tails = 2 if alternative == 2 else 1
        self.results = pd.DataFrame()
        self.visualize_plots = visualize

        if early_stopping_proportion:
            self.alpha = self.early_stopping(early_stopping_proportion)

        self.analysis()

    def __repr__(self):
        return str(self.__dict__)

    def early_stopping(self, early_stopping_proportion: float = None):
        """
        Implementing Lan-DeMets sequential boundaries based on approximation of O'Brian-Fleming GST
        function.
        t: proportion of data seen until the moment of peeking, in comparison to total expected data
        (non-clustered design)"""

        a = 2 - 2 * norm.cdf(
            norm.ppf(1 - (self.alpha / (2 * self.tails)))
            / np.sqrt(early_stopping_proportion)
        )

        return a * self.tails

    def apply_correction(self, results: pd.DataFrame) -> Dict:
        """applying multiple comparisons correction"""
        adjusted = multipletests(
            pvals=results["p_value"], alpha=self.alpha, method=self.correction
        )

        results["Significant"] = adjusted[0]
        results["adjusted pvalues"] = adjusted[1]

        return self.correct_confidence_intervals(results)

    def correct_confidence_intervals(self, results: pd.DataFrame) -> pd.DataFrame:

        for idx, i in results.iterrows():
            for bound in ["lower confidence boundary", "upper confidence boundary"]:
                if ~np.isnan(i["deg_f"]):
                    st_error = (i[bound] - i["ate"]) / t.ppf(
                        self.alpha / self.tails, df=i["deg_f"]
                    )
                    adjusted_alpha = self.alpha * i["p_value"] / (i["adjusted pvalues"])
                    results.at[idx, bound] = (
                        i["ate"]
                        + t.ppf(adjusted_alpha / self.tails, df=i["deg_f"]) * st_error
                    )
                else:
                    st_error = (i[bound] - i["ate"]) / norm.ppf(self.alpha / self.tails)
                    adjusted_alpha = self.alpha * i["p_value"] / (i["adjusted pvalues"])
                    results.at[idx, bound] = (
                        i["ate"] + norm.ppf(adjusted_alpha / self.tails) * st_error
                    )

        return results

    def prepare_viz_data(self, test: dict):
        """produces daily test data for visualization"""
        self.data[self.time_col] = self.data[self.time_col].dt.floor("d")

        daily_control_results = pd.DataFrame()
        daily_versions_results = pd.DataFrame()

        for i in self.data[self.time_col].unique():
            daily_data = self.data[self.data[self.time_col] == i]
            daily_experiment = StatTests(
                data=daily_data,
                **test,
                treatment_col=self.treatment_col,
                control_variant_name=self.control_variant_name,
                alpha=self.alpha,
                alternative=self.alternative,
                tails=self.tails,
            )
            daily_experiment.result["date"] = i
            daily_control_results = pd.concat(
                [
                    daily_control_results,
                    daily_experiment.result[
                        [
                            "metric",
                            "control name",
                            "control",
                            "control_intercept",
                            "date",
                        ]
                    ].drop_duplicates(keep="first"),
                ],
                ignore_index=True,
                axis=0,
            )
            daily_versions_results = pd.concat(
                [
                    daily_versions_results,
                    daily_experiment.result[
                        [
                            "metric",
                            "variant name",
                            "variant",
                            "variant_controlled",
                            "date",
                        ]
                    ],
                ],
                ignore_index=True,
                axis=0,
            )

        daily_control_results = daily_control_results.rename(
            columns={"control name": "variant name", "control": "variant"}
        )
        if "control_intercept" in daily_control_results.columns:
            daily_control_results = daily_control_results.rename(
                columns={"control_intercept": "variant_controlled"}
            )

        daily_results = pd.concat(
            [daily_control_results, daily_versions_results], ignore_index=True, axis=0
        )

        return daily_results

    def analysis(self):
        """orchestrates the testing and multiple comparisons corrections"""

        self.results = (
            pd.DataFrame()
        )  # emptying the results (if the method called again the previous results will be deleted)

        for idx, test in enumerate(self.tests):
            experiment = StatTests(
                data=self.data,
                **test,
                treatment_col=self.treatment_col,
                control_variant_name=self.control_variant_name,
                alpha=self.alpha,
                alternative=self.alternative,
                tails=self.tails,
            )
            experiment.result.set_index(
                [[idx] * experiment.result.shape[0]], inplace=True
            )
            self.results = pd.concat([self.results, experiment.result], axis=0)

        if self.correction:
            self.results = self.apply_correction(self.results)

        self.visualize()

        # return self

    def visualize(self):
        """orchestrates all operations relative to visualization"""
        # columns to render in the results table
        columns = list(
            filter(
                lambda x: x
                not in ["metric", "deg_f", "variant_controlled", "control_intercept"],
                self.results.columns,
            )
        )

        for idx, test in enumerate(self.tests):
            if self.visualize_plots:
                assert (
                    self.time_col is not None
                ), "'time_col' must be passed for visualization"
                metric_viz_stats = self.prepare_viz_data(test)

                plot = Visualization.lineplot(
                    data=metric_viz_stats,
                    x="date",
                    y="variant_controlled" or "variant",
                    color="variant name",
                    title=test["metric"],
                    xlabel="date",
                    ylabel="",
                )
                print(plot)

            metric_results = self.results.loc[[idx]][columns]

            Visualization.scorecard(metric_results, title=test["metric"])

        return self
