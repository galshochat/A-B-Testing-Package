import pandas as pd
import numpy as np
from config import PARAMS_DICT
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Literal, Optional
from stattests import StatTests
from scipy.stats import norm, t
from visualization import Visualization


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
        time_col=Optional[str],
        tests: List[Dict] = None,
        treatment_col: str = None,
        control_variant_name: str = "control",
        alpha: float = 0.05,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        correction: Optional[Literal["bonferroni", "fdr_bh"]] = None,
        early_stopping_proportion: Optional[float] = None,
    ):

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
        results["adjusted_pvalues"] = adjusted[1]

        return self.correct_confidence_intervals(results)

    def correct_confidence_intervals(self, results: pd.DataFrame) -> pd.DataFrame:

        for idx, i in results.iterrows():
            for bound in ["lower confidence boundary", "upper confidence boundary"]:
                if ~np.isnan(i["deg_f"]):
                    st_error = (i[bound] - i["ate"]) / t.ppf(
                        self.alpha / self.tails, df=i["deg_f"]
                    )
                    adjusted_alpha = self.alpha * i["p_value"] / (i["adjusted_pvalues"])
                    results.at[idx, bound] = (
                        i["ate"]
                        + t.ppf(adjusted_alpha / self.tails, df=i["deg_f"]) * st_error
                    )
                else:
                    st_error = (i[bound] - i["ate"]) / norm.ppf(self.alpha / self.tails)
                    adjusted_alpha = self.alpha * i["p_value"] / (i["adjusted_pvalues"])
                    results.at[idx, bound] = (
                        i["ate"] + norm.ppf(adjusted_alpha / self.tails) * st_error
                    )

        return results

    def prepare_visualization_data(self, test):
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
                        ["metric", "control name", "control", "date"]
                    ].drop_duplicates(keep="first"),
                ],
                ignore_index=True,
                axis=0,
            )
            daily_versions_results = pd.concat(
                [
                    daily_versions_results,
                    daily_experiment.result[
                        ["metric", "variant name", "variant", "date"]
                    ],
                ],
                ignore_index=True,
                axis=0,
            )

        daily_control_results = daily_control_results.rename(
            columns={"control name": "variant name", "control": "variant"}
        )

        daily_results = pd.concat(
            [daily_control_results, daily_versions_results], ignore_index=True, axis=0
        )

        return daily_results

    def analysis(self, visualize=True):
        """orchestrates the testing and multiple comparisons corrections"""
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

            if visualize:

                metric_viz_stats = self.prepare_visualization_data(test)

                plot = Visualization.lineplot(
                    data=metric_viz_stats,
                    x="date",
                    y="variant",
                    color="variant name",
                    title=test["metric"],
                    xlabel="date",
                    ylabel="",
                )
                print(plot)
                print(pd.DataFrame(experiment.result))

            self.results = pd.concat(
                [self.results, experiment.result], ignore_index=True, axis=0
            )

        if self.correction:
            self.results = self.apply_correction(self.results)

        return self
