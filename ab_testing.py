from typing import List, Optional
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import optimize
import pprint


class Gaussian:
    def __init__(
        self,
        n: int = None,
        delta: float = None,
        alpha: float = None,
        power: float = None,
        sd: float = None,
        two_sided: bool = True,
    ):
        # method that initializes the instance of the Normal class
        """n - minimal sample size for one group
        delta - effect size
        alpha - significance level (Type I error)
        power - power of the test while 0.8 being the default  (1 minus Type II error)
        sd    - standard deviation in the control group
        tails - default two sided test
        """
        # if is None:
        #   raise Exception("Parameter mean_control_group must be specified")

        if (
            sum([n is None, delta is None, alpha is None, power is None, sd is None])
            != 1
        ):
            print(
                sum(
                    [n is None, delta is None, alpha is None, power is None, sd is None]
                )
            )
            raise Exception(
                "One and only one of the parameters n, delta, alpha, power, sd must be None"
            )
        elif np.abs(any([alpha, power])) > 1:
            raise ValueError(
                "any of the arguments alpha, power must be within range 0-1"
            )
        else:
            self.delta = delta
            self.power = power
            self.alpha = alpha
            self.sd = sd
            self.two_sided = two_sided
            self.z_score_a = (
                norm.ppf(1 - self.alpha / (1 if self.two_sided is False else 2))
                if alpha is not None
                else None
            )
            self.z_score_b = norm.ppf(self.power) if power is not None else None

            # sample size attribute will be the floor of the provided argument
            # (integer of sample available) or the ceiling of the calculated value
            # (based on other parameters).
            self.n = self.Sample_Size_() if n is None else np.floor(n)
            self.power = self.Power_() if self.power is None else power
            self.delta = self.Delta_() if self.delta is None else delta
            self.alpha = self.Alpha_() if self.alpha is None else alpha
            self.sd = self.SD_() if self.sd is None else sd
            self.params = {
                "n": int(np.ceil(self.n)),
                "delta": round(self.delta, 4),
                "alpha": round(self.alpha, 4),
                "power": round(self.power, 4),
                "sd": round(self.sd, 4),
                "two-sided experiment": self.two_sided,
            }

    def Sample_Size_(self) -> int:
        # method that returns the minimal sample size of one group given other arguments

        n = (
            2
            * self.sd**2
            * (self.z_score_a + self.z_score_b) ** 2
            / (self.delta) ** 2
        )
        return n

    def Power_(self) -> float:
        # method that returns the power of the test given other arguments

        self.z_score_b = (
            np.sqrt((self.n * (self.delta) ** 2) / (2 * self.sd**2)) - self.z_score_a
        )
        power = norm.cdf(self.z_score_b)
        return power

    def Alpha_(self) -> float:
        """method that returns the probability of type 1 error (p_value) of the test
        given other arguments"""

        self.z_score_a = (
            np.sqrt((self.n * (self.delta) ** 2) / (2 * self.sd**2)) - self.z_score_b
        )
        alpha = self.alpha = (1 - norm.cdf(self.z_score_a)) * (
            1 if self.two_sided is False else 2
        )
        return alpha

    def Delta_(self) -> float:
        # method that returns the delta of the test given other arguments

        delta = np.sqrt(
            (2 * self.sd**2 * (self.z_score_a + self.z_score_b) ** 2) / self.n
        )
        return delta

    def SD_(self) -> float:
        # method that returns the standard deviation of the test given other arguments

        sd = np.sqrt(
            self.n * (self.delta) ** 2 / (2 * (self.z_score_a + self.z_score_b) ** 2)
        )
        return sd


class Binomial:
    def __init__(
        self,
        n: int = None,
        p1: float = None,
        delta: float = None,
        alpha: float = None,
        power: float = None,
        two_sided: bool = True,
    ):

        # method that initializes the instance of the Binomial class
        """n - minimal sample size for one group
        p - probability of success in control group
        delta - effect size
        alpha - significance level (Type I error)
        power - power of the test while 0.8 being the default  (1 minus Type II error)
        tails - default two sided test
        """

        if (
            sum([p1 is None, n is None, delta is None, alpha is None, power is None])
            != 1
        ):
            raise Exception(
                "One and only one of the parameters n, delta, alpha, power must be None"
            )
        elif np.abs(any([alpha, power])) > 1:
            raise ValueError(
                "any of the arguments alpha, power must be within range 0-1"
            )
        elif delta is not None and np.abs(delta) > 1:
            raise ValueError("delta must be within range 0-1")
        else:
            self.p1 = p1
            self.delta = delta
            self.power = power
            self.alpha = alpha
            self.two_sided = two_sided
            self.p2 = (
                self.p1 + self.delta
                if self.delta is not None and self.p1 is not None
                else None
            )
            self.z_score_a = (
                norm.ppf(1 - self.alpha / (1 if self.two_sided is False else 2))
                if alpha is not None
                else None
            )
            self.z_score_b = norm.ppf(self.power) if power is not None else None

            # sample size attribute will be the floor of the provided argument (integer of
            # sample available) or the ceiling of the calculated value (based on other parameters).
            self.n = self.Sample_Size() if n is None else np.floor(n)
            self.p1 = self.Probability_Control() if p1 is None else p1
            self.power = self.Power() if self.power is None else power
            self.delta = self.Delta() if self.delta is None else delta
            self.alpha = self.Alpha() if self.alpha is None else alpha
            self.params = {
                "n": int(np.ceil(self.n)),
                "p1": self.p1,
                "delta": self.delta,
                "alpha": self.alpha,
                "power": self.power,
                "two-sided experiment": self.two_sided,
            }

    def Sample_Size(self) -> int:
        # method that returns the minimal sample size of one group given other arguments
        n = (
            self.z_score_a
            * np.sqrt((self.p1 + self.p2) * (1 - self.p1 + 1 - self.p2) / 2)
            + self.z_score_b
            * np.sqrt((self.p1 * (1 - self.p1)) + (self.p2 * (1 - self.p2)))
        ) ** 2 / self.delta**2

        # http://meteo.edu.vn/GiaoTrinhXS/e-book/PQ220-6234F.Ch-10.pdf
        return n

    def Probability_Control(self) -> float:
        """method that returns the probability of success in control group"""
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        def func(p1):
            return (
                self.z_score_a
                * np.sqrt((2 * p1 + self.delta) * (1 - p1 + 1 - p1 - self.delta) / 2)
                + self.z_score_b
                * np.sqrt((p1 * (1 - p1)) + ((p1 + self.delta) * (1 - p1 - self.delta)))
            ) ** 2 / self.delta**2 - self.n

        p1 = round(optimize.bisect(func, a=0, b=1), 2)
        warnings.filterwarnings("always", category=RuntimeWarning)
        return p1

    def Power(self) -> float:
        """method that returns the power of the test given other arguments"""
        self.z_score_b = (
            (
                np.sqrt(self.n * self.delta**2)
                - self.z_score_a
                * np.sqrt((self.p1 + self.p2) * (1 - self.p1 + 1 - self.p2) / 2)
            )
            * 1
            / np.sqrt((self.p1 * (1 - self.p1)) + (self.p2 * (1 - self.p2)))
        )
        power = norm.cdf(self.z_score_b)
        return power

    def Alpha(self) -> float:
        """method that returns the probability of type 1 error (p_value)
        of the test given other arguments"""
        self.z_score_a = (
            np.sqrt(self.n * self.delta**2)
            - self.z_score_b
            * np.sqrt((self.p1 * (1 - self.p1)) + (self.p2 * (1 - self.p2)))
        ) / np.sqrt((self.p1 + self.p2) * (1 - self.p1 + 1 - self.p2) / 2)
        alpha = self.alpha = (1 - norm.cdf(self.z_score_a)) * (
            1 if self.two_sided is False else 2
        )
        return alpha

    def Delta(self) -> float:
        """method that returns the delta of the test given other arguments"""
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        def func(delta):
            return (
                self.z_score_a
                * np.sqrt(
                    (2 * self.p1 + delta) * (1 - self.p1 + 1 - self.p1 - delta) / 2
                )
                + self.z_score_b
                * np.sqrt(
                    (self.p1 * (1 - self.p1))
                    + ((self.p1 + delta) * (1 - self.p1 - delta))
                )
            ) ** 2 / delta**2 - self.n

        delta = round(optimize.bisect(func, a=0, b=1 - self.p1), 4)
        warnings.filterwarnings("always", category=RuntimeWarning)
        return delta


class ab_testing(Binomial, Gaussian):
    def __init__(
        self,
        data: pd.DataFrame = None,
        metric: str = None,
        distribution: str = None,
        date_column: str = None,
        experiment_unit_column: str = None,
        n: int = None,
        p1: float = None,
        delta: float = None,
        alpha: float = None,
        power: float = None,
        sd: float = None,
        two_sided: bool = True,
        num_comparisons: int = 1,
    ):

        self.data = data
        self.num_comparisons = num_comparisons
        alpha = alpha / self.num_comparisons if alpha is not None else None

        if distribution in ["Binomial", "Gaussian", "Normal"]:
            self.distribution = distribution
        else:
            raise ValueError('Distribution must be "Binomial" , "Gaussian" or "Normal"')
        if data is not None:

            self.metric = metric
            self.date_column = date_column
            self.experiment_unit_column = experiment_unit_column
            if distribution == "Binomial":
                self.p1 = self.data[self.metric].mean()
            else:
                self.sd = np.std(self.data[self.metric].values, ddof=1)

            self.alpha = alpha
            self.power = power
            self.two_sided = two_sided
            self.delta = delta

        else:
            if distribution == "Binomial":
                Binomial.__init__(self, n, p1, delta, alpha, power, two_sided)
            elif distribution in ("Normal", "Gaussian"):
                Gaussian.__init__(self, n, delta, alpha, power, sd, two_sided)

    def __str__(self):
        """method which returns the parameters and arguments of the ab-testing instance"""
        if self.data is None:
            self.params["distribution"] = self.distribution
            if self.num_comparisons > 2:
                self.params["number of comparisons"] = self.num_comparisons
                self.params["alpha"] = self.alpha * self.num_comparisons
                self.params[
                    "correction"
                ] = """Bonferroni correction to family wise alpha was
                applied for every pairwise comparison"""
            return pprint.pformat(width=100, object=self.params, sort_dicts=False)
        else:
            self.params = {
                "alpha": self.alpha * self.num_comparisons,
                "power": self.power,
                "two-sided experiment": self.two_sided,
            }
            return pprint.pformat(width=100, object=self.params, sort_dicts=False)

    def MDE(
        self,
        events_per_time_unit_and_variant: int = None,
        minimal_effect: float = None,
        maximum_effect: float = None,
        effect_step: float = None,
        two_sided: bool = True,
        plot: bool = False,
        save_path: bool = None,
        **plot_kwargs,
    ):
        """method which returns the Minimal Detectable Effect experiment time
        estimation for an array of effect values."""

        # produces an array of delta values
        if not all([minimal_effect, maximum_effect, effect_step]):
            raise ValueError(
                "The values of minimal_effect, maximal_effect, effect_step cannot be None"
            )
        # array of effect size values
        effect_magnitudes = np.arange(
            minimal_effect, maximum_effect + effect_step, effect_step
        )
        # list of experiment days needed for corresponding effect
        days = []
        sizes = []

        if self.data is not None:

            # calculates number of time units present in the dataset
            length_input_data = self.data[self.date_column].nunique()
            # calculates the floor of events per time unit
            self.events_per_time_unit_and_variant = np.floor(
                len(self.data[(self.data)[self.metric].notnull()])
                / length_input_data
                / (self.num_comparisons + 1)
            )
            # the metric of interest base value
            pre_experiment_value = self.data[self.metric].mean()

        else:
            if events_per_time_unit_and_variant is None:
                raise ValueError(
                    """if data with temporal dimension is not provided,
                      number of events per time unit and variant must be specified"""
                )

            else:
                self.events_per_time_unit_and_variant = events_per_time_unit_and_variant

        for i in effect_magnitudes:
            self.delta = i

            if self.distribution in ("Binomial"):

                self.p1 = self.p1 if self.p1 is not None else pre_experiment_value
                self.p2 = self.p1 + self.delta
                Binomial.__init__(
                    self,
                    n=None,
                    p1=self.p1,
                    delta=self.delta,
                    alpha=self.alpha / self.num_comparisons,
                    power=self.power,
                    two_sided=self.two_sided,
                )
                n = self.n
            else:
                Gaussian.__init__(
                    self,
                    n=None,
                    sd=self.sd,
                    delta=self.delta,
                    alpha=self.alpha / self.num_comparisons,
                    power=self.power,
                    two_sided=self.two_sided,
                )
                n = self.n
            sizes.append(int(np.ceil(n)))
            days.append(int(np.ceil(n / self.events_per_time_unit_and_variant)))

        results = pd.DataFrame(
            {
                "effect": effect_magnitudes,
                "sample size per group": sizes,
                "length of experiment": days,
            }
        )
        self.delta = effect_magnitudes
        self.n = sizes
        self.experiment_length = days

        if not plot:
            self.mde = results
            return self.mde
        else:
            self.mde = self.Plot_MDE(save_path, results, **plot_kwargs)
            return self.mde

    def Plot_MDE(
        self,
        save_path: bool = None,
        results: pd.DataFrame = None,
        figsize: tuple = (10, 5),
        title: str = "MDE",
        marker: str = "o",
        ls: str = "-",
        annot: bool = True,
        **plot_kwargs,
    ):

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.grid(visible=True, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Length of Experiment")
        ax.set_ylabel("Effect Δ")
        ax.plot(
            results["length of experiment"].values,
            results["effect"].values,
            marker=marker,
            ls=ls,
            **plot_kwargs,
        )
        if annot is True:
            for i in range(len(results)):
                plt.annotate(
                    xy=(
                        results["length of experiment"].values[i]
                        + np.min([25, results["length of experiment"].min()]),
                        results["effect"].values[i],
                    ),
                    text=results["length of experiment"].values[i],
                )

        ax.set_xlim(
            -1
            * (
                results["length of experiment"].max() * 1.05
                - results["length of experiment"].max()
            ),
            results["length of experiment"].values[i]
            + results["length of experiment"].max() * 1.05,
        )
        fig = ax.figure
        if save_path is not None:
            fig.savefig(save_path)
        else:
            plt.close()
            return fig

    def clustered_power(
        self,
        cluster_col: str,
        num_variants: int = None,
        covariates: Optional[List[str]] = None,
        delta: float = None,
        iterations: int = 200,
        verbose: bool = True,
    ):
        import statsmodels.api as sm

        num_variants = (
            num_variants if num_variants is not None else self.num_comparisons
        )
        family = sm.families.Gaussian()
        cov_struct = sm.cov_struct.Exchangeable()
        if covariates:
            cov_data = self.data[covariates].copy()

        def allocate_treatment(data, cluster_col, num_variants):
            clusters = data[cluster_col].unique()
            treatments = [np.random.choice(num_variants, 1)[0] for i in clusters]
            clusters = pd.DataFrame({"clusters": clusters, "treatments": treatments})
            clusters = pd.get_dummies(clusters, columns=["treatments"], drop_first=True)
            data = data.merge(
                clusters, how="inner", left_on=cluster_col, right_on="clusters"
            )
            data = data.drop(["clusters"], axis=1)
            return data

        def add_effect(data, delta):
            data[self.metric] = np.where(
                data.iloc[:, 2] == 1, data[self.metric] + delta, data[self.metric]
            )
            return data

        num_significant = 0
        n = 1
        alpha = self.alpha * self.num_comparisons
        delta = delta if delta is not None else self.delta
        if delta is None:
            raise ValueError(
                "Delta must be specified when instantiating the class or when calling the method"
            )
        for i in range(iterations):
            data = self.data[[cluster_col, self.metric]].copy()
            num_variants = (
                num_variants if num_variants is not None else self.num_comparisons + 1
            )
            data = add_effect(
                allocate_treatment(data, cluster_col, num_variants), delta
            )
            if covariates:
                data = pd.concat([data, cov_data], axis=1)
            model = sm.GEE.from_formula(
                f'{self.metric} ~ {" + ".join(data.iloc[:,2:].columns)}',
                data=data,
                groups=cluster_col,
                family=family,
                cov_struct=cov_struct,
            )
            result = model.fit()
            num_significant += result.pvalues[1] < alpha
            if verbose:
                print(f"Iteration {n}")
            n += 1

        print(f"The power of the test is {num_significant/iterations}")
