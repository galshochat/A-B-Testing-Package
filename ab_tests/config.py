PARAMS_DICT = {
    "method": ["t-test", "ols", "logistic-regression", "z-test", "gee", "delta"],
    "alternative": ["two-sided", "greater", "less"],
    "correction": [
        "bonferroni",
        "fdr_bh",
        "holm",
        "sidak",
        "holm-sidak",
        "simes-hochberg",
        "hommel",
        "fdr_by",
        "fdr_tsbh",
        "fdr_tsbky",
    ],
    "plotting_strategy": ["ground_truth", "controlled"],
}
