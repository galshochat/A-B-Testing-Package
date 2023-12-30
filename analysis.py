from typing import List, Optional, Literal

class Analysis:

    """class for analysis of experiment results"""

    def __init__(self, df , metrics: List[str],
                 test:str = 't-test',
                 treatment_col: str = None,
                 cluster_cols: Optional[List[str]] = None,
                 alpha: float = 0.05,
                 correction: Optional[Literal['bonferroni','hb']] = None,
                 alternative: Literal['two-sided','greater','less'] = 'two-sided',
                 covariates: Optional[List[str]] = None):
        
        self.data = df
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.alpha = alpha
        self.correction = correction
        self.alternative = alternative
        self.covaariates = covariates

    def __repr__(self):
        return self.__dict__
    

    
