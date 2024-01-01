import pandas as pd
import numpy as np
from config import PARAMS_DICT
from tests import Tests





class Analysis(Tests):

    """class for analysis of experiment results""" 

    @Tests.validate_selected_arguments(params_dict = PARAMS_DICT)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        self.methods_funcs_dict = {}

    def __repr__(self):
        return str(self.__dict__)
    
    def drop_null_rows(self, metric):
        data = self.data.copy()
        return data[data[metric].notnull()]
    
    def filter_unnecessary_columns(self, data:pd.DataFrame, metric:str) -> pd.DataFrame:
        """Drops columns which won't be used for the current test and metric"""
        columns = [metric] + [self.treatment_col]
        if self.covariates:
            columns.extend(self.covariates)
        if self.cluster_cols:
            columns.extend(self.cluster_cols)

        return data[columns]

       
    def select_method_func(self, **kwargs):
        """selects the appropriate function according to stat method chosen"""

        methods_funcs_dict = {
            't-test': super().ttest(**kwargs),
            'ols': super().ols(**kwargs)
        }
        
        return methods_funcs_dict[self.method]
    

    def analysis(self):
        for metric in self.metrics:
            data = self.drop_null_rows(metric)
            data = self.filter_unnecessary_columns(data, metric)
            
            for idx, i in enumerate(self.select_method_func(data=data, metric=metric)):
                self.results = pd.concat([self.results, pd.DataFrame(i, index=[idx])], axis=0)
        

        if self.correction:
            self.results = self.apply_correction(self.results)
                
        return self.results