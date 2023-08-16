import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime
from scipy import optimize
import pprint


class Gaussian():
    
    def __init__(self, n=None, delta=None, alpha=None, power=None, sd=None,  two_sided=True):
        # method that initializes the instance of the Normal class
        """ n - minimal sample size for one group
        delta - effect size 
        alpha - significance level (Type I error)
        power - power of the test while 0.8 being the default  (1 minus Type II error)
        sd    - standard deviation in the control group
        tails - default two sided test 
        """
        #if is None:
         #   raise Exception("Parameter mean_control_group must be specified")  

        if sum([n is None, delta is None,alpha is None,power is None,sd is None])!=1:
            print(sum([n is None, delta is None, alpha is None, power is None, sd is None]))
            raise Exception("One and only one of the parameters n, delta, alpha, power, sd must be None")
        elif np.abs(any([alpha, power]))>1:
            raise ValueError("any of the arguments alpha, power must be within range 0-1") 
        else:
            self.delta=delta
            self.power=power
            self.alpha=alpha
            self.sd=sd
            self.two_sided = two_sided
            self.z_score_a=norm.ppf(1-self.alpha/(1 if self.two_sided==False else 2)) if alpha is not None else None
            self.z_score_b=norm.ppf(self.power) if power is not None else None
            #self.mean_test_group=self.delta if self.delta is None else self.mean_control_group + self.delta
            
            # sample size attribute will be the floor of the provided argument (integer of sample available) or the ceiling of the calculated value (based on other parameters).
            self.n=self.Sample_Size_() if n is None else np.floor(n)
            self.power=self.Power_() if self.power is None else power
            self.delta=self.Delta_() if self.delta is None else delta
            self.alpha=self.Alpha_() if self.alpha is None else alpha
            self.sd=self.SD_() if self.sd is None else sd
            self.params={'n': int(np.ceil(self.n)), 'delta': round(self.delta,4), 'alpha': round(self.alpha,4), 'power': round(self.power,4), 'sd': round(self.sd, 4), 'two-sided experiment': self.two_sided}

    #def params_(self):
         # method which returns the parameters and arguments of the class
     #    return "n: {}, delta: {}, alpha: {}, power: {}, standard deviation: {}, two-sided experiment: {}".format(self.n,  self.delta, self.alpha, self.power, self.sd, self.two_sided)
     
    def Sample_Size_(self)-> int:
         # method that returns the minimal sample size of one group given other arguments

        n=2*self.sd**2 * (self.z_score_a + self.z_score_b)**2 / (self.delta)**2
        return n
    
    def Power_(self)-> float:
        # method that returns the power of the test given other arguments

        self.z_score_b = np.sqrt((self.n * (self.delta)**2)/(2*self.sd**2)) - self.z_score_a 
        power = norm.cdf(self.z_score_b)
        return power
    
    def Alpha_(self)-> float:
        # method that returns the probability of type 1 error (p_value) of the test given other arguments

        self.z_score_a= np.sqrt((self.n * (self.delta)**2)/(2*self.sd**2)) - self.z_score_b
        alpha=self.alpha=(1-norm.cdf(self.z_score_a)) * (1 if self.two_sided==False else 2)
        return alpha

    def Delta_(self)-> float:
        # method that returns the delta of the test given other arguments

        delta = np.sqrt((2*self.sd**2 * (self.z_score_a + self.z_score_b)**2)/self.n)
        return delta
    
    def SD_(self)-> float:
        # method that returns the standard deviation of the test given other arguments

        sd = np.sqrt(self.n * (self.delta)**2/(2* (self.z_score_a + self.z_score_b)**2))
        return sd
     


class Binomial(): 

    def __init__(self, n=None, p1=None, delta=None, alpha=None, power=None, two_sided=True):
 
        # method that initializes the instance of the Binomial class
        """ n - minimal sample size for one group
        p - probability of success in control group
        delta - effect size 
        alpha - significance level (Type I error)
        power - power of the test while 0.8 being the default  (1 minus Type II error)
        tails - default two sided test 
        """

        if sum([p1 is None, n is None, delta is None, alpha is None, power is None])!=1:
            raise Exception("One and only one of the parameters n, delta, alpha, power must be None")
        elif np.abs(any([alpha, power]))>1:
            raise ValueError("any of the arguments alpha, power must be within range 0-1")
        elif delta is not None and np.abs(delta)>1:
            raise ValueError("delta must be within range 0-1") 
        else:
            self.p1=p1
            self.delta=delta
            self.power=power
            self.alpha=alpha
            self.two_sided = two_sided
            self.p2=self.p1 + self.delta if self.delta is not None and self.p1 is not None else None
            self.z_score_a=norm.ppf(1-self.alpha/(1 if self.two_sided==False else 2)) if alpha is not None else None
            self.z_score_b=norm.ppf(self.power) if power is not None else None

            # sample size attribute will be the floor of the provided argument (integer of sample available) or the ceiling of the calculated value (based on other parameters).
            self.n=self.Sample_Size() if n is None else np.floor(n)
            self.p1=self.Probability_Control() if p1 is None else p1
            self.power=self.Power() if self.power is None else power
            self.delta=self.Delta() if self.delta is None else delta
            self.alpha=self.Alpha() if self.alpha is None else alpha
            self.params={"n": int(np.ceil(self.n)), "p1": self.p1, "delta": self.delta, "alpha": self.alpha, "power": self.power, "two-sided experiment": self.two_sided}
     

    def Sample_Size(self)-> int:
         # method that returns the minimal sample size of one group given other arguments
        n=(self.z_score_a * np.sqrt((self.p1+self.p2)*(1-self.p1 + 1-self.p2)/2) + self.z_score_b*np.sqrt((self.p1*(1-self.p1)) + (self.p2 * (1-self.p2))))**2/self.delta**2

        # http://meteo.edu.vn/GiaoTrinhXS/e-book/PQ220-6234F.Ch-10.pdf
        return n
    
    def Probability_Control(self)-> float:
        # method that returns the probability of success in control group
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        def func(p1):
            return (self.z_score_a * np.sqrt((2*p1+self.delta)*(1-p1 + 1-p1-self.delta)/2) + self.z_score_b*np.sqrt((p1*(1-p1)) + ((p1+self.delta) * (1-p1-self.delta))))**2/self.delta**2 - self.n
        p1=round(optimize.bisect(func, a=0, b=1), 2)
        warnings.filterwarnings("always", category = RuntimeWarning)
        return p1

    def Power(self)-> float:
        # method that returns the power of the test given other arguments
        self.z_score_b=(np.sqrt(self.n * self.delta**2) - self.z_score_a * np.sqrt((self.p1+self.p2)*(1-self.p1 + 1-self.p2)/2))* 1/np.sqrt((self.p1*(1-self.p1)) + (self.p2 * (1-self.p2))) 
        power=norm.cdf(self.z_score_b)
        return power
    
    def Alpha(self)-> float:
        # method that returns the probability of type 1 error (p_value) of the test given other arguments
        self.z_score_a=(np.sqrt(self.n * self.delta**2)-self.z_score_b*np.sqrt((self.p1*(1-self.p1)) + (self.p2 * (1-self.p2))))/np.sqrt((self.p1+self.p2)*(1-self.p1 + 1-self.p2)/2) 
        alpha=self.alpha=(1-norm.cdf(self.z_score_a)) * (1 if self.two_sided==False else 2)
        return alpha

    def Delta(self)-> float:
        # method that returns the delta of the test given other arguments
        warnings.filterwarnings("ignore", category = RuntimeWarning)
        def func(delta):
            return (self.z_score_a * np.sqrt((2*self.p1+delta)*(1-self.p1 + 1-self.p1-delta)/2) + self.z_score_b*np.sqrt((self.p1*(1-self.p1)) + ((self.p1+delta) * (1-self.p1-delta))))**2/delta**2 - self.n
        delta=round(optimize.bisect(func, a=0, b=1), 4)
        warnings.filterwarnings("always", category = RuntimeWarning)
        return delta


class ab_testing(Binomial,Gaussian):

    def __init__(self, data=None, metric=None, distribution=None ,date_column=None, experiment_unit_column=None, n=None, p1=None, delta=None, alpha=None, power=None, sd=None ,two_sided=True, num_comparisons=1):
        
        self.data=data
        self.num_comparisons=num_comparisons
        alpha=alpha/self.num_comparisons if alpha is not None else None
        
        if distribution in ["Binomial", "Gaussian", "Normal"]:
            self.distribution=distribution
        else: 
            raise ValueError('Distribution must be "Binomial" , "Gaussian" or "Normal"')
        if data is not None:
            
            self.metric=metric
            self.date_column=date_column
            self.experiment_unit_column=experiment_unit_column
            if distribution=='Binomial':
                self.p1=self.data[self.metric].mean()
            else:
                self.sd=np.std(self.data[self.metric].values, ddof=1)
                
            self.alpha=alpha
            self.power=power
            self.two_sided=two_sided
            
        else:
            if distribution=='Binomial':
                Binomial.__init__(self, n, p1, delta, alpha, power, two_sided)
            elif distribution in ('Normal','Gaussian'):
                Gaussian.__init__(self, n, delta, alpha, power, sd,  two_sided)

    def __str__(self):
        # method which returns the parameters and arguments of the ab-testing instance
         if self.data is None:
            self.params['distribution']=self.distribution
            if self.num_comparisons>2:
                self.params['number of comparisons']=self.num_comparisons
                self.params['alpha']=self.alpha*self.num_comparisons
                self.params['correction']='Bonferroni correction to family wise alpha was applied for every pairwise comparison'
            return pprint.pformat(width=100, object=self.params, sort_dicts=False)
         else:
            self.params={'alpha':self.alpha*self.num_comparisons, 'power':self.power, 'two-sided experiment': self.two_sided}
            return pprint.pformat(width=100, object=self.params, sort_dicts=False)
         
         

# method which returns the Minimal Detectable Effect experiment time estimation for an array of effect values.

    def MDE(self, events_per_time_unit_and_variant=None ,minimal_effect=None, maximal_effect=None, effect_step=None, two_sided=True, plot=False, save_path=None, **plot_kwargs):
        # produces an array of delta values
        if not all([minimal_effect,maximal_effect,effect_step]):
            raise ValueError('The values of minimal_effect, maximal_effect, effect_step cannot be None')
        #array of effect size values
        effect_magnitudes=np.arange(minimal_effect,maximal_effect + effect_step ,effect_step)
        #list of experiment days needed for corresponding effect
        days=[]
        sizes=[]

        if self.data is not None:

            #calculates number of time units present in the dataset
            length_input_data = self.data[self.date_column].nunique()
            #calculates the floor of events per time unit
            self.events_per_time_unit_and_variant=np.floor(len(self.data[(self.data)[self.metric].notnull()])/length_input_data/(self.num_comparisons+1))
            #the metric of interest base value
            pre_experiment_value=self.data[self.metric].mean()
            
        else:
            if events_per_time_unit_and_variant is None:
                raise ValueError('if data with temporal dimension is not provided, number of events per time unit and variant must be specified')
                
            else:
                self.events_per_time_unit_and_variant=events_per_time_unit_and_variant
            
        
        for i in effect_magnitudes:
            self.delta=i
            
            if self.distribution in ('Binomial'):
                
                self.p1=self.p1 if self.p1 is not None else pre_experiment_value
                self.p2=self.p1 + self.delta
                Binomial.__init__(self, n=None , p1=self.p1 , delta=self.delta, alpha=self.alpha/self.num_comparisons , power=self.power, two_sided=self.two_sided)
                n=self.n
            else:
                Gaussian.__init__(self, n=None , sd=self.sd , delta=self.delta, alpha=self.alpha/self.num_comparisons, power=self.power, two_sided=self.two_sided)
                n=self.n
            sizes.append(int(np.ceil(n)))
            days.append(int(np.ceil(n/self.events_per_time_unit_and_variant)))

        results=pd.DataFrame({'effect':effect_magnitudes,'sample size per group':sizes ,'length of experiment':days})
        self.delta=effect_magnitudes
        self.n=sizes
        self.experiment_length=days
        
        if not plot: 
            self.mde=results
            return self.mde
        else: 
            self.mde=self.Plot_MDE(save_path, results, **plot_kwargs)
            return self.mde
    
    def Plot_MDE(self, save_path=None ,results=None, figsize=(10,5), title='MDE', marker='o', ls='-',annot=True, **plot_kwargs):

        fig=plt.figure(figsize=figsize)
        ax=plt.axes()
        ax.grid(visible=True, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Length of Experiment')
        ax.set_ylabel(u'Effect Δ')
        ax.plot(results['length of experiment'].values, results['effect'].values, marker=marker, ls=ls, **plot_kwargs)
        if annot==True:
            for i in range(len(results)):
                plt.annotate(xy=(results['length of experiment'].values[i]+np.min([25,results['length of experiment'].min()]), results['effect'].values[i]), text=results['length of experiment'].values[i])
                
        ax.set_xlim(-1*(results['length of experiment'].max()*1.05-results['length of experiment'].max()), results['length of experiment'].values[i]+results['length of experiment'].max()*1.05)
        fig=ax.figure
        if save_path is not None:
            fig.savefig(save_path)
        else:
            plt.close()
            return fig
       

