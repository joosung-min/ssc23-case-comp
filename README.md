## The Causal impacts of climate change on Canadian industrial productions: An Instrumental variable analysis using geographical features as Instruments

* [Joosung (Sonny) Min](https://www.linkedin.com/in/joosung-sonny-min-35370b9b/), and [Donghui Son](https://www.linkedin.com/in/%EB%8F%99%ED%9D%AC-%EC%86%90-143054224/) (supervisor: Dr. [Lloyd T. Elliott](https://elliottlab.ca/))

Please refer to [A22_iv_tavg_example.ipynb](/A22_iv_tavg_example.ipynb) for an example of our analysis.


![This is our poster](/poster_SFU.png)



### INTRODUCTION
Instrumental variable analysis is a powerful method used to estimate causal effects for cases where randomized controlled trials are not feasible. By introducing exogenous instrumental variables (IVs) that affect the treatment variable but not the outcome variable, unmeasured confounders can be controlled for, and the average treatment effect on the outcome (ATE) can be computed. This method has found widespread applications in fields such as economics, epidemiology, and social sciences. Previous studies have demonstrated the robustness of geographical features, such as longitude and latitude, as IVs for which temperatures are treatment variables. In this study, we employ a commonly used IV method called two-stage least squares (IV-2SLS), which involves linear regression from the IVs to the treatment (temperatures) in the first step, and from the conditional expectation of the treatment to the outcome (industrial productions) in the second step.

### OBJECTIVE
This study aims to determine the causal impacts of climate change on industrial production in Canadian census subdivisions. We analyze three treatment factors: the monthly average temperature and the total number of extreme maximum and minimum temperature occurrences throughout the year. Our study utilizes publicly available data from Statistics Canada, Environment and Climate Change Canada, and the Bank of Canada to conduct all analyses.

### METHODS
IV-2SLS was utilized to determine the potential causal impact of average temperature changes on industrial production within Canadian census subdivisions. The analysis controlled for covariates known to influence industrial production, such as commodity prices (metals, agricultural products, oil, etc.) and the CAD-USD currency rate. Prior to conducting IV-2SLS, principal component analysis was performed to select the first five PC loadings, which accounted for 98% of the covariate variability. Additionally, population, year, and season were incorporated as covariates. The latitude and longitude of census subdivisions served as instrumental variables, and the average effects of treatments on log(production) were calculated for each of the 14 industries and 10 provinces. We employed Wu-Hausman's test to assess endogeneity ($H_0$: All endogenous variables are exogenous). If the hypothesis was not rejected, we set ATE(treatment) to 0 for the respective {treatment, industry, province} set. For extreme temperatures, we determined the occurrences of "Extr Max Temp Flag" and "Extr Min Temp Flag" throughout the year from the weather station data provided by Environment and Climate Change Canada and then utilized them as treatments.

### RESULTS
The census subdivisions (CSDs) that exhibited the greatest benefits from increasing monthly average temperature (tavg) were those with the "Agriculture, forestry, fishing, hunting, mining, quarrying, and oil and gas extraction" industry as the dominant sector in Saskatchewan (ATE(tavg)=0.40) and Manitoba (0.30), as well as the "Professional, scientific, and technical services" industry in Prince Edward Island (0.39). Conversely, CSDs primarily engaged in the "Utilities" sector in Nova Scotia (-0.67) and the "Manufacturing” industry in Prince Edward Island (-0.56) experienced negative impacts due to increasing monthly average temperature.

While the majority of industrial productions across the provinces remained unaffected by the occurrences of extreme maximum and minimum temperatures throughout the year, two of the major industries in Saskatchewan demonstrated adverse effects from extreme maximum temperature, including "Agriculture, forestry, fishing, hunting, mining, quarrying, and oil and gas extraction" (ATE(extr_max_temp)= -1.57), and "Finance and insurance, real estate and rental and leasing" (-2.10), which contrasts with their positive ATE(tavg). These results suggest that a gradual increase in average temperature may have positive impacts, while sudden temperature fluctuations, whether increases or decreases, could impede industrial activities. 

### DISCUSSION
A limitation of this study that we acknowledge is that the analysis was limited to 14 NAICS industries. A more comprehensive investigation with more segmented industry data may be necessary to obtain more accurate ATE estimates. Also, the IV-2SLS method assumes strong linear associations between variables. Employing more flexible estimators may be necessary for more accurate ATE estimation. Nonetheless, our study showcases the effectiveness of the causal inference method in revealing the impacts of climate change and establishing a solid groundwork for future research.

### CONCLUSION
Our study showcases the potential of instrumental variable analysis for assessing the causal effects of climate change on industrial production. The results reveal a mixed impact, with both positive and negative effects depending on the industry and region. It emphasizes the need to consider various temperature measures and the variation in effects across industries and regions when evaluating climate change's influence on industrial production. Policymakers should tailor interventions to the specific characteristics of each industry and region to mitigate adverse impacts. 

<br />
