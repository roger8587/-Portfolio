import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import f
# =============================================================================
# t檢定
# =============================================================================
df = pd.read_excel('已分類資料.xlsx')
def t_test(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1)
    std2 = np.std(group2)
    nobs1 = len(group1)
    nobs2 = len(group2)
    
    modified_std1 = np.sqrt(np.float32(nobs1)/
                    np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/
                    np.float32(nobs2-1)) * std2
    #f檢定
    f1 = np.square(modified_std1)/np.square(modified_std2)
    fp = 1 - f.cdf(f1,nobs1-1,nobs2-1)
    if fp > 0.05:
        (statistic, pvalue) = stats.ttest_ind_from_stats( 
                   mean1=mean1, std1=modified_std1, nobs1=nobs1,   
                   mean2=mean2, std2=modified_std2, nobs2=nobs2,
                   equal_var = True)
    else:
        (statistic, pvalue) = stats.ttest_ind_from_stats( 
                   mean1=mean1, std1=modified_std1, nobs1=nobs1,   
                   mean2=mean2, std2=modified_std2, nobs2=nobs2,
                   equal_var = False)
    return [mean1,std1,mean2,std2,fp, statistic, pvalue]
df1 = df[-df['score'].isnull()]
a1 = df1[df1['brief']=='跨域模組']
a2 = df1[df1['brief']=='本科']
print(t_test(a1['score'].values,a2['score'].values),a1.shape[0],a2.shape[0])
b = [105,106,107,108]
g = []
for i in b:
    data = df1[df1['學年'] == i]
    a1 = data[data['brief']=='跨域模組']
    a2 = data[data['brief']=='本科']
    g.append(t_test(a1['score'].values,a2['score'].values))

df2 = pd.DataFrame(g,columns=['跨域模組平均', '跨域模組標準差', '本科平均','本科標準差','f-pvalue','t','t-pvalue'],index=b)
df2.to_excel('跨域與本科t檢定.xlsx')

# =============================================================================
# anova&tuckey
# =============================================================================
import pandas as pd

df = pd.read_excel('畢業成績.xlsx')
a1 = df['program1輔系']==1
a2 = df['program2雙主修']==1
a3 = df['program3學分學程']==1
a4 = df['program4跨域']==1
a5 = -a1 & -a2 & -a3 & -a4
df1 = df[a1].reset_index(drop=True)
df2 = df[a2].reset_index(drop=True)
df3 = df[a3].reset_index(drop=True)
df4 = df[a4].reset_index(drop=True)
df5 = df[a5].reset_index(drop=True)
df1['類別'] = '輔系'
df2['類別'] = '雙主修'
df3['類別'] = '學分學程'
df4['類別'] = '跨域'
df5['類別'] = '一般'
data = pd.concat([df1,df2,df3,df4,df5],axis = 0).reset_index(drop=True)
data = data.drop(['program1輔系','program2雙主修','program3學分學程','program4跨域'], axis = 1)
#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

lm = ols('gaverage ~ 類別',data = data).fit()
table = sm.stats.anova_lm(lm)
print(table)
#tukey
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print (pairwise_tukeyhsd(data['gaverage'], data['類別']))
