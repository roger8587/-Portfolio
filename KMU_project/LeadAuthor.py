#%%
#切出第一作者欄位加以分析
import pandas as pd

df1 = pd.read_excel(r'C:\Users\roger\OneDrive\桌面\高醫大更新\高醫大論文0406.xlsx')
#df1 = df[df['系所單位'] == '高雄醫學大學及附屬醫療機構'].reset_index(drop=True)
df1['第一作者'] = df1['AF'].str.split(';',expand=True)[0]
#高醫大地址轉換
df1["RP"] = df1["RP"].str.replace("100 Shih Chuan 1st Rd, Kaohsiung 80708, | 100 Tzyou 1 Rd, Kaohsiung 807, | 100 Shih Chuan 1st Rd, Kaohsiung 80708, | 100 Shih Chuan 1st Rd, Kaohsiung 807, | 100 Shihcyuan 1st Rd, Kaohsiung 80708, | 100 Shih Chuan 1st Rd, Kaohsiung, | 100,Tzyou 1st Rd, Kaohsiung 80754, | 100,Tzyou 1st Rd, Kaohsiung 807,|100 Tzyou First Rd, Kaohsiung 80708,|100,Tzyou 1st Rd, Kaohsiung 80756,|100 Shiquan 1st Rd Kaohsiung, Kaohsiung 80708,|482 Shan Ming Rd, Kaohsiung 812,|100 Shihcyuan 1st Rd, Kaohsiung 80708,|100 Shichuan 1st Rd, Kaohsiung 80708,|100 Tzyou 1 st Rd, Kaohsiung 807,|100 Shih Chuan 1st Rd, Kaohsiung,|100 Shin Chuan 1st Rd, Kaohsiung 80708,| 100,Shih Chuan 1st Rd, Kaohsiung,|100 Tzyou 1st Rd, Kaohsiung 807,|100,Shih Chuan 1st Rd, Kaohsiung 80708,","Kaohsiung Med Univ,")
df1["C1"] = df1["C1"].str.replace("Kaohsiung Municipal HsiaoKang Hosp|Kaohsiung Med Univ & Hosp, Dept Surg|KMU|Kaohsiung Med Univ Hosp|Kaohsiung Municipal Tatung Hosp|Kaohsiung Municipal Siaogang Hosp|Kaohsiung Municipal Hsiao Kang Hosp|KaohsiungMed Univ|Kaohsiung Municipal Siao Gang Hosp|Kaohsiung Municipal Hsiaokang Hosp|Kaohsiung Municipal TA TUHG Hosp","Kaohsiung Med Univ")
df1["RP"] = df1["RP"].str.replace("Kaohsiung Municipal Cijin Hosp|Kaohsiung Med Univ Hosp|Kaohsiung Municipal Tatung Hosp|Kaohsiung Municipal Siaogang Hosp|Kaohsiung Municipal Hsiao Kang Hosp|KaohsiungMed Univ|Kaohsiung Municipal Siao Gang Hosp|Kaohsiung Municipal Hsiaokang Hosp|Kaohsiung Municipal TA TUHG Hosp","Kaohsiung Med Univ")

#合併第一作者欄位
d = df1['C1'].str.split('[',expand=True)
d1 = df1['RP'].str.split(';',expand=True)
a=[]
a2=[]
a3=[]
a4=[]
a5=[]
for j in range(df1.shape[0]):
    b = d.loc[j,:]
    b1 = d1.loc[j,:]
    b.dropna(axis=0, how='any', inplace=True)
    b1.dropna(axis=0, how='any', inplace=True)
    cols = [col for col in b if df1['第一作者'][j] in col]
    a1 = '['.join(cols)
    cols1 = [col for col in b if "Kaohsiung Med Univ" in col]
    cols2 = [col for col in b if "Taiwan" in col]
    cols3 = [col for col in b if "Taiwan" in col and df1['第一作者'][j] in col]
    cols4 = [col for col in b if "Kaohsiung Med Univ" in col and df1['第一作者'][j] in col]
    cols5 = [col for col in b1 if "Taiwan" in col]
    cols6 = [col for col in b1 if "Kaohsiung Med Univ" in col]
    if len(b)-1-len(cols2)>0:
        a2.append('國際合著')
    elif len(cols2)-len(cols1):
        a2.append('國內合著')
    else :
        a2.append('單一機構')
    if  'Kaohsiung Med Univ' in str(df1["RP"][j]) or 'Kaohsiung Med Univ' in a1:
        a3.append(1)
    else:
        a3.append(0)
    if  len(cols5)-len(cols6)>0 or len(cols3) - len(cols4)>0:
        a4.append(1)
    else:
        a4.append(0)
    if  str(df1["RP"][j]).count('(') - len(cols5)>0 or len(cols) - len(cols3)>0:
        a5.append(1)
    else:
        a5.append(0)
    a.append(a1)
df1['C2'] = a
df1['合著類型'] = a2
df1['第一作者高醫大'] = a3
df1['第一作者國內'] = a4
df1['第一作者國外'] = a5
#%%
#論文數
df2 = df1[df1['第一作者國內']==1]
a1 = df2.groupby(['PY','合著類型'])['合著類型'].count()
df3 = df1[df1['第一作者高醫大']==1]
a2 = df3.groupby(['PY','合著類型'])['合著類型'].count()
df4 = df1[df1['第一作者國外']==1]
a3 = df4.groupby(['PY','合著類型'])['合著類型'].count()
a4 = df1.groupby(['PY','合著類型'])['合著類型'].count()
result = pd.concat([a2,a1,a3,a4],axis=1)
result.columns = ['高雄醫學大學','國內校外','國外','合計']
#%%
#被引用次數
df5 = df1[df1['第一作者國內']==1]
a5 = df2.groupby(['PY','合著類型'])['TC'].sum()
df6 = df1[df1['第一作者高醫大']==1]
a6 = df3.groupby(['PY','合著類型'])['TC'].sum()
df7 = df1[df1['第一作者國外']==1]
a7 = df4.groupby(['PY','合著類型'])['TC'].sum()
a8 = df1.groupby(['PY','合著類型'])['TC'].sum()
result1 = pd.concat([a6,a5,a7,a8],axis=1)
result1.columns = ['高雄醫學大學','國內校外','國外','合計']
#%%
#hindex
df1['rank'] = df1.groupby(['一級單位'])['TC'].rank(ascending=False, method='first')
df1['rank(依年分)'] = df1.groupby(['一級單位','PY'])['TC'].rank(ascending=False, method='first')

b1 = df1[df1['TC']>=df1['rank']].reset_index(drop=True)
b2 = df1[df1['TC']>=df1['rank(依年分)']].reset_index(drop=True)
k5 = b2.pivot_table(index=['一級單位'],columns=['PY'], values=['AU'], aggfunc='count')
k6 = b1.groupby(['一級單位'])['一級單位'].count()
#%%
writer = pd.ExcelWriter('第一作者.xlsx')
df1.to_excel(writer,sheet_name='高醫大論文(第一作者)')
result.to_excel(writer,sheet_name='論文數')
result1.to_excel(writer,sheet_name='被引用次數')
k5.to_excel(writer,sheet_name='hindex')
k6.to_excel(writer,sheet_name='hindex總計')
writer.save()


