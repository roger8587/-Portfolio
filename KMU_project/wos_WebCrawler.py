from selenium import webdriver
import time
#import os
import pandas as pd

df = pd.read_excel(r'C:\Users\roger\OneDrive\桌面\高醫大\語法0924.xlsx',sheet_name = '各單位搜尋語法')
w_1 = "http://apps.webofknowledge.com/WOS_AdvancedSearch_input.do?product=WOS&search_mode=AdvancedSearch&replaceSetId=&goToPageLoc=SearchHistoryTableBanner&SID=D6Llz7ySVounXz6ScDg&errorQid=1#SearchHistoryTableBanner"
driver = webdriver.Chrome()

a1 = []
for i in range(80,df.shape[0]):
    #path = r'C:\Users\roger\OneDrive\桌面\高醫大' + '\\' + df['一級單位'][i]
    #if not os.path.isdir(path):
     #   os.mkdir(path)
    #path1 = path + '\\' + df['系所單位'][i]
    #os.mkdir(path1)
    print(df['系所單位'][i],':')
    driver.get(w_1)
    if df['搜尋語法'][i] == '':
        continue
    driver.find_element_by_id('value(input1)').clear()
    driver.find_element_by_id('value(input1)').send_keys(df['搜尋語法'][i])
    button = driver.find_element_by_id('search-button')
    button.click()
    time.sleep(1)
    driver.find_element_by_id('hitCount').click()
    time.sleep(1)
    num = int(driver.find_element_by_css_selector('#hitCount\.top').text.replace(",", ""))
    a1.append(num)
    for j in range(0,(int)(num/500)+1):
        driver.find_elements_by_class_name('selectedExportOption')[0].click()
        try:
            driver.find_element_by_name('匯出至其他檔案格式').click()
        except:
            time.sleep(0.1)
        head = 1+500*j      # 第 ? 筆資料(from)
        tail = 500*(j+1)    # 第 ? 筆資料(to)
        if j == (int)(num/500):
            tail = (int)(num)
        print(head,'~',tail)
        driver.find_element_by_id('markFrom').clear()
        driver.find_element_by_id('markFrom').send_keys(head)
        
        driver.find_element_by_id('markTo').clear()
        driver.find_element_by_id('markTo').send_keys(tail)
        #list1 = 完整記錄
        list1 = driver.find_element_by_id('select2-bib_fields-container')
        list1.click()
        while True:
            try:
                list1_4 = driver.find_elements_by_class_name('select2-results__option')
                break
            except:
                time.sleep(0.5)        
        list1_4[2].click()

        #list2 = 檔案格式WIN
        list2 = driver.find_element_by_id('select2-saveOptions-container')
        list2.click()
        while True:
            try:
                list2_5 = driver.find_elements_by_class_name('select2-results__option')
                break
            except:
                time.sleep(0.5)        
        list2_5[4].click()
        driver.find_element_by_id('numberOfRecordsRange').click()
        driver.find_element_by_id('exportButton').click()
        time.sleep(1)
        while True:
            try:
                d = driver.find_element_by_class_name('quickoutput-cancel-action')
                break
            except:
                time.sleep(0.5)
        d.click()
        time.sleep(1)
#%%

import pandas as pd
import os 
# =============================================================================
# 論文整理，txt轉excel
# =============================================================================
df = pd.read_excel('3101.xlsx') #爬蟲時順便匯出的論文數
path = r'C:\Users\roger\OneDrive\桌面\高醫大\論文更新\savedrecs ('
num1 = 0
for i in range(df.shape[0]):    
    num = int(df['論文數'][i]/500)+1
    data = pd.read_csv(path + str(num1) + ').txt',encoding='UTF-16 LE',delimiter="\t")
    data['一級單位'] = df['一級單位'][i]
    data['系所單位'] = df['系所單位'][i]
    for j in range(num1+1,num1+num):
        data1 = pd.read_csv(path + str(j) + ').txt',encoding='UTF-16 LE',delimiter="\t")
        data1['一級單位'] = df['一級單位'][i]
        data1['系所單位'] = df['系所單位'][i]
        data = pd.concat([data,data1],axis = 0)
    print(num1,num)
    data.to_excel(df['系所單位'][i] + '.xlsx')
    num1 += num
    
#%%
# =============================================================================
# 論文合併
# =============================================================================
a1 = os.listdir(r'C:\Users\roger\OneDrive\桌面\高醫大\論文')
path1 = r'C:\Users\roger\OneDrive\桌面\高醫大\論文'
data = pd.read_excel(path1 + '\\' + a1[0])
for i in range(1,len(a1)):
    data2 = pd.read_excel(path1 + '\\' + a1[i])
    data = pd.concat([data,data2],axis = 0)

#%%
# =============================================================================
#     論文數
# =============================================================================

df = pd.read_excel('醫學系分科.xlsx')
df1 = pd.read_excel('醫學系分科.xlsx', sheet_name = '醫學系全')
a1 = df.pivot_table(index=['科別'],columns=['PY'], values=['AU'], aggfunc='count')
a2 = df1.pivot_table(columns=['PY'], values=['AU'], aggfunc='count')

writer = pd.ExcelWriter('醫學系分科人數統計.xlsx')
a1.to_excel(writer,sheet_name='1')
a2.to_excel(writer,sheet_name='2')
writer.save()

    