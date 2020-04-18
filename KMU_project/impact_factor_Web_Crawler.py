from selenium import webdriver
import time

# WOS進階檢索 #
w_1 = "http://apps.webofknowledge.com/summary.do?product=WOS&parentProduct=WOS&search_mode=AdvancedSearch&qid=5&SID=E5ZkWbvEFXZhcVeuTIG&&page=1&action=changePageSize&pageSize=50"
driver = webdriver.Chrome()
driver.get(w_1)
a2=[]
driver1=webdriver.Chrome()
for j in range(2):
    a = driver.find_elements_by_class_name('smallV110')
    n = 1
    for i in range(50):
        web1 = a[n].get_attribute("href")
        driver1.get(web1)
        time.sleep(0.5)
        try:
            driver1.find_element_by_css_selector('.snowplow-JCRoverlay').click()
            if driver1.find_element_by_css_selector('#hidden_section .FR_field:nth-child(3) value').text == '':
                driver1.find_element_by_id('hidden_section_label').click()
            print(driver1.find_element_by_class_name('Impact_Factor_table').text , driver1.find_element_by_css_selector('#hidden_section .FR_field:nth-child(3) value').text)
            a2.append([driver1.find_element_by_class_name('Impact_Factor_table').text , driver1.find_element_by_css_selector('#hidden_section .FR_field:nth-child(3) value').text])
        except:
            time.sleep(0.5)
        n+=3
    driver.find_element_by_css_selector('.snowplow-navigation-nextpage-top i').click()
    
    
from pandas.core.frame import DataFrame

data=DataFrame(a2)
data = data.rename(columns={0:'IF值',1:'UT'})
d1 = data['IF值'].str.split("\n").str.get(0)
data['IF值'] = d1.str.split(" ").str.get(0)

data.to_excel('68.xlsx',sheet_name='sheet1')

