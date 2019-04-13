#!/usr/bin/env python
# coding: utf-8

# In[0]:


from selenium import webdriver
import numpy as np
import cv2
import csv
code = {'b6589fc6ab0dc82cf12099d1c2d40ab994e8410c': '0',
        '356a192b7913b04c54574d18c28d46e6395428ab': '1',
        'da4b9237bacccdf19c0760cab7aec4a8359010b0': '2',
        '77de68daecd823babbb58edb1c8e14d7106e83bb': '3',
        '1b6453892473a467d07372d45eb05abc2031647a': '4',
        'ac3478d69a3c81fa62e60f5c3696165a4e5e6ac4': '5',
        'c1dfd96eea8cc2b62785275bca38ac261256e278': '6',
        '902ba3cda1883801594b6e1b452790cc53948fda': '7',
        'fe5dbbcea5ce7e2988b8c69bcfdfde8904aabc1f': '8',
        '0ade7c2cf97f75d009975f4d720d1fa6c19f4897': '9'}
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('headless')
chrome_options.add_argument('window-size=1920x1080')
driver = webdriver.Chrome(
    executable_path='./chromedriver.exe', chrome_options=chrome_options)


# In[1]:


with open('./img/train/labeled.csv', 'w', encoding='utf-8', newline='') as csvfile:
    name = 0
    writer = csv.writer(csvfile)
    for z in range(3):
        driver.get('https://www.ais.tku.edu.tw/EleCos/login.aspx')
        driver.get('https://www.ais.tku.edu.tw/EleCos/Handler1.ashx')
        Handler = driver.find_element_by_tag_name('pre')
        res_code = ""
        for i in range(6):
            res_code = res_code + code[Handler.text[2+i*43:42+i*43]]
        for j in range(2):
            driver.get(
                'https://www.ais.tku.edu.tw/EleCos/BaseData/confirm.ashx')
            element = driver.find_element_by_tag_name('img')
            driver.save_screenshot('temp.png')
            img = cv2.imread("temp.png")
            left = element.location['x']
            right = element.location['x'] + element.size['width']
            top = element.location['y']
            bottom = element.location['y'] + element.size['height']
            crop_img = img[top:int(bottom), left:int(right)]
            name += 1
            lower = np.array([26, 43, 46])
            upper = np.array([34, 255, 255])
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            res = cv2.bitwise_and(crop_img, crop_img, mask=mask)
            cv2.imwrite("./img/train/"+str(name)+".png", res)
            array = []
            array.append(str(name))
            array.append(res_code)
            writer.writerow(array)


# In[2]:


driver.quit()
