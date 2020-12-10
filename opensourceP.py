import pandas as pd
import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import re

plt.rcParams['font.family'] = 'NanumGothic'

url = 'https://finance.naver.com/sise/lastsearch2.nhn'
req = requests.get(url)
html = req.text
soup = BeautifulSoup(html, 'html.parser')
tables = soup.select('table')
table_name = soup.select('table > caption')  
table_html = str(tables)
table_df_list = pd.read_html(table_html)

table_df = table_df_list[1]
table_df_clean = table_df.dropna() ## 비어있는 행 지우기 ##



## %없애기 ##
temp=[]
for i in table_df_clean["검색비율"]:
    j = i.replace('%', '')
    j = float(j)
    temp.append(j)
table_df_clean["검색비율"] = temp

temp=[]
for i in table_df_clean["등락률"]:
    j = i.replace('%', '')
    j = float(j)
    temp.append(j)
table_df_clean["등락률"] = temp


##거래량 평균구해서 더 많으면 1, 적으면 2로 키값을 만들어 그룹을 묶은다음 거래량 별 시가의 평균을 냄 ##
table_df_clean['key'] = 2
print(table_df_clean['거래량'].mean())
mean = table_df_clean['거래량'].mean()
table_df_clean.loc[table_df_clean['거래량'] >= mean,'key'] = 1

print(table_df_clean.groupby('key')['시가'].mean())

print(table_df_clean)

##머신러닝 부분##

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


pca = PCA(n_components=2)
pca.fit(table_df_clean.iloc[:10,[2,3,4,5,10,11,12]])


print(pca.explained_variance_ratio_)
print(pca.singular_values_)

pca_df = pca.transform(table_df_clean.iloc[:,[2,3,4,5,10,11,12]])

print(pca_df.shape)

plt.scatter(pca_df[:,0], pca_df[:,1], c=np.log(table_df_clean['현재가'] +1),s=50)

## 그래프 ##

fig = plt.figure()
axes1 = fig.add_subplot(1,1,1)
axes1.hist(table_df_clean['시가'], bins=50)
axes1.set_title('시가 분포도')
axes1.set_xlabel('시가')
axes1.set_ylabel('기업 수')


scatter_plot = plt.figure()
axes2 = scatter_plot.add_subplot(1,1,1)
axes2.scatter(table_df_clean['거래량'],table_df_clean['시가'] )
axes2.set_title('거래량 - 시가')
axes2.set_xlabel('거래량')
axes2.set_ylabel('시가')


