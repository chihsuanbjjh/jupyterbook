#### jupyter book

!pip install -U jupyter-book

!pip install myst_nb

#### 測試class self

class cal:
    def __init__(self,):
        self.x = 1
        self.y = 2
    def printnum(self):
        print(self.x)
        print(self.y)
        
ii = cal()
ii.printnum()



print("請輸入三邊長a,b,c")
a, b, c = float(input().split())
# a = float(input("a:"))
# b = float(input("b:"))
# c = float(input("c:"))
# print((a+b > c) & (a+c > b) & (b+c > a))


def helen(a, b, c):
    s = (a+b+c)/2
    area = (s*(s-a)*(s-b)*(s-c))**(1/2)
    print("面積是", area)


if ((a+b > c) & (a+c > b) & (b+c > a)):
    helen(a, b, c)
else:
    print("這不是三角形")

a,b,c = float(input().split())
print(a)

正 n 邊形的面積為

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/5cec645ad16f7bd961a28efc6dc7edee9b9a5402)


n：n邊形  t：邊長

from math import*
n=4
edge = 10

area = (n*edge*edge*sin(2*pi/n))/(4*(1-cos(2*pi/n)))
print(area)

from time import sleep
import sys
line_1 = "You have woken up in a mysterious maze"
line_2 = "The building has 5 levels"
line_3 = "Scans show that the floors increase in size as you go down"
for x in line_1:
    print(x, end='')
    sys.stdout.flush()
    sleep(0.1)

li = ['1-國文(初)','2-英文', '3-生物','4-法文','5-公民']
correct_numbers = 0
for i in range(5-correct_numbers):
    print(li[i])
    
li.remove(li[1])
correct_numbers = 1
for i in range(5-correct_numbers):
    print(li[i])

str = input("please input the number:")

if str.isdigit():
    print("true")

import numpy as np
total = 10  ##總戀人個數
truelove = total-1 
success = np.zeros(total)  
run = 1000 

for k in range(1,run+1):
  lovers = np.random.permutation(total)   
  for reject in range(0,total):
    if reject == 0 and lovers[reject] == truelove: 
      success[reject] = success[reject]+1       
    if reject>0:                                        
      reference = max(lovers[0:reject])              
#### 請從此開始完成程式

import numpy as np
total = 10  ##總戀人個數
truelove = total-1 
success = np.zeros(total)  
run = 100

# print('total = ',total)
# print('truelove = ',truelove)
# print('success = ',success)
# print('run = ',run)

# lovers = np.random.permutation(total)  
# print('lovers = ',lovers)


print('測試',run,'次')
for k in range(1,run+1):
    lovers = np.random.permutation(total)
    # print('lovers = ',lovers)
    for reject in range(0,total):
        if reject == 0 and lovers[reject] == truelove: #如果一開始就是9
            success[reject] = success[reject]+1 #在這個位置找到真愛的話就加1       
        if reject>0:                                        
            reference = max(lovers[0:reject]) 
            # print('-----')
            # print('reject=',reject)
            # print('lovers[0:',reject,']= ',lovers[0:reject])
            # print('reference=',reference)
            for j in range(reject,total-1):
                if lovers[j] > reference:
                    break
            if lovers[j] == truelove:
                    # print('j=',j)
                success[j] = success[j]+1            
print(success)
print(success/run*100)

import numpy as np


def bestchoice():
    total = 10  ##總共的選擇個數
    best = total-1 
    success = np.zeros(total)  
    run = 1000
    for reject in range(1,total):
        for j in range(0,run):
            choice = np.random.permutation(total) #把被選者打散
            # print('choice=',choice)
            reference = max(choice[0:reject]) 
            # print('reference=',reference)

            for i in range(reject,total): #從切的地方往後比大小
                if choice[i] > reference:
                    # print('next is ',choice[i],'at i=',i)
                    if choice[i] == best:
                        success[reject-1] = success[reject-1] + 1
                        # print(choice[i],'is the best\n')
                    # else:
                        # print(choice[i],'is not the best\n')
                    break

    print(success)
    aa = success.tolist()
    print(aa.index(max(aa))+1)
    

for times in range(10):
    bestchoice()

bb = aa.tolist()

bb.index(max(bb))

city = '台北市'
count = 10000
print('台北市有10000人')
print('{}有{}人'.format(city,count))

name = '小明'
bmi = 21.34745634
print('小明bmi是21.23')
print('{}bmi是{:.4}'.format(name,bmi))

#### List常用處理方法<br>
len<br>
index<br>
append<br>
pop<br>


subject = ['國文','英文','數學']
subject

for i in subject:
    print(i)

print(len(subject))
print(type(len(subject)))

subject.index('國文')

subject.append('自然')
subject


subject = subject + ['社會','體育']
subject

subject.count('國文')


subject
print(subject.count('國文'))

print(subject)
subject.index('國文')

subject.insert(0,'生涯')
subject

subject.remove('國文')
subject
subject.remove('國文')
subject

subject.reverse()
subject

subject[::]

subject[1:]

subject[0::2]

subject[-1]

list(enumerate(subject))

for i,j in enumerate(subject):
    print(i,'|',j)

weight = [1,4,4,3,3,2] #加權
for i in range(len(weight)):
    print('{}科的加權是 {}'.format(subject[i],weight[i]))


%time
zip(subject,weight)
list(zip(subject,weight))

for i, j in zip(subject, weight):
    print(i, '科的加權是 ', j, sep='')

from random import sample
sample(range(1,1000),1)
sample(range(1,1000),1)
sample(range(1,1000),1)
sample(range(1,1000),1)
sample(range(1,1000),1)

**使用 loop 構建 list 壓縮為簡潔單行的方法**

#產生一個平方數的陣列
squared_list = []
for i in range(10):
    squared_list.append(i**2)
print(squared_list)

# list comprehension 運算式 數值
squared_list = [i**2 for i in range(10)]
print(squared_list)

# list comprehension with if
even_numbers = [i for i in range(10) if i % 2 == 0]
print(even_numbers)

# list comprehension with if-else
is_even_numbers = [True if i % 2 == 0 else False for i in range(10)]
print(is_even_numbers)

#計算需產生幾次亂數才可以是56的倍數

from random import sample

# sampling_times = 0
sample_results = []
while True:
    sampled_num = sample(range(1,1000),1)[0]
#     sampling_times += 1
    sample_results.append(sampled_num)
    if sampled_num % 56 == 0:
        break
print(sample_results)
print()
print(len(sample_results))

#計算需產生幾次亂數才可以是56的倍數 副程式化
def random_56():
    from random import sample

    # sampling_times = 0
    sample_results = []
    while True:
        sampled_num = sample(range(1,1000),1)[0]
    #     sampling_times += 1
        sample_results.append(sampled_num)
        if sampled_num % 56 == 0:
            break
#     print(sample_results)
#     print()
#     print(len(sample_results))
    return len(sample_results)

final = []
for i in range(10):
    final.append(random_56())
final




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%pylab inline



X = list(range(1,11))
X

plt.bar(x=X,height=final,)

#使用者輸入一個英文字串，試計算出母音aeiou的個數
word = input("請輸入一英文")
print('你輸入的是',word)

word[0]
word[1]
len(word)

#確認word可以以list的方式處理後，利用迴圈的方式來遍歷並判斷

vowel = 0 #記錄母音
for i in range(len(word)):
    if word[i] == 'a' or word[i] == 'e' or word[i] == 'i' or word[i] == 'o' or word[i] == 'u':
        vowel +=1
        
print('你輸入的是',word)
print('母音的個數有',vowel,'個',sep='')


#使用者輸入一個英文字串，試計算出字串裡'bob'出現的個數

#試試字串如何一次出現三個並依序
word[0]
word[0:3]

num_bob = 0 #記錄出現bob的個數
for i in range(len(word)):
    if word[i:i+3] == 'bob':
        num_bob +=1
        
print('你輸入的是',word)
print('bob出現的個數有',num_bob,'個',sep='')

#### 副程式

def squared(*args):
    for x in args:
        print(x**2)
squared(2,3,4,5)

squared_2 = lambda x: x**2 # (輸入值):(處理方式)

squared_2(2)

#### 類別 物件導向

class City:
    '''
    information about a certain city.
    '''
    def __init__(self, name, country, location, current_weather):
        self._name = name
        self._country = country
        self._location = location
        self._current_weather = current_weather
    
    def get_name(self):
        return self._name
    
    def get_country(self):
        return self._country
    
    def get_location(self):
        return self._location
    
    def get_current_weather(self):
        return '我在{}天氣{}'.format(self._name, self._current_weather)
    


class City:
    '''
    information about a certain city.
    '''
    def __init__(self, name):
        self._name = name
  

tpe = City("Taipei")

tpe

tpe.__class__

tpe.__doc__

tpe._name

class Movie:
    def __init__(self, rating, movie_time):
        self._rating = rating
        self._movie_time = movie_time
        self._genre = []

    def get_rating(self):
        return self._rating

    def get_movie_time(self):
        return self._movie_time

    def get_genre(self):
        return self._genre

    def add_genre(self, genre):
        self._genre.append(genre)
        return True

avengers_endgame = Movie(8.8, '3h 1min') # 初始化
# 靜態的屬性
print(avengers_endgame._rating)
print(avengers_endgame._movie_time)
print(avengers_endgame._genre)

# 動態的方法
print(avengers_endgame.get_rating())
print(avengers_endgame.get_movie_time())
print(avengers_endgame.get_genre())
avengers_endgame.add_genre("Action")
avengers_endgame.add_genre("Adventure")
avengers_endgame.add_genre("Sci-Fi")
print(avengers_endgame.get_genre())

use_the_force = "Luke, use the Force!"

print(use_the_force.title())
print(use_the_force.upper())
print(use_the_force.lower())

use_the_force = """
     
Luke, use the Force!
     
"""

use_the_force

print(use_the_force.rstrip())
print(use_the_force.lstrip())
print(use_the_force.strip())

#### 自訂名為bmi的模組

#有一個名叫bmi.py的檔案在同一目錄下，bmi.py裡有一個副程式叫get_bmi

from bmi import get_bmi
get_bmi(64,174)

#### 爬json網頁<br>
* requests.get()
* r.json()
* 調整

'''
AQI
https://opendata.epa.gov.tw/ws/Data/AQI/?$format=json
'''

import requests
aqi_url = 'https://opendata.epa.gov.tw/ws/Data/AQI/?$format=json'

# r = requests.get(aqi_url)
# r
    #SSLError

r = requests.get(aqi_url,verify = False)
print(type(r))
print(r.status_code)

aqi_data = r.json()
print(type(aqi_data))
print(len(aqi_data))
print(aqi_data[0])
print(len(aqi_data[0]))

for k,v in aqi_data[0].items():
    print(k,'|',v)



'''
for site in aqi_data:
    if site['County'] == '新北市' or site['County'] == '台北市'  :
        tpe_ntp_sites.append(site['SiteName'])
        
print(tpe_ntp_sites)
print(len(tpe_ntp_sites))
'''    

# List comprehension
tpe_ntp_sites = [site['SiteName'] for site in aqi_data if site['County'] == '新北市' or site['County'] == '台北市'  ]


print(tpe_ntp_sites)
print(len(tpe_ntp_sites))

#### 過去一小時，pm2.5avg最高與最低的測站分別是？

#找pm2.5avg的欄位與值
aqi_data[0] # 'PM2.5_AVG': '4',
aqi_data[0]['PM2.5_AVG']

avg_list=[]
site_name=[]
for avg in aqi_data:
#     print(avg['PM2.5_AVG'])
    if avg['PM2.5_AVG'] != "":
        avg_list.append(avg['PM2.5_AVG'])
        site_name.append(avg['SiteName'])
# avg_list=[avg['PM2.5_AVG'] for avg in aqi_data]

# print(avg_list)

avg_list2=avg_list
for i in range(len(avg_list)):
    avg_list2[i] = float(avg_list[i])
    
for k,v in zip(site_name,avg_list2):
    print(k,'->',v)
    
# print(avg_list2)    

print('目前最高數值觀測位置在：',site_name[avg_list2.index(max(avg_list2))],'數值是',max(avg_list2))
print('目前最高數值觀測位置在：',site_name[avg_list2.index(min(avg_list2))],'數值是',min(avg_list2))


# aqi_data[0]['PM2.5_AVG']

#### 利用pandas處理

#利用pandas就可以連動資料

import pandas as pd

ser = pd.Series(avg_list,index = site_name)
ser



ser.idxmax()
ser.max()

ser.idxmin()
ser.min()

#### 爬網頁

import requests
from bs4 import BeautifulSoup

url = 'https://www.imdb.com/title/tt4154796'
r = requests.get(url,)

# r.text

soup = BeautifulSoup(r.text,'lxml')

soup

#### 利用select gadget 做html資料的定位<br>
* 獲得資料requests<br>
* 存成str<br>
* soup<br>
* 定位資料 select gadget<br>
* soup.select<br>
* .text<br>
* .get(attr)<br>

float(soup.select('strong span')[0].text)

soup.select('h1')[0].text

soup.select('#titleYear a')[0].text

soup.select('.subtext a')[3].text



soup.select('.subtext a')[3].get('href')

import requests
from bs4 import BeautifulSoup

url ='https://www.imdb.com/title/tt4154796/releaseinfo?ref_=tt_ov_inf'
r = requests.get(url)

soup = BeautifulSoup(r.text,'lxml')

soup

country_name=[]
release_date=[]
for ele in soup.select('.release-date-item__country-name a'):
    country_name.append(ele.text.strip())
for ele in soup.select('.release-date-item__date'):
    release_date.append(ele.text)
for k,v in zip(country_name,release_date):
    print(k,v)

import pandas as pd
release_info = pd.DataFrame()
release_info['countries'] = country_name
release_info['release date'] = release_date
release_info.head()

release_info.groupby('release date').count()

#類似篩選器


#### 隨堂練習：請計算註冊於開曼群島的上市公司股價中位數

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://tw.stock.yahoo.com/d/i/rank.php?t=pri&e=tse&n=100'

r = requests.get(url)
print(r.status_code)

soup = BeautifulSoup(r.text,'lxml')
soup

#找出上市股價數值 .name+ td

stock_price=[]
stock_name=[]

#用select gadget找不到
# soup.find_all("table")[2].find_all("td")[0].find_all('td')[4].text
# soup.find_all("table")[2].find_all("td")[0].find_all('td')[5].text
# soup.find_all("table")[2].find_all("td")[0].find_all('td')[15].text #間隔10

for i in range(5,1000,10):
    stock_price.append(float(soup.find_all("table")[2].find_all("td")[0].find_all('td')[i].text))
print(stock_price)    
# print(len(stock_price))

for ele in soup.select('.name'):
    stock_name.append(ele.text)
    
# print(stock_name)    
# print(len(stock_name))



import numpy as np
 
#中位数
median = np.median(stock_price)
median

import pandas as pd
df = pd.read_csv('pew.csv')
df.head(10)

df = df.set_index('religion')
df = df.stack()
df.index = df.index.rename('income', level=1)
df.name = 'frequency'
df = df.reset_index()
df.head(10)

df

df = pd.read_csv('pew.csv')

#frame: 需要處理的數據框；
# id_vars: 保持原樣的數據列；
# value_vars: 需要被轉換成變量值的數據列；
# var_name: 轉換後變量的列名；
# value_name: 數值變量的列名。

df = pd.melt(df, id_vars=['religion'], value_vars=list(df.columns)[1:],
             var_name='income', value_name='frequency')
df = df.sort_values(by='religion')
df.to_csv('pew-tidy.csv', index=False)
df.head(10)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%pylab inline

# import
# 107年桃園市市長選舉候選人在各投開票所得票數一覽表

file_path = "https://s3-ap-northeast-1.amazonaws.com/tw-election-2018/city-mayor/300.xls"
xls_df = pd.read_excel(file_path)
xls_df

xls_df = pd.read_excel(file_path,skiprows=[0,1,3,4])
xls_df.head()

# 整理資料：把對應的欄位名稱重整
# 前三欄：行政區(district)、里、鄰(village)、開票所(office)
# 只有五位候選人

# xls_df.columns[0:2]=['District','Vilige','Office']
xls_df.columns
candidate_numbers_names = xls_df.columns

result = list(range(10))
result

pair = [(num1,num2) for num1 in range(2) for num2 in range(6,8)]
pair

# generator

result_list = [num for num in range(10)]
result_list
result_tuple = (num for num in range(10))
result_tuple

print(next(result_tuple))
print(next(result_tuple))
print(next(result_tuple))
print(next(result_tuple))
print(next(result_tuple))

mega_list = [num for num in range(10**100000)]
mega_list # 會跑很久

mega_tuple = (num for num in range(10**100000))
mega_tuple

import os 
os.getcwd()

with open('world_ind_pop_data.csv') as file:

    # Skip the column names
    file.readline()

# Open a connection to the file
with open('world_ind_pop_data.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(0, 1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)

import pandas as pd

# Initialize reader object: df_reader
df_reader = pd.read_csv('world_ind_pop_data.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))

df_reader2 = pd.read_csv('world_ind_pop_data.csv', )
df_reader
df_reader2



urb_pop_reader = pd.read_csv('world_ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], 
            df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)

import matplotlib.pyplot as plt
urb_pop_reader = pd.read_csv('world_ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'], 
            df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()

def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()
    
    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                    df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1]) for tup in pops_list]
    
        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()

# Set the filename: fn
fn = 'world_ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn, 'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn, 'ARB')

import numpy as np
import cv2
# 引入Python的可视化工具包 matplotlib
from matplotlib import pyplot as plt


def print_img_info(img):
    print("================打印一下图像的属性================")
    print("图像对象的类型 {}".format(type(img)))
    print(img.shape)
    print("图像宽度: {} pixels".format(img.shape[1]))
    print("图像高度: {} pixels".format(img.shape[0]))
    # GRAYScale 没有第三个维度哦， 所以这样会报错
    # print("通道: {}".format(img.shape[2]))
    print("图像分辨率: {}".format(img.size))
    print("数据类型: {}".format(img.dtype))
print_img_info(img)

# 导入一张图像 模式为彩色图片
img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

# 将色彩空间转变为灰度图并展示
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 打印图片信息
print_img_info(gray)

# 打印图片的局部
# print("打印图片局部")
print(gray[100:105, 100:105])

# plt.imshow(gray)

@interact
def get_cmap(x=['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']):
    plt.imshow(gray, cmap=x)


# 需要添加colormap 颜色映射函数为gray
# plt.imshow(gray, cmap=get_cmap(x))


# 隐藏坐标系
plt.axis('off')
# 展示图片

plt.show();



@interact
def fff(x=(10, 100000, 10),y=(10, 100000, 10),z=(10, 100000, 10)):
    print(x+20)

a = widgets.IntSlider()
b = widgets.IntSlider()
c = widgets.IntSlider()
ui = widgets.HBox([a, b, c])
def f(a, b, c):
    print((a, b, c))

out = widgets.interactive_output(f, {'a': a, 'b': b, 'c': c})

display(ui, out)

#### matplotlib 畫圖 PIL做成動畫

fig_list=[]
for i in range(2,100,10):
    x = np.linspace(0,4*np.pi,i)
    y = np.sin(x)
    fig = plt.figure()
    plt.plot(x,y);

    fig.savefig(f'{i}.png')



fig_list[0]

from PIL import Image
img_list = []
for i in range(2,100,10):
    img = Image.open(f'{i}.png')
    img_list.append(img)
    
img_list


img_list[0].save("out.gif", save_all=True, append_images=img_list[1:], duration=1000, loop=0)

import matplotlib.pyplot as plt
import numpy as np
%matplotlib notebook
%matplotlib notebook

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-') 

for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()

#### 解決matplotlib無法顯示中文


import matplotlib as mpl
import matplotlib.pyplot as plt

arr = np.random.normal(size=10000)
myfont = mpl.font_manager.FontProperties(fname='simhei.ttf')
fig, ax = plt.subplots()
ax.hist(arr, bins=30)
ax.set_title("標準常態分配", fontproperties=myfont)
ax.set_xlabel("x")
ax.set_ylabel("頻率", fontproperties=myfont)
plt.show()


import sympy as sp

triangle, circle, square = sp.symbols("triangle circle square")

eq1 = sp.Eq(2 * triangle + 1 * circle, 2 * circle + 2 * square)
eq2 = sp.Eq(1 * triangle + 1 * circle, 3 * square)
eq3 = sp.Eq(1 * triangle + 2 * circle, 2 * triangle + 1 * square)

choices = [2 * square + 1 * triangle, 2 * square + 2 * triangle,
           2 * square + 3 * triangle, 3 * square + 1 * triangle,
           3 * square + 2 * triangle]

# testing which choice let the linear system be solvable
for choice in choices:
    eq4 = sp.Eq(4 * circle, choice)
    print(f"{choice}: {[(k, v) for k, v in sp.solve([eq1, eq2, eq3, eq4]).items() if v != 0]}")

import pandas as pd