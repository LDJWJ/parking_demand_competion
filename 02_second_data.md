### 학습 목표 
 * 데이터 오류에 대해 확인하고, 데이터 전처리를 수행한다.
 * 모델을 제출해 본다.

* https://dacon.io/competitions/official/235745/talkboard/403708?page=1&dtype=recent


```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
```


```python
import pandas as pd

train = pd.read_csv("../data/parking_demand/train.csv")
test = pd.read_csv("../data/parking_demand/test.csv")
sub = pd.read_csv("../data/parking_demand/sample_submission.csv")
age = pd.read_csv("../data/parking_demand/age_gender_info.csv")

train.shape, test.shape, sub.shape, age.shape
```




    ((2952, 15), (1022, 14), (150, 2), (16, 23))




```python
train.columns
```




    Index(['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
           '자격유형', '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
           '도보 10분거리 내 버스정류장 수', '단지내주차면수', '등록차량수'],
          dtype='object')




```python
train.columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
       '자격유형', '임대보증금', '임대료', '10분내지하철수',
       '10분내버스정류장수', '단지내주차면수', '등록차량수']

test.columns = ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
       '자격유형', '임대보증금', '임대료', '10분내지하철수',
       '10분내버스정류장수', '단지내주차면수']

```

### 데이터 오류로 인한 데이터 제외

* - 테스트셋에서 평가 제외되는 데이터는 'C2675'(2번 사항에 해당), 'C2335', 'C1327'(3번 사항에 해당) 3개 단지입니다.


```python
train.단지코드.unique()
```




    array(['C2483', 'C2515', 'C1407', 'C1945', 'C1470', 'C1898', 'C1244',
           'C1171', 'C2073', 'C2513', 'C1936', 'C2049', 'C2202', 'C1925',
           'C2576', 'C1312', 'C1874', 'C2650', 'C2416', 'C2013', 'C1424',
           'C2100', 'C2621', 'C2520', 'C2319', 'C1616', 'C1704', 'C2258',
           'C1032', 'C2038', 'C1859', 'C1722', 'C1850', 'C2190', 'C1476',
           'C1077', 'C1068', 'C1983', 'C2135', 'C2034', 'C1109', 'C1497',
           'C2289', 'C2597', 'C2310', 'C1672', 'C2132', 'C1439', 'C1613',
           'C2216', 'C1899', 'C1056', 'C2644', 'C1206', 'C2481', 'C1718',
           'C1655', 'C1430', 'C1775', 'C1519', 'C2221', 'C1790', 'C2109',
           'C1698', 'C1866', 'C1005', 'C1004', 'C1875', 'C2156', 'C2212',
           'C2401', 'C2571', 'C1175', 'C1833', 'C2445', 'C1885', 'C2368',
           'C2016', 'C2371', 'C2536', 'C2538', 'C1014', 'C1592', 'C1867',
           'C2326', 'C1015', 'C1620', 'C1049', 'C2000', 'C2097', 'C1668',
           'C1689', 'C1234', 'C2514', 'C1368', 'C1057', 'C2336', 'C1026',
           'C2256', 'C1900', 'C2666', 'C2361', 'C1642', 'C1013', 'C2232',
           'C1973', 'C2458', 'C2574', 'C2133', 'C2096', 'C2010', 'C1879',
           'C1131', 'C1468', 'C1213', 'C1173', 'C2492', 'C2032', 'C2094',
           'C1880', 'C2089', 'C1744', 'C2046', 'C2071', 'C2635', 'C2390',
           'C2561', 'C1663', 'C2490', 'C2066', 'C1585', 'C2276', 'C1155',
           'C1693', 'C1889', 'C2518', 'C1962', 'C1666', 'C1988', 'C1537',
           'C1329', 'C1762', 'C2008', 'C1319', 'C1141', 'C2340', 'C1929',
           'C1681', 'C1184', 'C2383', 'C1579', 'C2173', 'C1911', 'C1638',
           'C2412', 'C1871', 'C1309', 'C1527', 'C2208', 'C1940', 'C2596',
           'C2227', 'C2563', 'C2358', 'C1492', 'C1601', 'C1687', 'C1236',
           'C1487', 'C1379', 'C1386', 'C1656', 'C2526', 'C1022', 'C1896',
           'C1269', 'C1916', 'C2070', 'C1967', 'C2021', 'C1143', 'C2188',
           'C2651', 'C1036', 'C2657', 'C2527', 'C1502', 'C2262', 'C1084',
           'C2530', 'C1046', 'C1761', 'C1102', 'C2420', 'C1122', 'C2042',
           'C1375', 'C1410', 'C1641', 'C1706', 'C1307', 'C2601', 'C1085',
           'C2385', 'C1059', 'C2162', 'C1819', 'C2325', 'C2394', 'C1133',
           'C1281', 'C1194', 'C2308', 'C2036', 'C1394', 'C1180', 'C2503',
           'C1907', 'C2181', 'C1768', 'C1783', 'C2192', 'C2346', 'C2680',
           'C2631', 'C2141', 'C1569', 'C2099', 'C2287', 'C2055', 'C1428',
           'C2522', 'C2560', 'C2068', 'C2603', 'C1965', 'C1660', 'C2378',
           'C1268', 'C1994', 'C1837', 'C1000', 'C1465', 'C1448', 'C1516',
           'C2670', 'C1365', 'C1177', 'C1360', 'C2488', 'C1406', 'C1566',
           'C1227', 'C2460', 'C2486', 'C2106', 'C1572', 'C1773', 'C1677',
           'C1823', 'C1344', 'C2692', 'C2505', 'C2587', 'C2127', 'C1316',
           'C1674', 'C1713', 'C1845', 'C2082', 'C1328', 'C2357', 'C2565',
           'C1804', 'C1397', 'C2255', 'C1343', 'C1987', 'C2479', 'C2352',
           'C1310', 'C1738', 'C1039', 'C1863', 'C1426', 'C2659', 'C2489',
           'C2211', 'C2314', 'C1861', 'C2389', 'C1490', 'C1024', 'C1788',
           'C1740', 'C2620', 'C1286', 'C2085', 'C1089', 'C2237', 'C1341',
           'C1338', 'C2405', 'C1969', 'C2274', 'C1699', 'C2251', 'C1340',
           'C2373', 'C1455', 'C1095', 'C2137', 'C1985', 'C2583', 'C2663',
           'C2450', 'C2329', 'C1834', 'C1649', 'C1848', 'C1743', 'C1350',
           'C1402', 'C1103', 'C1129', 'C1027', 'C2377', 'C2431', 'C2661',
           'C1263', 'C1136', 'C2605', 'C2393', 'C1673', 'C1017', 'C2539',
           'C1933', 'C2316', 'C2051', 'C2414', 'C1301', 'C1700', 'C1636',
           'C2612', 'C1757', 'C2507', 'C1163', 'C2627', 'C2040', 'C2609',
           'C2001', 'C1065', 'C1363', 'C2579', 'C1048', 'C1210', 'C1320',
           'C1941', 'C1326', 'C1685', 'C2618', 'C1451', 'C2143', 'C1968',
           'C2470', 'C1258', 'C2453', 'C1659', 'C1724', 'C1802', 'C1939',
           'C1284', 'C2595', 'C2351', 'C2506', 'C1697', 'C2259', 'C1786',
           'C1357', 'C2570', 'C1652', 'C1565', 'C1910', 'C2359', 'C2139',
           'C1979', 'C1803', 'C2508', 'C2531', 'C1695', 'C2556', 'C2086',
           'C1544', 'C2154', 'C2496', 'C1756', 'C2362', 'C2568', 'C2245',
           'C2059', 'C2549', 'C1584', 'C2298', 'C2225', 'C1218', 'C2328',
           'C1045', 'C1207', 'C1970', 'C1732', 'C2433', 'C1894', 'C1156',
           'C2142', 'C2153', 'C2186', 'C1176', 'C2446', 'C2586', 'C2035',
           'C2020', 'C2437', 'C2532'], dtype=object)




```python
train.loc[ ((train['단지코드']=='C2675') | 
           (train['단지코드']=='C2335') |
           (train['단지코드']=='C1327') ) , :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>10분내지하철수</th>
      <th>10분내버스정류장수</th>
      <th>단지내주차면수</th>
      <th>등록차량수</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
test.loc[ ((test['단지코드']=='C2675') | 
           (test['단지코드']=='C2335') |
           (test['단지코드']=='C1327') ) , :].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>10분내지하철수</th>
      <th>10분내버스정류장수</th>
      <th>단지내주차면수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>579</th>
      <td>C2675</td>
      <td>512</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>36.65</td>
      <td>130</td>
      <td>9.0</td>
      <td>A</td>
      <td>18476000</td>
      <td>154790</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1016.0</td>
    </tr>
    <tr>
      <th>580</th>
      <td>C2675</td>
      <td>512</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>44</td>
      <td>9.0</td>
      <td>A</td>
      <td>34082000</td>
      <td>232200</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1016.0</td>
    </tr>
    <tr>
      <th>581</th>
      <td>C2675</td>
      <td>512</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>80</td>
      <td>9.0</td>
      <td>A</td>
      <td>34082000</td>
      <td>232200</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1016.0</td>
    </tr>
  </tbody>
</table>
</div>



### 테스트 데이터 셋에서 세개의 코드 데이터를 없애기


```python
test = test.loc[ ~((test['단지코드']=='C2675') | 
           (test['단지코드']=='C2335') |
           (test['단지코드']=='C1327') ) , :]
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>10분내지하철수</th>
      <th>10분내버스정류장수</th>
      <th>단지내주차면수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>39.79</td>
      <td>116</td>
      <td>14.0</td>
      <td>H</td>
      <td>22830000</td>
      <td>189840</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.81</td>
      <td>30</td>
      <td>14.0</td>
      <td>A</td>
      <td>36048000</td>
      <td>249930</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>112</td>
      <td>14.0</td>
      <td>H</td>
      <td>36048000</td>
      <td>249930</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>120</td>
      <td>14.0</td>
      <td>H</td>
      <td>36048000</td>
      <td>249930</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C1072</td>
      <td>754</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>51.46</td>
      <td>60</td>
      <td>14.0</td>
      <td>H</td>
      <td>43497000</td>
      <td>296780</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>683.0</td>
    </tr>
  </tbody>
</table>
</div>



### 확인


```python
test.loc[ ((test['단지코드']=='C2675') | 
           (test['단지코드']=='C2335') |
           (test['단지코드']=='C1327') ) , :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>10분내지하철수</th>
      <th>10분내버스정류장수</th>
      <th>단지내주차면수</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### 오류 데이터 처리

* ※ 동일한 단지에 코드가 2개로 부여된 단지 코드 (3쌍) : ['C2085', 'C1397'], ['C2431', 'C1649'], ['C1036', 'C2675'] 

* - (참고 사항) 주차면수는 하나의 단지임을 전제로 산정된 것이고 총세대수는 두 개 단지의 합계입니다. 다만 등록차량대수는 ['C2085', 'C1397'] 단지의 경우 동일 수치


```python
train.loc[ train['단지코드']=='C2085',  "총세대수" ] = 1339
train.loc[ train['단지코드']=='C1397',  "총세대수" ] = 1339
```

* 단지코드를 C2085,C1397 => N2085로 변경


```python
print( train.loc[ train['단지코드']=='C2085', : ].shape  )
print( train.loc[ train['단지코드']=='C1397', : ].shape  )
```

    (8, 15)
    (6, 15)
    

### 변경 후, 처리 후, 단지코드를 N을 붙여 N2085로 변경


```python
train.loc[ train['단지코드']=='C2085',  "단지코드" ] = 'N2085'
train.loc[ train['단지코드']=='C1397',  "단지코드" ] = 'N2085'
```


```python
train.loc[ train['단지코드']=='N2085', : ].shape
```




    (14, 15)



### 오류 코드 변경
 * C2431, C1649의 총세대수를 1047로 변경
 * C2431, C1649의 등록차량대수를 1214로 변경
 * C2431, C1649의 단지코드를 N2431로 변경
 


```python
a = train.loc[ train['단지코드']=='C2431', : ]
b = train.loc[ train['단지코드']=='C1649', : ]

print(  a.shape, b.shape )
print( a['총세대수'], b['총세대수'])
print( a['등록차량수'], b['등록차량수'])
```

    (2, 15) (4, 15)
    2372    472
    2373    472
    Name: 총세대수, dtype: int64 2315    575
    2316    575
    2317    575
    2318    575
    Name: 총세대수, dtype: int64
    2372    359.0
    2373    359.0
    Name: 등록차량수, dtype: float64 2315    855.0
    2316    855.0
    2317    855.0
    2318    855.0
    Name: 등록차량수, dtype: float64
    


```python
train.loc[ train['단지코드']=='C2431',  "총세대수" ] = 1047
train.loc[ train['단지코드']=='C1649',  "총세대수" ] = 1047

train.loc[ train['단지코드']=='C2431',  "등록차량수" ] = 1214
train.loc[ train['단지코드']=='C1649',  "등록차량수" ] = 1214

train.loc[ train['단지코드']=='C2431',  "단지코드" ] = 'N2431'
train.loc[ train['단지코드']=='C1649',  "단지코드" ] = 'N2431'
```


```python
train.loc[ train['단지코드']=='N2431', : ].shape
```




    (6, 15)



### 오류 코드 변경
 * C1036의 총세대수를 1243로 변경
 * C1036의 단지코드를 N1036로 변경


```python
a = train.loc[ train['단지코드']=='C2675', : ]
b = train.loc[ train['단지코드']=='C1036', : ]
a.shape, b.shape
```




    ((0, 15), (7, 15))




```python
train.loc[ train['단지코드']=='C1036',  "총세대수" ] = 1243
train.loc[ train['단지코드']=='C1036',  "단지코드" ] = 'N1036'
```


```python
train.loc[ train['단지코드']=='N1036', : ].shape
```




    (7, 15)



### 오류 3
```
3. 단지코드 등 기입 실수로 데이터 정제 과정에서 매칭 오류 발생  
 - (오류 내용) 단지코드 등 기입 실수로 총세대수가 주차면수에 비해 과하게 많거나 적은 경우가 발생하였고, 점검 결과 일부 데이터의 단지코드, 총세대수, 주차면수 등에서 오류가 검출되었습니다.
 - (발생 원인) 원천데이터 수집 과정에서 단지 코드 등이 잘못 기입되었고 이를 인지하지 못한 채 데이터 정제를 하여 오류가 발생하였습니다.
 - (관련 데이터) 아래와 같이 총 9개 단지에서 같은 문제가 확인되었습니다. 
※ 실수가 발생한 단지 코드 (9개 단지) : ['C2335', 'C1327', 'C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']
 - C2335, C1327 단지는 테스트셋, 나머지는 트레인셋 입니다.
```

### 오류 처리
 * train 데이터 셋에 오류 발생 코드를 ERR04로 변경 후, 데이터 셋을 두개로 분리


```python
train.loc[ train['단지코드']=='C1095',  "단지코드" ] = 'ERR04_1095'
train.loc[ train['단지코드']=='C2051',  "단지코드" ] = 'ERR04_2051'
train.loc[ train['단지코드']=='C1218',  "단지코드" ] = 'ERR04_1218'
train.loc[ train['단지코드']=='C1894',  "단지코드" ] = 'ERR04_1894'
train.loc[ train['단지코드']=='C2483',  "단지코드" ] = 'ERR04_2483'
train.loc[ train['단지코드']=='C1502',  "단지코드" ] = 'ERR04_1502'
train.loc[ train['단지코드']=='C1988',  "단지코드" ] = 'ERR04_1988'
```


```python
train.loc[ train['단지코드'].str.contains('ERR'), :].shape
```




    (56, 15)




```python
train.loc[ train['단지코드'].str.contains('ERR'), :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>단지코드</th>
      <th>총세대수</th>
      <th>임대건물구분</th>
      <th>지역</th>
      <th>공급유형</th>
      <th>전용면적</th>
      <th>전용면적별세대수</th>
      <th>공가수</th>
      <th>자격유형</th>
      <th>임대보증금</th>
      <th>임대료</th>
      <th>10분내지하철수</th>
      <th>10분내버스정류장수</th>
      <th>단지내주차면수</th>
      <th>등록차량수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>39.72</td>
      <td>134</td>
      <td>38.0</td>
      <td>A</td>
      <td>15667000</td>
      <td>103680</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>39.72</td>
      <td>15</td>
      <td>38.0</td>
      <td>A</td>
      <td>15667000</td>
      <td>103680</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.93</td>
      <td>385</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.93</td>
      <td>15</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.93</td>
      <td>41</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.95</td>
      <td>89</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>51.95</td>
      <td>135</td>
      <td>38.0</td>
      <td>A</td>
      <td>27304000</td>
      <td>184330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ERR04_2483</td>
      <td>900</td>
      <td>아파트</td>
      <td>경상북도</td>
      <td>국민임대</td>
      <td>59.88</td>
      <td>86</td>
      <td>38.0</td>
      <td>A</td>
      <td>30357000</td>
      <td>214270</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1425.0</td>
      <td>1015.0</td>
    </tr>
    <tr>
      <th>1228</th>
      <td>ERR04_1988</td>
      <td>475</td>
      <td>아파트</td>
      <td>전라남도</td>
      <td>국민임대</td>
      <td>36.63</td>
      <td>200</td>
      <td>12.0</td>
      <td>A</td>
      <td>12026000</td>
      <td>87940</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>722.0</td>
      <td>402.0</td>
    </tr>
    <tr>
      <th>1229</th>
      <td>ERR04_1988</td>
      <td>475</td>
      <td>아파트</td>
      <td>전라남도</td>
      <td>국민임대</td>
      <td>36.63</td>
      <td>43</td>
      <td>12.0</td>
      <td>A</td>
      <td>12026000</td>
      <td>87940</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>722.0</td>
      <td>402.0</td>
    </tr>
    <tr>
      <th>1230</th>
      <td>ERR04_1988</td>
      <td>475</td>
      <td>아파트</td>
      <td>전라남도</td>
      <td>국민임대</td>
      <td>46.22</td>
      <td>204</td>
      <td>12.0</td>
      <td>A</td>
      <td>15304000</td>
      <td>103850</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>722.0</td>
      <td>402.0</td>
    </tr>
    <tr>
      <th>1231</th>
      <td>ERR04_1988</td>
      <td>475</td>
      <td>아파트</td>
      <td>전라남도</td>
      <td>국민임대</td>
      <td>46.22</td>
      <td>28</td>
      <td>12.0</td>
      <td>A</td>
      <td>15304000</td>
      <td>103850</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>722.0</td>
      <td>402.0</td>
    </tr>
    <tr>
      <th>1521</th>
      <td>ERR04_1502</td>
      <td>407</td>
      <td>아파트</td>
      <td>울산광역시</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>241</td>
      <td>7.0</td>
      <td>A</td>
      <td>19895000</td>
      <td>160400</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>552.0</td>
      <td>438.0</td>
    </tr>
    <tr>
      <th>1522</th>
      <td>ERR04_1502</td>
      <td>407</td>
      <td>아파트</td>
      <td>울산광역시</td>
      <td>국민임대</td>
      <td>46.90</td>
      <td>166</td>
      <td>7.0</td>
      <td>A</td>
      <td>19895000</td>
      <td>160400</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>552.0</td>
      <td>438.0</td>
    </tr>
    <tr>
      <th>2270</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>29.95</td>
      <td>66</td>
      <td>37.0</td>
      <td>A</td>
      <td>11586000</td>
      <td>151930</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2271</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>36.90</td>
      <td>36</td>
      <td>37.0</td>
      <td>A</td>
      <td>13663000</td>
      <td>189090</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2272</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>36.98</td>
      <td>102</td>
      <td>37.0</td>
      <td>A</td>
      <td>13663000</td>
      <td>189090</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2273</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>36.98</td>
      <td>320</td>
      <td>37.0</td>
      <td>A</td>
      <td>13663000</td>
      <td>189090</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2274</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.94</td>
      <td>178</td>
      <td>37.0</td>
      <td>A</td>
      <td>25140000</td>
      <td>240470</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2275</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>46.96</td>
      <td>240</td>
      <td>37.0</td>
      <td>A</td>
      <td>25140000</td>
      <td>240470</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2276</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>51.70</td>
      <td>202</td>
      <td>37.0</td>
      <td>A</td>
      <td>30605000</td>
      <td>262330</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>ERR04_1095</td>
      <td>1256</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>국민임대</td>
      <td>59.94</td>
      <td>112</td>
      <td>37.0</td>
      <td>A</td>
      <td>38256000</td>
      <td>318070</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>528.0</td>
      <td>505.0</td>
    </tr>
    <tr>
      <th>2426</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>59.39</td>
      <td>2</td>
      <td>0.0</td>
      <td>A</td>
      <td>39000000</td>
      <td>440000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2427</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>59.87</td>
      <td>48</td>
      <td>0.0</td>
      <td>A</td>
      <td>39000000</td>
      <td>440000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2428</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>59.87</td>
      <td>241</td>
      <td>0.0</td>
      <td>A</td>
      <td>39000000</td>
      <td>440000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2429</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>59.96</td>
      <td>3</td>
      <td>0.0</td>
      <td>A</td>
      <td>39000000</td>
      <td>440000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2430</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>65.39</td>
      <td>24</td>
      <td>0.0</td>
      <td>A</td>
      <td>39000000</td>
      <td>480000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2431</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>72.82</td>
      <td>64</td>
      <td>0.0</td>
      <td>A</td>
      <td>49000000</td>
      <td>520000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2432</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>74.55</td>
      <td>267</td>
      <td>0.0</td>
      <td>A</td>
      <td>49000000</td>
      <td>535000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2433</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>79.61</td>
      <td>7</td>
      <td>0.0</td>
      <td>A</td>
      <td>55000000</td>
      <td>570000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2434</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>79.74</td>
      <td>35</td>
      <td>0.0</td>
      <td>A</td>
      <td>55000000</td>
      <td>570000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2435</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>79.93</td>
      <td>3</td>
      <td>0.0</td>
      <td>A</td>
      <td>55000000</td>
      <td>570000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2436</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.32</td>
      <td>22</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2437</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.51</td>
      <td>8</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2438</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.58</td>
      <td>2</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>590000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2439</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.59</td>
      <td>10</td>
      <td>0.0</td>
      <td>A</td>
      <td>63000000</td>
      <td>590000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2440</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.82</td>
      <td>6</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2441</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.82</td>
      <td>23</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2442</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.86</td>
      <td>163</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2443</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.89</td>
      <td>3</td>
      <td>0.0</td>
      <td>A</td>
      <td>65000000</td>
      <td>590000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2444</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.91</td>
      <td>126</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2445</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.92</td>
      <td>87</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>585000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2446</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.93</td>
      <td>2</td>
      <td>0.0</td>
      <td>A</td>
      <td>61000000</td>
      <td>590000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2447</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.93</td>
      <td>6</td>
      <td>0.0</td>
      <td>A</td>
      <td>65000000</td>
      <td>590000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2448</th>
      <td>ERR04_2051</td>
      <td>1164</td>
      <td>아파트</td>
      <td>세종특별자치시</td>
      <td>공공임대(10년)</td>
      <td>84.93</td>
      <td>12</td>
      <td>0.0</td>
      <td>A</td>
      <td>65000000</td>
      <td>590000</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>755.0</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>2849</th>
      <td>ERR04_1218</td>
      <td>1048</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>16.45</td>
      <td>336</td>
      <td>28.0</td>
      <td>J</td>
      <td>41200000</td>
      <td>164800</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1140.0</td>
      <td>921.0</td>
    </tr>
    <tr>
      <th>2850</th>
      <td>ERR04_1218</td>
      <td>1048</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>26.52</td>
      <td>180</td>
      <td>28.0</td>
      <td>J</td>
      <td>64400000</td>
      <td>257600</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1140.0</td>
      <td>921.0</td>
    </tr>
    <tr>
      <th>2851</th>
      <td>ERR04_1218</td>
      <td>1048</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>36.37</td>
      <td>524</td>
      <td>28.0</td>
      <td>J</td>
      <td>86800000</td>
      <td>347200</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1140.0</td>
      <td>921.0</td>
    </tr>
    <tr>
      <th>2882</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>16.77</td>
      <td>110</td>
      <td>13.0</td>
      <td>J</td>
      <td>18915000</td>
      <td>75660</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2883</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>16.77</td>
      <td>16</td>
      <td>13.0</td>
      <td>J</td>
      <td>18915000</td>
      <td>75660</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2884</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>23.80</td>
      <td>8</td>
      <td>13.0</td>
      <td>J</td>
      <td>26316000</td>
      <td>105260</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2885</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>23.92</td>
      <td>26</td>
      <td>13.0</td>
      <td>J</td>
      <td>26316000</td>
      <td>105260</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2886</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>23.92</td>
      <td>24</td>
      <td>13.0</td>
      <td>J</td>
      <td>26316000</td>
      <td>105260</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2887</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>30.10</td>
      <td>12</td>
      <td>13.0</td>
      <td>J</td>
      <td>33307000</td>
      <td>133220</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2888</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>30.85</td>
      <td>8</td>
      <td>13.0</td>
      <td>J</td>
      <td>32484000</td>
      <td>129930</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
    <tr>
      <th>2889</th>
      <td>ERR04_1894</td>
      <td>307</td>
      <td>아파트</td>
      <td>경기도</td>
      <td>행복주택</td>
      <td>36.87</td>
      <td>92</td>
      <td>13.0</td>
      <td>J</td>
      <td>38652000</td>
      <td>154610</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>212.0</td>
      <td>419.0</td>
    </tr>
  </tbody>
</table>
</div>



### 데이터 오류 처리 후, csv파일을 만들기


```python
train_df = train.copy()
train_df_errno = train.loc[ ~train['단지코드'].str.contains('ERR'), :]
test_df = test.copy()
```


```python
train_df.to_csv("train_df.csv", index=False)
train_df_errno.to_csv("train_df_errno.csv", index=False)

test_df.to_csv("test_df.csv", index=False)
```

### 데이터 전처리 후, 오류 2,3번 해결된 csv파일 생성됨.
 * train_df.csv : 오류 2번 해결, 3번은 ERR 코드를 붙임.
 * train_df_errno.csv : 오류 2,3번 해결
 * test_df.csv : 오류 코드 제외 csv파일


```python

```
