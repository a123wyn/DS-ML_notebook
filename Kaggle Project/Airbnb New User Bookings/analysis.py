import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import time
import datetime

train = pd.read_csv("D:\\software\\Anaconda3\\KaggleData\\airbnb\\train_users_2.csv")
test = pd.read_csv('D:\\software\\Anaconda3\\KaggleData\\airbnb\\test_users.csv')
session = pd.read_csv('D:\\software\\Anaconda3\\KaggleData\\airbnb\\sessions.csv')
#age_gender=pd.read_csv('D:\\software\\Anaconda3\\KaggleData\\airbnb\\age_gender_bkts.csv')

#train data washing
train.info()
'''
RangeIndex: 213451 entries, 0 to 213450
Data columns (total 16 columns):
id                         213451 non-null object   用户id
date_account_created       213451 non-null object   账号创建日期
timestamp_first_active     213451 non-null int64    首次活动时间戳 可能会比用户创建时间或首次预定时间更早 因为有可能在预定之前执行过查找操作
date_first_booking         88908 non-null object    首次预定的日期
gender                     213451 non-null object   性别
age                        125461 non-null float64  年龄
signup_method              213451 non-null object   注册方式
signup_flow                213451 non-null int64    用户注册的界面
language                   213451 non-null object   语言偏好
affiliate_channel          213451 non-null object   营销方式
affiliate_provider         213451 non-null object   营销来源
first_affiliate_tracked    207386 non-null object   在注册之前 用户与之交互的第一个营销广告
signup_app                 213451 non-null object   注册来源
first_device_type          213451 non-null object   注册时设备类型
first_browser              213451 non-null object   注册时使用的浏览器
country_destination        213451 non-null object   目的地国家
dtypes: float64(1), int64(2), object(13)
'''

#null sum
for col in train.columns:
    null_num=train[col].isnull().sum()
    if null_num>0:
        print("column %s \t null sum= %s" % (col, null_num))
'''
column date_first_booking   null sum= 124543
column age  null sum= 87990
column first_affiliate_tracked  null sum= 6065
'''
#date_first_booking null
train['date_first_booking'].head()
'''
0           NaN
1           NaN
2    2010-08-02
3    2012-09-08
4    2010-02-18
Name: date_first_booking, dtype: object
'''
test['date_first_booking'].isnull().sum()/len(test['date_first_booking'])
'''
Out[29]: 1.0
'''
train1=train.drop('date_first_booking', axis=1) #删除某列
test1=test.drop('date_first_booking',axis=1)
#train1.info()

#first_affiliate_tracked 在注册之前，用户与之交互的第一个营销广告 null
#未收集到
train1['first_affiliate_tracked'].head(10)
'''
Out[38]: 
0    untracked
1    untracked
2    untracked
3    untracked
4    untracked
5          omg
6    untracked
7          omg
8    untracked
9          omg
Name: first_affiliate_tracked, dtype: object
'''
train1['first_affiliate_tracked'].fillna('untracked',inplace=True)
test1['first_affiliate_tracked'].fillna('untracked',inplace=True)
train1['first_affiliate_tracked'].isnull().sum()/len(train1['first_affiliate_tracked'])
'''
Out[43]: 0.0
'''
#first_browser
train1['first_browser'].isnull().sum()/len(train1['first_browser'])
test1['first_browser'].isnull().sum()/len(test1['first_browser'])
'''
Out[62]: 0.0
'''
train1['first_browser']
'''
1                Chrome
2                    IE
3               Firefox
4                Chrome
5                Chrome
6                Safari
7                Safari
8                Safari
9               Firefox
10              Firefox
11            -unknown-
12            -unknown-
13              Firefox
'''
#unknown统一成untracked
#train1.loc[train1['first_browser']=='-unknown-','first_browser']='untracked'

#gender 统一
train1.loc[train1['gender']=='OTHER','gender']='-unknown-'
test1.loc[test1['gender']=='OTHER','gender']='-unknown-'


#age 异常值处理

train1['age']=train1['age'].apply(lambda x:np.nan if x>120 or x<0 else x)
test1['age']=test1['age'].apply(lambda x:np.nan if x>120 or x<0 else x)

train1['age'].isnull().sum()/len(train1['age'])
'''
Out[94]: 0.415884676108334
'''
test1['age'].isnull().sum()/len(test1['age'])
'''
Out[24]: 0.46581100231899
'''
'''
1、如果缺值的样本占总数比例极高 就直接舍弃 作为特征加入的话 会带入noise
2、如果缺值的样本占比适中 而该属性非连续值特征属性（比如说类目属性） 就把NAN作为一个新类别 加到类别特征中
3、如果缺值的样本占比适中 而该属性为连续值特征属性 考虑给定一个step 离散化 吧NAN作为一个type加到属性类目中
4、如果缺失的样本占比不是很大 根据已有的值拟合数据
'''
def simplify_ages(df):
    # 把缺失值补上，方便分组 离散化
    df.age = df.age.fillna(-0.5)

    # 把Age分为不同区间,-1到0,0-3,3-12...,60及以上,放到bins里，八个区间，对应的八个区间名称在group_names那
    bins = [-1, 0, 10, 20, 30, 40, 50, 60, 120]
    group_names = ['Unknown', '[0,10)', '[10,20)', '[20,30)', '[30,40)', '[40,50)', '[50,60)','[60,120]']

    # 开始对数据进行离散化，pandas.cut就是这个功能
    catagories = pd.cut(df['age'], bins, labels=group_names,right=False)
    df.age = catagories
    #print(catagories.value_counts())
    return df
train2=simplify_ages(train1.copy())
test2=simplify_ages(test1.copy())
'''
### 使用 RandomForestRegressor 填补缺失的年龄属性

#train['date_account_created']  str转换成int
train1['dac_train_int_dt']=train1['date_account_created'].apply(lambda x: int(x[0:4]+x[5:7]+x[8:]))

from sklearn.preprocessing import OneHotEncoder
gender_enc = pd.get_dummies(train1[['gender']])
browser_enc=pd.get_dummies(train1[['first_browser']])
device_enc=pd.get_dummies(train1[['first_device_type']])
app_enc=pd.get_dummies(train1[['signup_app']])
affili_enc=pd.get_dummies(train1[['affiliate_provider']])
affilitrac_enc=pd.get_dummies(train1[['first_affiliate_tracked']])
affilichann_enc=pd.get_dummies(train1[['affiliate_channel']])
lang_enc=pd.get_dummies(train1[['language']])
signupme_enc=pd.get_dummies(train1[['signup_method']])
country_enc=pd.get_dummies(train1[['country_destination']])
train3 = pd.concat([train3, device_enc,app_enc,affili_enc,affilitrac_enc,affilichann_enc,lang_enc,signupme_enc], axis=1)
#train3=train3.drop('date_account_created','gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser', axis=1)
#train3=train3.drop(train3.columns[4:9],axis=1) 类似



from sklearn.ensemble import RandomForestRegressor
def rtfillage(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    #age_df = df[['age', 'timestamp_first_active', 'signup_flow','dac_train_int_dt']]

    #把age放在第一列
    df_age = df.age
    df = df.drop('age', axis=1)
    df.insert(0, 'age', df_age)

    #id也不可以放


    #整合起来成列表
    age_df=df[df.columns.tolist()]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.age.notnull()].values
    unknown_age = age_df[age_df.age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.age.isnull()), 'age'] = predictedAges

    return df, rfr

train3,rfr=rtfillage(train3.copy())
'''

#数据类型转换
train1['dac_train_dt']=pd.to_datetime(train['date_account_created'],format='%Y-%m-%d',errors='coerce')
train1['tfa_train_dt']=pd.to_datetime(train['timestamp_first_active'],format='%Y%m%d%H%M%S',errors='coerce')

test2['dac_train_dt']=pd.to_datetime(test2['date_account_created'],format='%Y-%m-%d',errors='coerce')
test2['tfa_train_dt']=pd.to_datetime(test2['timestamp_first_active'],format='%Y%m%d%H%M%S',errors='coerce')

#重复值处理
len(train['id'].unique())
'''
Out[76]: 213451
'''
train['id'].count()
'''
Out[72]: 213451
'''
len(test2['id'].unique())
'''
Out[26]: 62096
'''
test2['id'].count()
'''
Out[27]: 62096
'''
#没有重复值


#session data washing
session.info()
'''
RangeIndex: 10567737 entries, 0 to 10567736
Data columns (total 6 columns):
user_id          object
action           object     
action_type      object
action_detail    object
device_type      object
secs_elapsed     float64
dtypes: float64(1), object(5)
'''
#action null
session['action'].isnull().sum()
'''
Out[34]: 79626
'''
session['action'].fillna('-unknown-',inplace=True)

#action_type null
session['action_type'].isnull().sum()
'''
Out[80]: 1126204
'''
session['action_type'].fillna('-unknown-',inplace=True)

#action_detail
session['action_detail'].isnull().sum()
'''
Out[87]: 1126204
'''
session['action_detail'].fillna('-unknown-',inplace=True)

#secs_elapsed null
session['secs_elapsed'].isnull().sum()/len(session['secs_elapsed'])
'''
Out[110]: 0.012872292336571207
'''
#secs_elapsed 缺失值填0处理
session['secs_elapsed'].fillna(0,inplace=True)

#action_type null
session['action_type'].isnull().sum()/len(session['action_type'])
'''
Out[11]: 0.10657002535169072
'''
session['action_type'].fillna('-unknown-',inplace=True)

#重复值处理 不知道需不需要 不需要
session1=session.drop_duplicates(keep='first')

#数据分析
'''
用户画像
    性别比例
    年龄分布
    语言偏好 地区分布
    目的国家分布
转化漏斗
    用户量
    注册用户量
    活跃用户（非僵尸用户）占比
    预定用户占比
    付款用户占比
    复购用户占比
流量指标
    每日新增用户量
    不同营销方式带来的注册量
    不同营销内容带来的注册量
    不同用户端带来的注册量
    不同设备。。。
    不同浏览器
用户行为

'''
#用户画像

#性别分布
gender_num=train1.groupby('gender')['id'].count()
gender_num.drop('-unknown-',axis=0,inplace=True)

#柱状图
trace = [go.Bar(x=gender_num.index.tolist(),
                y=gender_num.values.tolist(),
                text=gender_num.values.tolist(),
                textposition='auto',
                marker= dict(color=['red','blue'],opacity=0.5))]

layout = go.Layout(title='Airbnb用户性别分布', xaxis=dict(title='gender'),yaxis=dict(title='count'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb用户性别分布.html')

#饼图

trace=go.Pie(labels=gender_num.index.tolist(),
             values=gender_num.values.tolist(),
             #hole=0,
             text=['女性用户','男性用户'],
             textfont=dict(size=12,color='white'),
             showlegend=False,
             opacity=0.9)
data=[trace]
fig = dict(data=data)
py.offline.plot(fig, filename='Airbnb用户性别分布.html')


#年龄分布
#用train2
age_num=train2.groupby('age')['id'].count()
age_num.drop('Unknown',axis=0,inplace=True)

trace = [go.Bar(x=age_num.index.tolist(),
                y=age_num.values.tolist(),
                text=age_num.values.tolist(),
                textposition='outside',
                marker= dict(color=['orange','orange','orange','red','orange','orange','orange'],opacity=0.5))]

layout = go.Layout(title='Airbnb用户年龄段分布', xaxis=dict(title='年龄段'),yaxis=dict(title='数量/人'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb用户年龄段分布.html')

#语言偏好
lang_num=train1.groupby('language')['id'].count().sort_values(ascending=False)[:10]

trace = [go.Bar(y=lang_num.index.tolist()[::-1],
                x=lang_num.values.tolist()[::-1],
                text=lang_num.values.tolist()[::-1],
                textposition='outside',
                orientation = 'h',
                marker= dict(opacity=0.5))]

layout = go.Layout(title='Airbnb用户语言偏好情况', yaxis=dict(title='语种'),xaxis=dict(title='数量/人'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb用户语言偏好情况.html')

#地区分布
#地图
#英语地区太多不好划分 可能没必要

#目的国家分布
descoun_num=train1.groupby('country_destination')['id'].count()
descoun_num.drop('NDF',axis=0,inplace=True)
descoun_num.drop('other',axis=0,inplace=True)

descoun_num_df=pd.DataFrame({'num':descoun_num.values,'country':descoun_num.index})

#country short name convert
descoun_num_df.loc[descoun_num_df['country']=='AU','country']='AUS'
descoun_num_df.loc[descoun_num_df['country']=='CA','country']='CAN'
descoun_num_df.loc[descoun_num_df['country']=='DE','country']='DEU'
descoun_num_df.loc[descoun_num_df['country']=='ES','country']='ESP'
descoun_num_df.loc[descoun_num_df['country']=='FR','country']='FRA'
descoun_num_df.loc[descoun_num_df['country']=='GB','country']='GBR'
descoun_num_df.loc[descoun_num_df['country']=='IT','country']='ITA'
descoun_num_df.loc[descoun_num_df['country']=='NL','country']='NLD'
descoun_num_df.loc[descoun_num_df['country']=='PT','country']='PRT'
descoun_num_df.loc[descoun_num_df['country']=='US','country']='USA'

import plotly.express as px

def get_color(number):
    return 'rgb({r}, 0, {b})'.format(
        r=(number-217)/100*255,
        b=(1-(number-217)/100)*255
    )

fig = px.choropleth(descoun_num_df,
                    locations = 'country',
                    color='num',
                    #color_continuous_scale=[get_color(number) for number in descoun_num_df.num.tolist()],
                    color_continuous_scale='turbid',
                    projection = 'natural earth')

py.offline.plot(fig, auto_open=True,filename='Airbnb用户目的国家分布.html')

#转化漏斗

#用户总数量
user_total_num=len(session.groupby('user_id')['user_id'])
#活跃用户总数量
l=session.groupby('user_id')['user_id'].count()>10
active_user_num=len(l[l==True])
#注册用户总数量
from pandas import merge
l=merge(train1,session,left_on='id',right_on='user_id')
register_user_num=len(l.groupby('id')['id'])
#下单用户总数量
l=session[session['action_detail']=='reservations'].groupby('user_id')['user_id']
reservations_user_num=len(l)
#实际支付用户总数量
l=session[session['action_detail']=='payment_instruments'].groupby('user_id')['user_id']
pay_user_num=len(l)
#复购支付用户总数量
l=session[session['action_detail']=='payment_instruments']['user_id'].value_counts()>=2
repay_user_num=len(l[l==True])

#画图
trace = go.Funnel(x = [user_total_num,active_user_num,register_user_num,reservations_user_num,pay_user_num,repay_user_num],
                  y = ["用户总数量","活跃用户总数量","注册用户总数量","下单用户总数量","实际支付用户总数量","复购支付用户总数量"],
                   textinfo = "value+percent initial+percent previous",
                   textposition='auto',
                   textfont=dict(size=12),
                   marker=dict(color=["red"]*6),
                   connector = {"line": {"color": "lightsalmon", "dash": "solid"}},
                   opacity=0.5)

layout = go.Layout(title='Airbnb用户转化漏斗图',dragmode="pan")

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb用户转化漏斗图.html')

#流量指标
#每月用户增长情况
dac={}
l=range(1,10)
for item in train1['dac_train_dt']:
    if item.month in l:
        mon='0'+'%s' %str(item.month)
    else:
        mon="%s" %str(item.month)
    key="%d-%s" % (item.year,mon)
    if dac.__contains__(key)==True:
        dac[key]=dac[key]+1
    else:
        dac[key]=0
dac_series=pd.Series(dac).sort_index()
trace = go.Scatter(
    x=dac_series.index.tolist(),
    #xcalendar="%Y-%M",
    y=dac_series.values.tolist(),
    mode='markers+lines',
    connectgaps=False #是否连接缺失值
)
layout = go.Layout(title='Airbnb每月用户增长情况',xaxis = dict(title = '年月'),
              yaxis = dict(title = '人数'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb每月用户增长情况.html')

#不同营销方式带来的注册量
df=train1.groupby(['affiliate_channel','affiliate_provider']).size()
ps=dict(df)

x=train1['affiliate_provider'].value_counts().index.tolist()
y=train1['affiliate_channel'].value_counts().index.tolist()
z = np.zeros((len(y), len(x)), dtype=np.int)
for channel in y:
    for pro in x:
        z[y.index(channel)][x.index(pro)]=ps.get(tuple([channel,pro]), 0)
trace=go.Heatmap(
    z=z,
    x=x,
    y=y,
    colorscale='Reds'
    #text=dict(x=x, y=y)
)

layout = go.Layout(title='不同营销方式带来的注册量', xaxis=dict(title='流量渠道'),yaxis=dict(title='推广渠道'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb不同营销方式带来的注册量.html')

#不同营销内容带来的注册量
affili_track_num=train1.groupby('first_affiliate_tracked')['id'].count()
affili_track_num.drop('untracked',axis=0,inplace=True)

trace = [go.Bar(x=affili_track_num.index.tolist(),
                y=affili_track_num.values.tolist(),
                text=affili_track_num.values.tolist(),
                textposition='outside',
                marker= dict(color=['red','orange','orange','red','orange','orange'],opacity=0.5))]

layout = go.Layout(title='不同营销内容带来的注册量', xaxis=dict(title='营销内容'),yaxis=dict(title='注册人数/人'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb不同营销内容带来的注册量.html')

#不同用户端带来的注册量
signup_app_num=train1.groupby('signup_app')['id'].count()

trace = [go.Bar(x=signup_app_num.index.tolist(),
                y=signup_app_num.values.tolist(),
                text=signup_app_num.values.tolist(),
                textposition='outside',
                marker= dict(color=['orange','orange','red','orange'],opacity=0.5))]

layout = go.Layout(title='不同用户端带来的注册量', xaxis=dict(title='用户端'),yaxis=dict(title='注册人数/人'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb不同用户端带来的注册量.html')

#不同设备。。。
devi_num=train1.groupby('first_device_type')['id'].count()

trace = [go.Bar(x=devi_num.index.tolist(),
                y=devi_num.values.tolist(),
                text=devi_num.values.tolist(),
                textposition='outside',
                marker= dict(color=['orange','orange','orange','red','orange','orange','red','orange','orange'],opacity=0.5))]

layout = go.Layout(title='不同设备类型带来的注册量', xaxis=dict(title='设备类型'),yaxis=dict(title='注册人数/人'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb不同设备类型带来的注册量.html')


#不同浏览器
browser_num=train1.groupby('first_browser')['id'].count()
browser_num.drop('-unknown-',axis=0,inplace=True)
browser_num=browser_num.sort_values(ascending=False)[:10]
#水平柱状图
trace = [go.Bar(y=browser_num.index.tolist()[::-1],
                x=browser_num.values.tolist()[::-1],
                text=browser_num.values.tolist()[::-1],
                textposition='outside',
                orientation = 'h',
                marker= dict(opacity=0.5))]

layout = go.Layout(title='不同浏览器带来的注册量', xaxis=dict(title='浏览器'),yaxis=dict(title='注册人数/人'))

figure = go.Figure(data=trace, layout=layout)
py.offline.plot(figure, auto_open=True,filename='Airbnb不同浏览器带来的注册量.html')

#session特征提取
#这是为了后面的数据合并 将session里的user_id改名为id
session['id'] = session['user_id']
session = session.drop(['user_id'],axis=1) #按行删除


# 对action特征进行细化
f_act = session.action.value_counts().argsort() #argsort函数返回的是数组值从小到大的索引值
f_act_detail = session.action_detail.value_counts().argsort()
f_act_type = session.action_type.value_counts().argsort()
f_dev_type = session.device_type.value_counts().argsort()

# 按照id进行分组
dgr_sess = session.groupby(['id'])
# Loop on dgr_sess to create all the features.
samples = []  # samples列表
ln = len(dgr_sess)  # 计算分组后df_sessions的长度
k=1

def session_feature():
        global k
        for g in dgr_sess:  # 对dgr_sess中每个id的数据进行遍历
            gr = g[1]  # data frame that comtains all the data for a groupby value 'zzywmcn0jv'

            l = []  # 建一个空列表，临时存放特征

            # the id    for example:'zzywmcn0jv'
            l.append(g[0])  # 将id值放入空列表中

            # number of total actions
            l.append(len(gr))  # 将id对应数据的长度放入列表

            # action features 特征-用户行为
            # 每个用户行为出现的次数，各个行为类型的数量，平均值以及标准差
            c_act = [0] * len(f_act)
            for i, v in enumerate(gr.action.values):  # i是从0-1对应的位置，v 是用户行为特征的值
                c_act[f_act[v]] += 1
            _, c_act_uqc = np.unique(gr.action.values, return_counts=True)
            # 计算用户行为行为特征各个类型数量的长度，平均值以及标准差
            c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]
            l = l + c_act

            # action_detail features 特征-用户行为具体
            # (how many times each value occurs, numb of unique values, mean and std)
            c_act_detail = [0] * len(f_act_detail)
            for i, v in enumerate(gr.action_detail.values):
                c_act_detail[f_act_detail[v]] += 1
            _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)
            c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]
            l = l + c_act_detail

            # action_type features  特征-用户行为类型 click等
            # (how many times each value occurs, numb of unique values, mean and std
            # + log of the sum of secs_elapsed for each value)
            l_act_type = [0] * len(f_act_type)
            c_act_type = [0] * len(f_act_type)
            sev = gr.secs_elapsed.values
            for i, v in enumerate(gr.action_type.values):
                l_act_type[f_act_type[v]] += sev[i]  # sev = gr.secs_elapsed.fillna(0).values ，求每个行为类型总的停留时长
                c_act_type[f_act_type[v]] += 1
            l_act_type = np.log(1 + np.array(l_act_type)).tolist()  # 每个行为类型总的停留时长，差异比较大，进行log处理
            _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)
            c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]
            l = l + c_act_type + l_act_type

            # device_type features 特征-设备类型
            # (how many times each value occurs, numb of unique values, mean and std)
            c_dev_type = [0] * len(f_dev_type)
            for i, v in enumerate(gr.device_type.values):
                c_dev_type[f_dev_type[v]] += 1
            c_dev_type.append(len(np.unique(gr.device_type.values)))
            _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)
            c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]
            l = l + c_dev_type

            # secs_elapsed features  特征-停留时长
            l_secs = [0] * 5
            l_log = [0] * 15
            if len(sev) > 0:
                # Simple statistics about the secs_elapsed values.
                l_secs[0] = np.log(1 + np.sum(sev))
                l_secs[1] = np.log(1 + np.mean(sev))
                l_secs[2] = np.log(1 + np.std(sev))
                l_secs[3] = np.log(1 + np.median(sev))
                l_secs[4] = l_secs[0] / float(l[1])  #

                # Values are grouped in 15 intervals. Compute the number of values
                # in each interval.
                # sev = gr.secs_elapsed.fillna(0).values
                log_sev = np.log(1 + sev).astype(int)
                # np.bincount():Count number of occurrences of each value in array of non-negative ints.
                l_log = np.bincount(log_sev, minlength=15).tolist()
                print("\r进度：%.2f%%" % (float(k / len(dgr_sess) * 100)), end=' ')
                k=k+1
            l = l + l_secs + l_log

            # The list l has the feature values of one sample.
            samples.append(l)
        print("\n")

session_feature()

# preparing objects
samples = np.array(samples)
samp_ar = samples[:, 1:].astype(np.float16)  # 取除id外的特征数据
samp_id = samples[:, 0]  # 取id，id位于第一列

# 为提取的特征创建一个dataframe
col_names = []  # name of the columns
for i in range(len(samples[0]) - 1):  # 减1的原因是因为有个id
    col_names.append('c_' + str(i))  # 起名字的方式
df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)
df_agg_sess['id'] = samp_id
df_agg_sess.index = df_agg_sess.id  # 将id作为index
#经过特征提取后，session文件由6个特征变为584个特征

# 对train和test文件进行特征提取

