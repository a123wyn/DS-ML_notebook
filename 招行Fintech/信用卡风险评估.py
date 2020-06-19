#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
import time
import ssl
import os


# In[2]:


# 读取文件
train_tag=pd.read_csv("tag.csv")
train_trd=pd.read_csv("tradition.csv")
train_beh=pd.read_csv("behavior.csv")
test_tag=pd.read_csv("test_tag.csv")
test_trd=pd.read_csv("test_tradition.csv")
test_beh=pd.read_csv("test_behavior.csv")


# # 一、EDA

# In[3]:


# tag数据
train_tag.info()
# trd数据
train_trd.info()
# beh数据
train_beh.info()

test_tag.info()
test_trd.info()
test_beh.info()


# In[4]:


total=train_tag.shape[0]
tradition_total=train_trd.groupby('id').count().shape[0]
behavior_total=train_beh.groupby('id').count().shape[0]
print(total)
print(tradition_total/total) # 大概80%的用户有交易记录
print(behavior_total/total) # 仅有大约30%的用户有APP行为数据
# x=np.array([total,tradition_total,behavior_total])
# plt.figure(figsize=(10,8))
# plt.hist(x)
# plt.xticks(['total','tradition_total','behavior_total'])


# # 二、数据预处理

# ## 1. tag表

# In[5]:


train_tag=pd.read_csv("tag.csv")
test_tag=pd.read_csv("test_tag.csv")


# In[6]:


# 合并训练集与测试集
train_tag_row = train_tag.shape[0] # 训练集与测试集分界线
labels=train_tag['flag']
# 删除标签列
train_tag.drop(['flag'],axis=1,inplace=True)
tag=pd.concat([train_tag, test_tag], axis = 0)


# In[7]:


# 查看有缺失值的字段的情况
print(tag['edu_deg_cd'].value_counts())
print(tag['acdm_deg_cd'].value_counts())
print(tag['deg_cd'].value_counts())
print(tag['atdd_type'].value_counts())

# 缺失值与\N、~字段并不是等价的，可以考虑将这些当做新的种类
# 其中\N字段基本相同，所以可能会因为缺失值补充造成分布比例变化
# 所以有~的可以将缺失值补成~，只有\N的就补充\N
tag['edu_deg_cd'].fillna('~',inplace=True)
tag['acdm_deg_cd'].fillna(r'\N',inplace=True)
tag['deg_cd'].fillna('~',inplace=True)
tag['atdd_type'].fillna(r'\N',inplace=True)


# In[8]:


# 其余特征根据实际情况进行类型转换
# 将int类型的特征的‘\N’进行处理，原则是不要干扰到原来的比例，将\N当做一个新的类型
# columns1将\N转成0是因为字段本身有特殊的一类-1，需要将\N与-1区分开来，故将其置为0
columns1=['frs_agn_dt_cnt','fin_rsk_ases_grd_cd','confirm_rsk_ases_lvl_typ_cd',
         'cust_inv_rsk_endu_lvl_cd','tot_ast_lvl_cd','pot_ast_lvl_cd','hld_crd_card_grd_cd']
for i in columns1:
    tag[i].replace({r'\N':0},inplace=True)
    # 转成int
    tag[i]=tag[i].astype(int)
    
# columns2将\N转成-1，思路其实一样，为了将\N与数据区分开来，字段里表示数字的含有0，故将\N转为-1
columns2=['job_year','l12mon_buy_fin_mng_whl_tms','l12_mon_fnd_buy_whl_tms','l12_mon_insu_buy_whl_tms',
          'l12_mon_gld_buy_whl_tms','ovd_30d_loan_tot_cnt','his_lng_ovd_day','l1y_crd_card_csm_amt_dlm_cd']
for i in columns2:
    tag[i].replace({r'\N':-1},inplace=True)
    # 转成int
    tag[i]=tag[i].astype(int)
    
# 转成str类型
columns3=['gdr_cd','mrg_situ_cd','edu_deg_cd','acdm_deg_cd','deg_cd','ic_ind','fr_or_sh_ind',
          'dnl_mbl_bnk_ind','dnl_bind_cmb_lif_ind','hav_car_grp_ind','hav_hou_grp_ind',
          'l6mon_agn_ind','vld_rsk_ases_ind','loan_act_ind','crd_card_act_ind','atdd_type']
for i in columns3:
    # 转成str
    tag[i]=tag[i].astype(str)


# In[9]:


# 保存补充完后的数据，用于数据分析
completed_tag = tag.iloc[:train_tag_row, :]
completed_tag['flag']=labels.values
completed_tag.to_csv("completed_tag.csv")


# In[10]:


# 将str类型的进行one-hot编码
for i in columns3:
    tag[i]=pd.get_dummies(tag[i],prefix=i,dummy_na=True)


# In[11]:


# 训练集与测试集分开
train_tag=tag.iloc[:train_tag_row,:]
test_tag=tag.iloc[train_tag_row:,:]


# ## 2. tradition表

# In[12]:


train_trd=pd.read_csv("tradition.csv")
test_trd=pd.read_csv("test_tradition.csv")


# In[13]:


# 合并训练集与测试集
train_trd_row = train_trd.shape[0] # 训练集与测试集分界线
# 删除标签列
train_trd.drop(['flag'],axis=1,inplace=True)
trd=pd.concat([train_trd, test_trd], axis = 0)


# In[14]:


# 将交易时间trx_tm特征进行提取，提取出年月日周等信息
trd['date']=trd['trx_tm'].apply(lambda x: x[0:10])
trd['month']=trd['trx_tm'].apply(lambda x: int(x[5:7]))
trd['day_1']=trd['trx_tm'].apply(lambda x: int(x[8:10]))
trd['hour']=trd['trx_tm'].apply(lambda x: int(x[11:13]))
trd['trx_tm']=trd['trx_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
trd['day']=trd['trx_tm'].apply(lambda x: x.dayofyear)
trd['weekday']=trd['trx_tm'].apply(lambda x: x.weekday())
trd['isWeekend']=trd['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
trd['trx_tm']=trd['trx_tm'].apply(lambda x: int(time.mktime(x.timetuple())))


# In[15]:


# 保存补充完后的数据，用于数据分析
completed_trd = trd.iloc[:train_trd_row, :]
completed_trd.to_csv("completed_trd.csv")


# In[16]:


# 对交易方向，支付方式以及收支分类代码进行one-hot编码
columns=['Dat_Flg1_Cd','Dat_Flg3_Cd','Trx_Cod1_Cd','Trx_Cod2_Cd']
for i in columns:
    trd[i]=pd.get_dummies(trd[i],prefix=i,dummy_na=True)


# In[17]:


# 训练集与测试集分开
train_trd=trd.iloc[:train_trd_row,:]
test_trd=trd.iloc[train_trd_row:,:]


# # 三、特征提取

# ## 1. tradition表

# ### 按id进行特征提取，主要是F和M特征

# In[18]:


# trd_id为trd基于id进行的特征提取结果
trd_id=trd[['id']].drop_duplicates().reset_index(drop=True)
trd.sort_values(by=['id','trx_tm'], ascending=True, inplace=True)


# In[19]:


# 提取每个用户的交易总次数、天数以及金额
# 交易总次数
trd['count']=1
trd_count=trd.groupby('id')['count'].agg({'trd_count': 'sum'}).reset_index()
trd_id=pd.merge(trd_id,trd_count,how='left',on='id')
# 交易总天数
day_count=trd.groupby('id')['date'].agg({'day_count': 'nunique'}).reset_index()
trd_id=pd.merge(trd_id,day_count,how='left',on='id')
# 交易总金额
trd_amt=trd.groupby('id')['cny_trx_amt'].agg({'trd_amt': 'sum'}).reset_index()
trd_id=pd.merge(trd_id,trd_amt,how='left',on='id')


# In[20]:


# 平均每天交易次数、交易金额以及每次的平均交易金额
trd_id['avg_perday_trd_count']=trd_id['trd_count']/trd_id['day_count']
trd_id['avg_perday_trd_amt']=trd_id['trd_amt']/trd_id['day_count']
trd_id['avg_pertime_trd_amt']=trd_id['trd_amt']/trd_id['trd_count']


# In[21]:


# 对其余交易方向、交易方式、一级二级代码、分别计总次数与平均每天次数
columns=['Dat_Flg1_Cd','Dat_Flg3_Cd','Trx_Cod1_Cd','Trx_Cod2_Cd']
for i in columns:
    total_count=trd.groupby(['id',i])['count'].agg({'sum'})
    total_count=total_count.unstack().reset_index()
    total_count.fillna(0,inplace=True)
    tmp=list(total_count.columns)
    tmp[0]='id'
    tmp[1:]=[i+'_trd_count_'+str(x[1]) for x in tmp[1:]]
    # 计算总次数
    total_count.columns=tmp
    trd_id=pd.merge(trd_id,total_count,how='left',on='id')
    # 平均每天次数
    for j in total_count.columns[1:]:
        trd_id[i+'_avg_perday_trd_count_'+j.split('_')[-1]]=trd_id[j]/trd_id['day_count']


# In[22]:


# 对其余特征进行分别计金额数,平均每天金额数和平均每次金额数
for i in columns:
    total_amt=trd.groupby(['id',i])['cny_trx_amt'].agg({'sum'})
    total_amt=total_amt.unstack().reset_index()
    total_amt.fillna(0, inplace=True)
    tmp=list(total_amt.columns)
    tmp[0]='id'
    tmp[1:]=[i+'_amt_'+str(x[1]) for x in tmp[1:]]
    # 计算总金额
    total_amt.columns=tmp
    trd_id=pd.merge(trd_id,total_amt,how='left',on='id')
    # 平均每天的交易金额
    for j in total_amt.columns[1:]:
        # 平均每天金额数
        time.sleep(1)
        trd_id[i+'_avg_perday_trd_amt_'+j.split('_')[-1]]=trd_id[j]/trd_id['day_count']
        # 平均每次金额数
        trd_id[i+'_avg_pertime_trd_amt_'+j.split('_')[-1]]=trd_id[j]/trd_id[i+'_trd_count_'+j.split('_')[-1]]


# In[23]:


# 保存文件用于数据分析
trd_id.to_csv("trd_id.csv")


# ### R特征提取（即最近一次交易时间，RFM模型中的R）

# In[24]:


trd_R=trd[['id']].drop_duplicates().reset_index(drop=True)
trd.sort_values(by=['id','trx_tm'],ascending=True,inplace=True)
# 有交易记录的最后一天和最初一天的差值
trd['day']=trd['day']-trd['day'].min()


# In[25]:


# 计算每个id最后一次的交易时间
trd_latest=trd.groupby('id')['day'].agg({'trd_latest': 'max'}).reset_index()
trd_latest['trd_latest']=60-trd_latest['trd_latest']
# 计算该id的最后一个交易时间是否晚于平均值
trd_latest['over_trd_latest_mean']=(trd_latest['trd_latest']>trd_latest['trd_latest'].mean())
trd_R=pd.merge(trd_R,trd_latest,how='left',on='id')


# In[32]:


# 对交易方向、交易方式、一级二级代码、分别计最近一次时间
for i in columns:
    cols_value=trd_R[i].unique()
    # 对每个特征中的每个取值都计算最后一次时间
    for value in cols_value:
        trd_1=trd[(trd[i]==value)]
        trd_latest=trd_1.groupby('id')['day'].agg({i+str(value)+'_trd_latest': 'max'}).reset_index()
        trd_latest[i+str(value)+'_trd_latest']=60-trd_latest[i+str(value)+'_trd_latest']
        trd_latest['over_'+i+str(value)+'_latest_mean']=(trd_latest[i+str(c_v)+'_trd_latest']>trd_latest[i+str(value)+'_trd_latest'].mean())
        trd_R=pd.merge(trd_R,trd_latest,how='left',on='id')

# 带有mean的即为与平均值比较的结果，填补缺失值
for i in trd_R.columns[1:]:
    if 'mean' in i:
        trd_R[i].fillna(-1,inplace=True)
    else:
        trd_R[i].fillna(-1,inplace=True)


# In[ ]:


# 保存文件用于数据分析
trd_R.to_csv("trd_R.csv")


# In[ ]:




