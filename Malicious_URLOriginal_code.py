#!/usr/bin/env python
# coding: utf-8

# Imported data for training model

import pandas as pd
import numpy as np
import re
from autocorrect import spell  
from textdistance import levenshtein
from pandas.io.json import json_normalize
from manual_spellchecker import spell_checker
from sklearn.preprocessing import LabelEncoder
import pickle
import vaex
from webscraping import webscraping
import mailmail
#from testcheckrange import check_ipv4_in


df = vaex.from_json('data4.json', lines=True)
df_pandas = df._source.to_pandas_series()
dff1 = pd.json_normalize(df_pandas)
df2 = dff1[['hostname']]
print(df2)
#import Test data
'''df = pd.read_json('data3.json', lines=True)
df2 = df[['hostname']]
print(df)'''
df2.rename(columns={'hostname':'url'},inplace=True)
df2.dropna(inplace=True)
df3=df2.drop(df2[df2.url.str.contains(r'^-')].index)


# Remove whitespace in any rows
df3['url']=df3['url'].str.strip()

# Lower case
df3['url']=df3['url'].str.lower()
 
#  Cut_http
df3['url']=df3['url'].replace(r'\w{1,5}:\/\/|\w{1,5}:\\{1,3}',' ',regex=True)
df3['url']=df3['url'].str.strip()

# Slash_all แบ่งข้อมูลที่อยู่หลัง path เราไม่เอา เอาแค่ประโยคที่อยู่หน้า path

df3['slash_all']=df3['url'].str.extract('(?P<slash_all>^.*?/.*?)')
df3['slash_all'].fillna(df3['url'],inplace= True)

# TLD
df3['TLD']=df3['slash_all'].str.extract('(?P<TLD>[^.]+$)') 

#  Delete slash and space from TLD 
df3['TLD'] = df3['TLD'].str.replace('/','')
df3['TLD'] = df3['TLD'] .str.strip() 
df3['slash_all'] = df3['slash_all'].str.replace('/','')
df3['slash_all'] = df3['slash_all'] .str.strip() 

#  Subdomain
df3['subdomain'] =df3['slash_all'].str.extract('(?P<subdomain>^[\w\d\s]+|^[\W]+\w+)') 

# split
df3['split']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?P<sliit>\W+)')
df3['slash']=df3['split'].str.contains('/')

# Subdomain1+ split1
df3['subdomain1']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?P<subdomain1>[\w\d\s]+)')
df3['split1']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?:[\w\d\s]+)(?P<split1>\W+)')

# Subdomain2 +split2
df3['subdomain2']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?:[\w\d\s]+)(?:\W+)(?P<subdomain2>[\w\s\d]+)')
df3['split2']=df3['slash_all'].str.extract('(?:^[\w\d\s]+|^[\W]+\w+)(?:\W+)(?:[\w\d\s]+)(?:\W+)(?:[\w\s\d]+)(?P<split2>\W+)')

# slash_num
df3['slash_num']=df3['slash_all'].str.count('(?P<slash_num>\d)')
df3['slash_ch']=df3['slash_all'].str.count('(?P<slash_ch>\W)')
df3['slash_word']=df3['slash_all'].str.count('(?P<slash_word>[a-zA-Z])')
sum_column = df3["slash_num"] + df3["slash_word"] + df3["slash_ch"] 
df3["slash_all_score"] = sum_column
df3['slash_w']=df3['slash_all'].str.count('(?P<slash_w>\.)') #นับว่ามี dot มากแค่ไหนใน 1 url
sum1_column = df3["slash_ch"] - df3["slash_w"] 
df3["not_dot"] = sum1_column

# one 
len(df3.slash_all_score.unique())
df4= df3.slash_all_score.unique()
df5=pd.DataFrame(df4,columns=['slash_all_score'])
df5['one']=df5['slash_all_score']
for i in range(len(df5["slash_all_score"])):
    if df5["slash_all_score"][i] <= 15 :
        print("1")
        df5["one"][i] = "1"
    elif df5['slash_all_score'][i] ==16:
        print("2")
        df5["one"][i] = "2"
    elif df5["slash_all_score"][i] ==17:
        print("3")
        df5["one"][i] = "3"
    elif df5["slash_all_score"][i] ==18:
        print("4")
        df5["one"][i] = "4"
    elif df5["slash_all_score"][i] >=19:
        print("5")
        df5["one"][i] = "5"

df6 = pd.merge(df3,df5,how='left',on= 'slash_all_score',indicator=True)
df7 = df6.drop('_merge', 1)

# two
len(df7.slash_num.unique())
df8=df7.slash_num.unique()
df9=pd.DataFrame(df8,columns=['slash_num'])
df9['two']=df9['slash_num']
for i in range(len(df9["slash_num"])):
    if df9["slash_num"][i] <= 3 :
        print("1")
        df9["two"][i] = "1"
    elif df9["slash_num"][i] ==4:
        print("2")
        df9["two"][i] = "2"
    elif df9["slash_num"][i] ==5:
        print("3")
        df9["two"][i] = "3"
    elif df9["slash_num"][i] ==6:
        print("4")
        df9["two"][i] = "4"
    elif df9["slash_num"][i] ==7:
        print("5")
        df9["two"][i] = "5"
    elif df9["slash_num"][i] ==8:
        print("6")
        df9["two"][i] = "6"
    elif df9["slash_num"][i] >=9:
        print("7")
        df9["two"][i] = "7"

df10 = pd.merge(df7,df9,how='left',on= 'slash_num',indicator=True)
df11= df10.drop('_merge', 1)

# TLD ต่อมาเราจะทำการ match TLD
dt = pd.read_csv('TLD_real.csv')
dt.rename(columns={'TLD_real' :'TLD'},inplace=True)
len(dt.TLD.unique())
dt1=dt[['TLD']]
dt2= dt1.TLD.unique()
dt3=pd.DataFrame(dt2,columns=['TLD'])
dt3['TLD'] = dt3['TLD'].str.strip()

#  Unique : TLD from Training set 
len(df11.TLD.unique())
df11['TLD'] = df11['TLD'].str.strip()
df12=df11.TLD.unique()
df13=pd.DataFrame(df12,columns=['TLD'])
df13['TLD_real']=df13['TLD']
df14 = pd.merge(df13,dt3,on='TLD',how='left',indicator=True)
df15 =df14[['TLD','_merge']]
df16 = pd.merge(df11,df15,how='left',on= 'TLD',indicator='exists')
df17= df16.drop(['exists'],axis=1)
dg=df17.replace({'_merge':{'both': 1,'left_only':0}})
dg.rename(columns ={'_merge':'TLD_real'},inplace=True)

# subdomain-num
dg['subdomain_num']=dg['subdomain'].str.count('(?P<subdomain_num>\d)')
dg['subdomain_word']=dg['subdomain'].str.count('(?P<subdomain_word>[a-zA-Z])')
dg['subdomain_all']=dg['subdomain'].str.count('(?P<subdomain_all>\w)')

# subdomain1
dg['subdomain1_num']=dg['subdomain1'].str.count('(?P<subdomain1_num>\d)') #นับว่ามีตัวเลขมากแค่ไหน
dg['subdomain1_word']=dg['subdomain1'].str.count('(?P<subdomain1_word>[a-zA-Z])') #นับว่ามีตัวเลขมากแค่ไหน
dg['subdomain1_all']=dg['subdomain1'].str.count('(?P<subdomain1_all>\w)')

# # subdomain2 
dg['subdomain2_num']=dg['subdomain2'].str.count('(?P<subdomain2_num>\d)') #นับว่ามีตัวเลขมากแค่ไหน
dg['subdomain2_word']=dg['subdomain2'].str.count('(?P<subdomain2_word>[a-zA-Z])') #นับว่ามีตัวเลขมากแค่ไหน
dg['subdomain2_all']=dg['subdomain2'].str.count('(?P<subdomain2_all>\w)')

#  split
dg['split']=dg['split'].str.strip()
dg['split_dot']=dg['split'].str.match('(?P<split_dot>^\.$)')
dg['split_count']=dg['split'].str.count('(?P<split_count>\W)')

#  split2
dg['split1']=dg['split1'].str.strip()
dg['split1_dot']=dg['split1'].str.match('(?P<split1_dot>^\.$)')
dg['split1_count']=dg['split1'].str.count('(?P<split1_count>\W)')

#  First_condition
len(dg.subdomain_all.unique())
dg1=dg.subdomain_all.unique()
dg2=pd.DataFrame(dg1,columns=['subdomain_all'])
dg2['first_con']=dg2['subdomain_all']
for i in range(len(dg2["subdomain_all"])):
    if dg2["subdomain_all"][i] <= 10 :
        print("1")
        dg2["first_con"][i] = "1"
    elif dg2["subdomain_all"][i] ==11:
        print("2")
        dg2["first_con"][i] = "2"
    elif dg2["subdomain_all"][i] ==12:
        print("3")
        dg2["first_con"][i] = "3"
    elif dg2["subdomain_all"][i] ==13:
        print("4")
        dg2["first_con"][i] = "4"
    elif dg2["subdomain_all"][i] ==14:
        print("5")
        dg2["first_con"][i] = "5"
    elif dg2["subdomain_all"][i] ==15:
        print("6")
        dg2["first_con"][i] = "6"
    elif dg2["subdomain_all"][i] ==16:
        print("7")
        dg2["first_con"][i] = "7"
    elif dg2["subdomain_all"][i] ==17:
        print("8")
        dg2["first_con"][i] = "8"
    elif dg2["subdomain_all"][i] ==18:
        print("9")
        dg2["first_con"][i] = "9"
    elif dg2["subdomain_all"][i] >=19:
        print("10")
        dg2["first_con"][i] = "10"

dg3= pd.merge(dg,dg2,how='left',on= 'subdomain_all',indicator=True)
dg4 = dg3.drop('_merge', 1)

#  Second_condition 
len(dg4.subdomain_num.unique())
dg5=dg4.subdomain_num.unique()
dg6=pd.DataFrame(dg5,columns=['subdomain_num'])
dg6['second_con']=dg6['subdomain_num']
for i in range(len(dg6["subdomain_num"])):
    if dg6["subdomain_num"][i] <= 3 :
        print("1")
        dg6["second_con"][i] = "1"
    elif dg6["subdomain_num"][i] == 4:
        print("2")
        dg6["second_con"][i] = "2"
    elif dg6["subdomain_num"][i] ==5:
        print("3")
        dg6["second_con"][i] = "3"
    elif dg6["subdomain_num"][i] ==6:
        print("4")
        dg6["second_con"][i] = "4"
    elif dg6["subdomain_num"][i] ==7:
        print("5")
        dg6["second_con"][i] = "5"
    elif dg6["subdomain_num"][i] ==8:
        print("6")
        dg6["second_con"][i] = "6"
    elif dg6["subdomain_num"][i] >=9:
        print("7")
        dg6["second_con"][i] = "7"
        
dg7 = pd.merge(dg4,dg6,how='left',on= 'subdomain_num',indicator=True)
dg8 = dg7.drop('_merge', 1)

#  Third condition 
dd= dg8['split_dot'].map(str)+dg8['split_count'].map(str)
dd1=pd.DataFrame(dd,columns=['split_dc'])
dd2 = pd.concat([dg8,dd1],axis =1)
dd3=dd2.split_dc.unique()
dd4=pd.DataFrame(dd3,columns=['split_dc'])
dd4['third_con']=dd4['split_dc']
for i in range(len(dd4["split_dc"])):
    if dd4["split_dc"][i] == 'True1.0' :
        print("0")
        dd4["third_con"][i] = "0"
    elif dd4["split_dc"][i] =='False1.0':
        print("1")
        dd4["third_con"][i] = "1"
    elif dd4["split_dc"][i] =='False2.0':
        print("2")
        dd4["third_con"][i] = "2"
    elif dd4["split_dc"][i] =='False3.0':
        print("3")
        dd4["third_con"][i] = "3"
    elif dd4["split_dc"][i] == 'False4.0':
        print("4")
        dd4["third_con"][i] = "4"
    elif dd4["split_dc"][i] == 'False5.0':
        print("5")
        dd4["third_con"][i] = "5"
        
    elif dd4["split_dc"][i] == 'True1' :
        print("7")
        dd4["third_con"][i] = "7"
    elif dd4["split_dc"][i] =='False1':
        print("8")
        dd4["third_con"][i] = "8"
    elif dd4["split_dc"][i] =='False2':
        print("9")
        dd4["third_con"][i] = "9"
    elif dd4["split_dc"][i] =='False3':
        print("10")
        dd4["third_con"][i] = "10"
    elif dd4["split_dc"][i] == 'False4':
        print("11")
        dd4["third_con"][i] = "11"
    elif dd4["split_dc"][i] == 'False5':
        print("12")
        dd4["third_con"][i] = "12"
    else:
        print('13')
        dd4["third_con"][i] = "13"
dg9 = pd.merge(dd2,dd4,how='left',on= 'split_dc',indicator=True)
ds3 = dg9.drop('_merge', 1)
ds3['split_dot']=ds3['split_dot']*1 #true=1
ds3['split1_dot']=ds3['split1_dot']*1

#  Brand name word
pp = pd.read_csv('Domain.csv')
pp.rename(columns = {"Domain":"slash_all"}, inplace = True)

#  Matching brand name word
ds3['slash_all'] = ds3['slash_all'].str.strip()
ds3['slash_all']= ds3['slash_all'].astype('string')
pp['slash_all']= pp['slash_all'].astype('string')
d2=pd.merge(ds3,pp,on='slash_all',how='left',indicator=True)
d6=d2.replace({'_merge':{'both': 1,'left_only':0}})
d6.rename(columns ={'_merge':'brand_name'},inplace=True)
#  Random word  Detection

#  Random_subdomain
d6['subdomain_d']=d6['subdomain']
d6["subdomain_d"]= d6["subdomain_d"].astype(str)
d6['subdomain_d']=d6['subdomain_d'].str.strip()
d6[['subdomain_d']] = d6[['subdomain_d']].fillna('N')
ob = spell_checker(d6,'subdomain_d')
ob.spell_check()
dd=ob.get_all_errors()
d7 = pd.DataFrame(dd,columns=['subdomain_d'])
d8= d7[~d7.subdomain_d.str.contains("www")]
d9 = d6.merge(d8.drop_duplicates(subset=['subdomain_d']), how='left', indicator='_merge')
dd =d9.loc[d9['_merge']== 'left_only'] #row เหล่านี้คือคำที่สะกดถูก 
dt=d9.replace({'_merge':{'both': 1,'left_only':0}})#both=1= คือสะกดผิด >> left_only = 0=สะกดถูก
dt.rename(columns ={'_merge':'random_sub_d'},inplace=True)

# Random_subdomain1
dt['subdomain1_d']=dt['subdomain1']
dt["subdomain1_d"]= dt["subdomain1_d"].astype(str)
dt['subdomain1_d']=dt['subdomain1_d'].str.strip()
dt[['subdomain1_d']] = dt[['subdomain1_d']].fillna('N')

#  manual-spellchecker
ob1 = spell_checker(dt,'subdomain1_d')
ob1.spell_check()
ff=ob1.get_all_errors()
hh = pd.DataFrame(ff,columns=['subdomain1_d'])
h1 = dt.merge(hh.drop_duplicates(subset=['subdomain1_d']), how='left', indicator='_merge')
h2=h1.replace({'_merge':{'both': 1,'left_only':0}})#both=1= คือสะกดผิด >> left_only = 0=สะกดถูก
h2.rename(columns ={'_merge':'random_sub1_d'},inplace=True)

# Random_subdomain2
h2['subdomain2_d']=h2['subdomain2']
h2["subdomain2_d"]= h2["subdomain2_d"].astype(str)
h2['subdomain2_d']=h2['subdomain2_d'].str.strip()
h2[['subdomain2_d']] = h2[['subdomain2_d']].fillna('N')
ob2 = spell_checker(h2,'subdomain2_d')
ob2.spell_check()
f=ob2.get_all_errors()
h3 = pd.DataFrame(f,columns=['subdomain2_d'])
h4= h2.merge(h3.drop_duplicates(subset=['subdomain2_d']), how='left', indicator='_merge')
h5=h4.replace({'_merge':{'both': 1,'left_only':0}})#both=1= คือสะกดผิด >> left_only = 0=สะกดถูก
h5.rename(columns ={'_merge':'random_sub2_d'},inplace=True)

# Dictionary_check
h5['dictionary_check']=h5['subdomain']
h5['dictionary_check'] = h5['dictionary_check'].str.replace('\d+', '')
h5['dictionary_check'] = h5['dictionary_check'].str.strip()
h5[['dictionary_check']] = h5[['dictionary_check']].fillna('N')
dc = spell_checker(h5,'dictionary_check')
dc.spell_check()
dcc=dc.get_all_errors()
dt6 = pd.DataFrame(dcc,columns=['dictionary_check'])
dt7= dt6[~dt6.dictionary_check.str.contains("www")]
dt8= h5.merge(dt7.drop_duplicates(subset=['dictionary_check']), how='left', indicator='_merge')
dt9=dt8.replace({'_merge':{'both': 1,'left_only':0}})#both=1= คือสะกดผิด >> left_only = 0=สะกดถูก
dt9.rename(columns ={'_merge':'random_dictionary_check'},inplace=True)

# Dictionary_check1
dt9['dictionary_check1']=dt9['subdomain1']
dt9['dictionary_check1'] = dt9['dictionary_check1'].str.replace('\d+', '')
dt9['dictionary_check1'] = dt9['dictionary_check1'].str.strip()
dt9[['dictionary_check1']] = dt9[['dictionary_check1']].fillna('N')
sf = spell_checker(dt9,'dictionary_check1')
sf.spell_check()
sff=sf.get_all_errors()
dt10 = pd.DataFrame(sff,columns=['dictionary_check1'])
dt12= dt9.merge(dt10.drop_duplicates(subset=['dictionary_check1']), how='left', indicator='_merge')
dt13=dt12.replace({'_merge':{'both': 1,'left_only':0}})#both=1= คือสะกดผิด >> left_only = 0=สะกดถูก
dt13.rename(columns ={'_merge':'random_dictionary_check1'},inplace=True)

# Dictionary_check2
dt13['dictionary_check2']=dt13['subdomain2']
dt13['dictionary_check2'] = dt13['dictionary_check2'].str.replace('\d+', '')
dt13['dictionary_check2'] = dt13['dictionary_check2'].str.strip()
dt13[['dictionary_check2']] = dt13[['dictionary_check2']].fillna('N')
sff= spell_checker(dt13,'dictionary_check2')
sff.spell_check()
sfff=sff.get_all_errors()
ppd = pd.DataFrame(sfff,columns=['dictionary_check2'])
dt14= dt13.merge(ppd.drop_duplicates(subset=['dictionary_check2']), how='left', indicator='_merge')
dt15=dt14.replace({'_merge':{'both': 1,'left_only':0}})#both=1= คือสะกดผิด >> left_only = 0=สะกดถูก
dt15.rename(columns ={'_merge':'random_dictionary_check2'},inplace=True)

# Checking Not Brand name and random subdomain

# **1 คือ คำที่ไม่ใช่ brand_name word + ไม่ได้มีการ random_word(R_sub-d =0) หรือไม่ได้สะกดผิดด้วย เราจะเอาข้อมูลนี้ไป check ว่า length > 7 หรือไม่**         
# **Yes** คือ เข้า the word decompser model   
# **No** เป็น feature ใหม่ ชื่อว่า leange>7

# subdomain
dt15['not_brand_random']=dt15['random_sub_d']
for i in range(len(dt15["brand_name"])): # เป็นการสร้างคอลัมน์ใหม่ที่เราจะเอาข้อมูลที่ไม่ใช่ brandname และ random word ไปเข้า decompose model 
    if dt15["brand_name"][i] == 0 and dt15["random_sub_d"][i] == 0: #ไม่ใช่ brand_name word และ random_sub-d = 0 (ไม่ใช่การสะกดผิด) กำหนดให้เป็น 1 
        print("1")
        dt15["not_brand_random"][i] = "1" #not brand and not random 
    else :
        print("0")
        dt15["not_brand_random"][i] = "0"
dt15['sub_length>7'] = dt15['subdomain_all'] 
for i in range(len(dt15["subdomain_all"])): 
    if dt15["subdomain_all"][i] >7 and dt15["not_brand_random"][i] == 1:  
        print("1")
        dt15["sub_length>7"][i] = "1"
    else :
        print("0")
        dt15["sub_length>7"][i] = "0"

#  subdomain1
dt15['not_brand_random1']=dt15['random_sub1_d']
for i in range(len(dt15["brand_name"])): # เป็นการสร้างคอลัมน์ใหม่ที่เราจะเอาข้อมูลที่ไม่ใช่ brandname และ random word ไปเข้า decompose model 
    if dt15["brand_name"][i] == 0 and dt15["random_sub1_d"][i] == 0: #ไม่ใช่ brand_name word และ random_sub-d = 0 (ไม่ใช่การสะกดผิด) กำหนดให้เป็น 1 
        print("1")
        dt15["not_brand_random1"][i] = "1" #not brand and not random 
    else :
        print("0")
        dt15["not_brand_random1"][i] = "0"
dt15['sub1_length>7'] = dt15['subdomain1_all'] 
for i in range(len(dt15["subdomain1_all"])): 
    if dt15["subdomain1_all"][i] >7 and dt15["not_brand_random1"][i] == 1:  
        print("1")
        dt15["sub1_length>7"][i] = "1"
    else :
        print("0")
        dt15["sub1_length>7"][i] = "0"

# subdomain2
dt15['not_brand_random2']=dt15['random_sub2_d']
for i in range(len(dt15["brand_name"])): # เป็นการสร้างคอลัมน์ใหม่ที่เราจะเอาข้อมูลที่ไม่ใช่ brandname และ random word ไปเข้า decompose model 
    if dt15["brand_name"][i] == 0 and dt15["random_sub2_d"][i] == 0: #ไม่ใช่ brand_name word และ random_sub-d = 0 (ไม่ใช่การสะกดผิด) กำหนดให้เป็น 1 
        print("1")
        dt15["not_brand_random2"][i] = "1" #not brand and not random 
    else :
        print("0")
        dt15["not_brand_random2"][i] = "0"
dt15['sub2_length>7'] = dt15['subdomain2_all'] 
for i in range(len(dt15["subdomain2_all"])): 
    if dt15["subdomain2_all"][i] >7 and dt15["not_brand_random2"][i] == 1:  
        print("1")
        dt15["sub2_length>7"][i] = "1"
    else :
        print("0")
        dt15["sub2_length>7"][i] = "0"

# ### เมื่อทำการ ลบ digit ออกแล้ว และได้ทำการ spell_checker อีกรอบ บางคำก็ยังมีการเขียนผิดอยู่ ต่อมา เราต้องการที่จะเช็คว่า มันเขียนผิดแบบตั้งใจเช่น xyz หรือพยายามสะกดผิด เช่น whatappp ซึ่งเราจะทำการหารคำที่สะกดใกล้เคียงที่สุดที่ถูกต้อง โดยการใช้ autocorrect จาหนั้นก็จะนำไปเข้า MAM Model ต่อ 
dtt =dt15.loc[dt15['not_brand_random']==1] #ข้อสังเกตคือ จริงๆเราต้องตัดคำที่เป็น brand name ย่อยออกไปอีก เช่น มันอาจจะไม่ match >> www.google.co.th ตอนที่เราทำ brand_name  เราสามารถนำไปทำตอน้ายได้ โดยการ match Brand name แค่ตรงคำว่า google


# ## subdomain_d
dtt[['subdomain_d']] = dtt[['subdomain_d']].fillna('No')

#  Autocorrect 
dtt1 = pd.DataFrame(dtt, columns=["subdomain_d"])
dtt1["correct"] = [' '.join([spell(i) for i in x.split()]) for x in dtt1["subdomain_d"]]
dtt1['Levenshtein']=dtt1.apply(lambda x: levenshtein.distance(x['subdomain_d'],  x['correct']), axis=1)
dt16 = dt15.merge(dtt1.drop_duplicates(subset=['subdomain_d']), how='left', indicator='_merge')
dt17=dt16.replace({'_merge':{'both': 1,'left_only':0}})#both= 1 คือ มีการ suggest คำที่ถูกต้องออกมาให้ หรืออาจจะออกคำเดิมมาเลยก็ได้ แต่ว่ายังอยู่ภายใต้เงื่อนไขของ not brand and random
dt17.rename(columns ={'_merge':'Autocorrect'},inplace=True)

#  subdomain1_d
dt18=dt17.loc[dt17['not_brand_random1']==1]
dt18[['subdomain1_d']] = dt18[['subdomain1_d']].fillna('No') 
dt19 = pd.DataFrame(dt18, columns=["subdomain1_d"])
dt19["correct1"] = [' '.join([spell(i) for i in x.split()]) for x in dt19["subdomain1_d"]]
dt19['Levenshtein_1']=dt19.apply(lambda x: levenshtein.distance(x['subdomain1_d'],  x['correct1']), axis=1)
dt20 = dt17.merge(dt19.drop_duplicates(subset=['subdomain1_d']), how='left', indicator='_merge')
dt21=dt20.replace({'_merge':{'both': 1,'left_only':0}})#both= 1 คือ มีการ suggest คำที่ถูกต้องออกมาให้ หรืออาจจะออกคำเดิมมาเลยก็ได้ แต่ว่ายังอยู่ภายใต้เงื่อนไขของ not brand and random
dt21.rename(columns ={'_merge':'Autocorrect_1'},inplace=True)

# subdomain2
dt22=dt21.loc[dt21['not_brand_random2']==1]
dt22[['subdomain2_d']] = dt22[['subdomain2_d']].fillna('No')
dt23 = pd.DataFrame(dt22, columns=["subdomain2_d"])
dt23["correct2"] = [' '.join([spell(i) for i in x.split()]) for x in dt23["subdomain2_d"]]
dt23['Levenshtein_2']=dt23.apply(lambda x: levenshtein.distance(x['subdomain2_d'],  x['correct2']), axis=1)
dt24 = dt21.merge(dt23.drop_duplicates(subset=['subdomain2_d']), how='left', indicator='_merge')
dt25=dt24.replace({'_merge':{'both': 1,'left_only':0}})#both= 1 คือ มีการ suggest คำที่ถูกต้องออกมาให้ หรืออาจจะออกคำเดิมมาเลยก็ได้ แต่ว่ายังอยู่ภายใต้เงื่อนไขของ not brand and random
dt25.rename(columns ={'_merge':'Autocorrect_2'},inplace=True)

#  Final_feature
# **เนื่องจากข้อมูลมีการซ๊ำกัน เราต้องสร้างคอลัมน์ใหม่ที่ถูกต้อง ตั้งชื่อว่า autocorrect**
dt25['autocorrect']=dt25['Autocorrect']
for i in range(len(dt25["Autocorrect"])): 
    if dt25["Autocorrect"][i] == 1 and dt25["not_brand_random"][i] == 1: 
        print("1")
        dt25["autocorrect"][i] = "1" #not brand and not random 
    else :
        print("0")
        dt25["autocorrect"][i] = "0"
dt25['autocorrect_1']=dt25['Autocorrect_1']
for i in range(len(dt25["Autocorrect_1"])): 
    if dt25["Autocorrect_1"][i] == 1 and dt25["not_brand_random1"][i] == 1: 
        print("1")
        dt25["autocorrect_1"][i] = "1" #not brand and not random 
    else :
        print("0")
        dt25["autocorrect_1"][i] = "0"
dt25['autocorrect_2']=dt25['Autocorrect_2']
for i in range(len(dt25["Autocorrect_2"])): 
    if dt25["Autocorrect_2"][i] == 1 and dt25["not_brand_random2"][i] == 1: 
        print("1")
        dt25["autocorrect_2"][i] = "1" #not brand and not random 
    else :
        print("0")
        dt25["autocorrect_2"][i] = "0"

#  จัดเรียง feature 
dt26= dt25[['url', 'slash_all', 'TLD', 'subdomain', 'split', 'slash', 'subdomain1',
       'split1', 'subdomain2', 'split2', 'slash_num', 'slash_ch', 'slash_word',
       'slash_all_score', 'slash_w', 'not_dot', 'one', 'two', 'TLD_real',
       'subdomain_num', 'subdomain_word', 'subdomain_all', 'subdomain1_num',
       'subdomain1_word', 'subdomain1_all', 'subdomain2_num',
       'subdomain2_word', 'subdomain2_all', 'split_dot', 'split_count',
       'split1_dot', 'split1_count', 'first_con', 'second_con', 'split_dc',
       'third_con', 'brand_name', 
          
        'subdomain_d', 'correct', 'Levenshtein', 'Autocorrect','autocorrect',
          'random_sub_d',
       'subdomain1_d', 'correct1','Levenshtein_1', 'Autocorrect_1', 'autocorrect_1',
          'random_sub1_d',
          'subdomain2_d', 'correct2', 'Levenshtein_2',
       'Autocorrect_2', 'autocorrect_2',
          'random_sub2_d',
       'dictionary_check', 'random_dictionary_check', 'dictionary_check1',
       'random_dictionary_check1', 'dictionary_check2',
       'random_dictionary_check2', 'not_brand_random', 'sub_length>7',
       'not_brand_random1', 'sub1_length>7', 'not_brand_random2',
       'sub2_length>7']]

# Model
# test =dt26.drop_duplicates(subset='url', keep="first")
# test.reset_index(inplace=True)

df= dt26[['slash_all','url','slash_num','slash_ch', 'slash_word',
       'slash_all_score', 'slash_w',  'subdomain_num', 'subdomain_word', 'subdomain_all', 'subdomain1_num',
       'subdomain1_word', 'subdomain1_all', 'subdomain2_num',
       'subdomain2_word', 'subdomain2_all','split_count', 'split1_count',
        'Levenshtein',   'Levenshtein_1', 'Levenshtein_2',

        'not_dot', 'one', 'two', 'TLD_real','split_dot', 
       'split1_dot', 'first_con', 'second_con', 'split_dc',
       'third_con', 'brand_name','Autocorrect', 'autocorrect', 'random_sub_d',
        'Autocorrect_1','autocorrect_1', 'random_sub1_d',
        'Autocorrect_2', 'autocorrect_2', 'random_sub2_d',
         'random_dictionary_check','random_dictionary_check1','random_dictionary_check2',
         'not_brand_random','not_brand_random1','not_brand_random2',
        'sub_length>7','sub1_length>7','sub2_length>7']]

le = LabelEncoder()
df['Split_dc']= le.fit_transform(df['split_dc']) #หลักการของมันคือ เรียงตามตัวอักษรในที่นี้คือ bad=0, good=1 
df2=df.drop(['split_dc'],axis=1)
df3=df2.fillna(0)
#xx=df3.drop(['Label'],axis=1)
x_test = df3.drop(['url','slash_all'],axis=1)
#y_test= df3['Label']
x_test['third_con'] = x_test['third_con'].astype(float)

pd.set_option("max_rows", None)

# Catboost_Result
'''loaded_model = pickle.load(open('Catboost_model.sav', 'rb'))
#result = loaded_model.score(x_test, y_test) #score ใช้ในกรณีมีตัวแปร y เป็นการดูค่า accuracy 
y_pred =loaded_model.predict(x_test)
dt11 = pd.DataFrame(y_pred,columns=['y_pred'])
Prob = loaded_model.predict_proba(x_test)
dt10 = pd.DataFrame(Prob,columns=['Prob_0','Prob_1'])
dg=pd.concat([df3,dt10,dt11], axis='columns')
dff =dg[[ 'url','slash_all','y_pred','Prob_0', 'Prob_1']]'''

'''# เอาเฉพาะ  Probability >= 80% ที่น่าจะเป็น bad 
dfff = dff.loc[dff['Prob_0']>=0.8] 
#print(dfff)
#print('Accuracy:{:.4f}'.format(result))'''

# # Logistic_Result
loaded_model1 = pickle.load(open('Logistic_model.sav', 'rb'))
#result1 = loaded_model1.score(x_test, y_test) #score ใช้ในกรณีมีตัวแปร y เป็นการดูค่า accuracy 
y_pred1 =loaded_model1.predict(x_test)
dt12 = pd.DataFrame(y_pred1,columns=['y_pred'])
Prob1 = loaded_model1.predict_proba(x_test)
dt13= pd.DataFrame(Prob1,columns=['Prob_0','Prob_1'])
dgg=pd.concat([df3,dt12,dt13], axis='columns')
dff1 =dgg[[ 'url','slash_all','y_pred','Prob_0', 'Prob_1']]

# เอาเฉพาะ  Probability >= 80% ที่น่าจะเป็น bad 
dfff1 = dff1.loc[dff1['Prob_0']>=0.999]
#print(dfff1)
#print('Accuracy:{:.4f}'.format(result1))

# Lgbm
'''loaded_model2 = pickle.load(open('Lgbm_model.sav', 'rb'))
#result2 = loaded_model2.score(x_test, y_test) #score ใช้ในกรณีมีตัวแปร y เป็นการดูค่า accuracy 
y_pred2 =loaded_model.predict(x_test)
dt14 = pd.DataFrame(y_pred2,columns=['y_pred'])
Prob2 = loaded_model.predict_proba(x_test)
dt15= pd.DataFrame(Prob2,columns=['Prob_0','Prob_1'])
dgg1=pd.concat([df3,dt14,dt15], axis='columns')
dff2 =dgg1[[ 'url','slash_all','y_pred','Prob_0', 'Prob_1']]

# เอาเฉพาะ  Probability >= 80% ที่น่าจะเป็น bad 
dfff2 = dff2.loc[dff2['Prob_0']>=0.8] 
#print(dfff2)'''
#print('Accuracy:{:.4f}'.format(result2))
dfff200 = dfff1[['slash_all']]
#print(dfff200)
listurl = dfff200.values.tolist()
listurl2 = []
finallisturl = []
ips = re.compile('^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
for f in listurl:
    for q in f:
        listurl2.append(q)
for s in listurl2:
    if not ips.match(s):
        finallisturl.append(s)
listdict = []
listresult = []
check = ["not Malicious Domain","We expected a valid IP address or Domain name.","Your request has been resolved to ","probably a private IP","not Malicious IP"]
'''for dictt in finallisturl:
    for neko in dictt:
        listdict.append(neko)'''
listurl3 = list(dict.fromkeys(finallisturl))
for web in listurl3:
    re = webscraping(web,'catagory','action')
    if re not in check:
        listresult.append(re)
print(listresult)
