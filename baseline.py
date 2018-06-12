import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt


#Time Handle Function
def timeInterval(str1, str2):
    str1 = str(str1).zfill(8)
    str2 = str(str2).zfill(8)
    day1 = int(str1[0:2])
    hour1 = int(str1[2:4])
    minute1 = int(str1[4:6])
    second1 = int(str1[6:8]) # start time
    day2 = int(str2[0:2])
    hour2 =  int(str2[2:4])
    minute2 = int(str2[4:6])
    second2 = int(str2[6:8]) # end time
    if (day2 > day1):
        hour2 += (day2 - day1) * 24
    interval = (hour2  - hour1) * 3600 + (minute2 - minute1) * 60 + (second2 - second1)
    return interval



uid_train = pd.read_csv('../data/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../data/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('../data/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../data/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

voice_test = pd.read_csv('../data/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('../data/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('../data/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../data/uid_test_b.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)



voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()

voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()

voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)

voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice['voice_time'] = voice.apply(lambda row: timeInterval(row['start_time'], row['end_time']), axis=1)

voice_time_sum_by_ID = voice.groupby(['uid'])['voice_time'].agg(['std','max','min','median','mean','sum']).add_prefix('vocie_all_time_').reset_index().fillna(0)

voice_time_sum_by_ID_type = voice.groupby(['uid', 'call_type'])['voice_time'].sum().unstack('call_type').add_prefix('voice_time_call_type_').reset_index().fillna(0)

voice_time_sum_by_ID_IO = voice.groupby(['uid', 'in_out'])['voice_time'].sum().unstack('in_out').add_prefix('voice_time_in_out_').reset_index().fillna(0)

voice_time_count_by_ID = voice.groupby(['uid'])['uid'].agg({'count':'count'}).add_prefix('voice_all_').reset_index().fillna(0)

sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()

sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()

sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)

sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)

mes_count_by_ID = sms.groupby(['uid'])['uid'].agg({'count': 'count'}).add_prefix('sms_all_').reset_index().fillna(0)





wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()

visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()

visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()


up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()

down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()


wa_type_cnt_by_ID = wa.groupby(['uid', 'wa_type'])['visit_cnt'].sum().unstack('wa_type').add_prefix('wa_visit_cnt_type_').reset_index().fillna(0)

wa_type_dura_by_ID = wa.groupby(['uid', 'wa_type'])['visit_dura'].sum().unstack('wa_type').add_prefix('wa_visit_dura_type_').reset_index().fillna(0)

wa_type_up_by_ID = wa.groupby(['uid', 'wa_type'])['up_flow'].sum().unstack('wa_type').add_prefix('wa_visit_up_flow_type_').reset_index().fillna(0)

wa_type_down_by_ID = wa.groupby(['uid', 'wa_type'])['down_flow'].sum().unstack('wa_type').add_prefix('wa_visit_down_flow_type_').reset_index().fillna(0)


feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,voice_time_sum_by_ID,
           voice_time_sum_by_ID_type, voice_time_sum_by_ID_IO, voice_time_count_by_ID,
           sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,mes_count_by_ID,
           wa_name,visit_cnt,visit_dura,up_flow,down_flow, wa_type_cnt_by_ID, wa_type_dura_by_ID, wa_type_up_by_ID,
           wa_type_down_by_ID]


train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')

train_feature = train_feature.fillna(0)
test_feature = test_feature.fillna(0)

print(train_feature.info())
print(train_feature.info())

train_feature.to_csv('../data/train_featureV1.csv',index=None)
test_feature.to_csv('../data/test_featureV1.csv',index=None)

