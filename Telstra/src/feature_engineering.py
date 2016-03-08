import numpy as np
import pandas as pd
import scipy as sp
import math
import random

#----load the data set ----#
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train['location'] = train['location'].str.replace('location ', '')
test['location'] = test['location'].str.replace('location ', '')

severity = pd.read_csv('../input/severity_type.csv')
log_feature = pd.read_csv("../input/log_feature.csv")
resource = pd.read_csv("../input/resource_type.csv")
event = pd.read_csv("../input/event_type.csv")

severity['severity_type'] = severity['severity_type'].str.replace('severity_type ',"")
log_feature["log_feature"] = log_feature["log_feature"].str.replace("feature ", "")
resource["resource_type"] = resource["resource_type"].str.replace("resource_type ", "")
event["event_type"] = event["event_type"].str.replace("event_type ", "")

test['fault_severity'] = -1
whole = pd.concat([train,test],ignore_index=True)

log_id_order = log_feature.id.unique()
magic=[]
i = 0
while i<log_id_order.shape[0]:
    magic.append([1.0,0,0])
    targets = [0,0,0]
    
    timeid = log_id_order[i]
    locid = whole[whole['id'] == timeid]["location"].values[0]
    tg = int(whole[whole['id'] == timeid]['fault_severity'].values[0])
    j = i+1
    while j<log_id_order.shape[0]:
        timeid = log_id_order[j]
        locid1 = whole[whole['id'] == timeid]['location'].values[0]
        
        if locid1 != locid:
            i = j
            break
        if tg == -1 and sum(targets) == 0:
            magic.append([1.0,0.0,0.0])
        else:
            if tg > -1:
                targets[tg] += 1
            magic.append([targets[0]*1.0/sum(targets), targets[1]*1.0/sum(targets), targets[2]*1.0/sum(targets)])
            
        tg = int(whole[whole['id'] == timeid]["fault_severity"].values[0])
        j = j+1
    if j>=log_id_order.shape[0]:
        break
        
data = pd.merge(whole, severity, on='id',how='left')
magic_fea = []
res = []
eve = []
log = []
for i in range(0, data.shape[0]):
    lineid = data.at[i,"id"]
    ind = list(log_id_order).index(lineid)
    magic_fea.append(magic[ind])
    
    li = [0 for k in range(10)]
    ser = resource[resource["id"] == lineid]["resource_type"]
    for j in range(0, ser.shape[0]):
        li[int(ser.values[j])-1] = 1
    res.append(li)

    li = [0 for k in range(54)]
    ser = event[event['id'] == lineid]["event_type"]
    for j in range(0, ser.shape[0]):
        li[int(ser.values[j])-1] = 1
    eve.append(li)
    
    li = [0 for k in range(386)]
    ser = log_feature[log_feature['id'] == lineid]["log_feature"]
    volume = log_feature[log_feature['id']==lineid]["volume"]
    for j in range(0, ser.shape[0]):
        li[int(ser.values[j])-1] = volume.values[j]
    log.append(li)

data['magic1'] = np.array(magic_fea)[:,0]
data['magic2'] = np.array(magic_fea)[:,1]
data['magic3'] = np.array(magic_fea)[:,2]

for i in range(10):
    data["resource_type"+str(i)] = np.array(res)[:, i]

for i in range(54):
    data["event_type"+str(i)] = np.array(eve)[:, i]

for i in range(386):
    data['log_feature'+str(i)] = np.array(log)[:, i]
    
data['location'] = map(int, list(data['location'].values))
data['severity_type'] = map(int, list(data['severity_type'].values))

data.to_csv('../input/dataset.csv',index=False)