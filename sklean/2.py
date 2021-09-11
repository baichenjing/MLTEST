import numpy as np
from tqdm import tqdm

pred_dict={}
count=0
scount=0
for page,row in tqdm(zip(train.index,train)):
    last_month=np.array(row[-preriod:])
    slast_month=np.array(row[-2*period:-PERIOD])
    prev_last_month=np.array(row[:period])
    use_last_year=False
    if ~np.isnan(row[0]):
        year_increase=np.median(slast_month)/np.median(prev_slast_month)
        year_error=np.sum(list(map(lambda x:smape(x[0],x[1]),zip(last_month,prev_last_month*year_increase))))
    smedian=np.median(slast_month)
    month_error=np.sum(list(map(lambda x:smape(x,smedian),last_month)))
    error_diff=(month_error-year_error)/PERIOD
    if error_diff>0.1:
        scount+=1
        use_last_year=True
    if use_last_year:
        last_year=np.array(row[2*PERIOD:2*PERIOD+PREDICT_PERIOD])
        preds=last_year*year_increase
    else:
        preds=[0]*PREDICT_PERIOD
        windows=np.array([2,3,4,7,11,18,29,47])*7
        medians=np.zeros(len(windows),7)
        for i in range(7):
            for k in range(len(windows)):
                array=np.array(row[-windows[k]:]).reshape(-1,7)
                s=np.hstack(array[:,(i-1)%7],array[:,i],array[:,(i+1)%7]).reshape(-1)
                medians[k,i]=safe_median(s)
            for i in range(PREDICT_PERIOD):
                preds[i]=safe_median(medians[:,i%7])
            pred_dict[page]=preds
            count+=1
            if count%1000==0:
                print(count,scount)
            del train
print("yearly prediction is done on the percentage:{}".format(scount/count))