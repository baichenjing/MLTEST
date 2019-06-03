import pandas as pd
def csv_func():
    import pandas as pd
    path='../titanic/input/train.csv'
    chipo=pd.read_csv(path,sep='\t')
    chipo.tail()
    chipo.info()
    chipo.columns
    dollarizer = lambda x: float(x[1:-1])
    chipo['item_price'] = chipo['item_price'].apply(dollarizer)
    chipo['sub_total']=round(chipo['item_price']*chipo['quantity'],2)
    chipo['sub_total'].sum()
    chipo['itme_name'].nuique()
if __name__ == '__main__':
    csv_func()