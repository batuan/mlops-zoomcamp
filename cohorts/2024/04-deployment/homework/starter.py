import pickle
import pandas as pd
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run_sciprt(year, month):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')


    # In[22]:


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)


    # In[23]:


    print(f"std: {y_pred.std()}")
    print(f"mean: {y_pred.mean()}")


    # In[24]:


    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    # In[25]:


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred

    output_file = 'output'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__=='__main__':
    print(sys.argv)
    year=int(sys.argv[1])
    month=int(sys.argv[2])
    run_sciprt(year=year, month=month)