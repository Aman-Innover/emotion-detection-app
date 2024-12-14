import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

import logging

logger=logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
consol_handler=logging.StreamHandler()
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consol_handler.setFormatter(formatter)
logger.addHandler(consol_handler)



test_size=yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size']
data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
df = pd.read_csv(data_url)

df.drop(columns=['tweet_id'], inplace=True)
logger.debug(f"Data loaded from {data_url}")

final_df=df[df['sentiment'].isin(['happiness','sadness'])]
final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})   
train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

data_path = os.path.join('data', 'raw')
logger.info(f"Saving data to {data_path}")

os.makedirs(data_path,exist_ok=True)
train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)


print(df.head())