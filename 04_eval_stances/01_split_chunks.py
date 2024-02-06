import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    n_chunks = 20
    os.makedirs('chunks')
    df = pd.read_csv('../03_text_inference/df_responses_unique.csv')

    # split the responses into different chunks to speed up the sentiment inferencing process
    chunks = np.array_split(df, n_chunks)
    for i in range(n_chunks):
        print(i, chunks[i].shape)
        chunks[i].to_csv(f'chunks/df_unique_{i}.csv', index=False)