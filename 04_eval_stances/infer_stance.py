import os
import time
import pandas as pd
import torch
from transformers import pipeline, set_seed
import argparse

import sys
sys.path.append('..')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='chunks/df_responses_unique_0.csv')
    args = parser.parse_args()

    os.makedirs('inferred_chunks', exist_ok=True)

    batch_size = 2800
    n_workers = 4
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    seed = 2023
    set_seed(2023)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    df_responses_unique = pd.read_csv(args.data_path)
    n = df_responses_unique.shape[0]
    print(f'{n} texts')
    responses = df_responses_unique['response'].tolist()
    response_ids = df_responses_unique['response_id'].tolist()

    sentiment_pipeline = pipeline('sentiment-analysis',
                                  model=model_id,
                                  tokenizer=model_id,
                                  max_length=64,
                                  truncation=True,
                                  device=device,
                                  batch_size=batch_size,
                                  num_workers=n_workers)
    print('running inference....')
    outputs = sentiment_pipeline(responses)
    print('Done')

    sentiment_dict = {
        "negative": 0,
        "positive": 100,
        "neutral": 50
    }

    labels = [sentiment_dict[output['label']] for output in outputs]

    df = pd.DataFrame({'response_id': response_ids, 'label': labels})
    idx = args.data_path[:-4].split('_')[-1]
    df.to_csv(f'inferred_chunks/df_sent_unique_{idx}.csv', index=False)
