import pandas as pd
from multiprocessing import Pool


if __name__ == '__main__':
    n_chunks = 20
    dfs = []
    for i in range(n_chunks):
        df = pd.read_csv(f'inferred_chunks/df_sent_unique_{i}.csv')
        print(i, df.shape[0])
        dfs.append(df)
    df_stances_unique = pd.concat(dfs)
    print(df_stances_unique.shape[0])
    df_responses_all = pd.read_csv('../03_text_inference/df_responses_all.csv')

    response_id2score = dict(zip(df_stances_unique['response_id'].tolist(),
                                 df_stances_unique['label'].tolist()))
    df_responses_all['score'] = df_responses_all['response_id'].map(response_id2score)

    print(df_responses_all.shape[0])
    df_responses_all = df_responses_all.dropna(subset=['score'])
    print(df_responses_all.shape[0])

    df_responses_all.to_csv('df_stances_all.csv', index=False)

    seeds = df_responses_all['seed'].unique().tolist()
    pids = df_responses_all['pid'].unique().tolist()
    prompt_types = df_responses_all['prompt_type'].unique().tolist()
    communities = df_responses_all['community'].unique().tolist()
    models = df_responses_all['model'].unique().tolist()


    def filter_df(seed, pid, prompt_type, community, model):
        query = f"seed == {seed} and pid == '{pid}' and prompt_type == {prompt_type} " \
                f"and community == {community} and model == '{model}'"
        df = df_responses_all.query(query)
        avg_score = df['score'].mean()
        row = [model, seed, pid, prompt_type, community, avg_score]
        return row

    args = []
    for seed in seeds:
        for pid in pids:
            for prompt_type in prompt_types:
                for community in communities:
                    for model in models:
                        print(seed, pid, prompt_type, community, model)
                        args.append((seed, pid, prompt_type, community, model))
    print(f'{len(args)} combinations.')

    with Pool(processes=32) as pool:
        rows = pool.starmap(filter_df, args)

    df_group_scores = pd.DataFrame(rows, columns=['model', 'seed', 'pid', 'prompt_type', 'community', 'score'])
    df_group_scores.to_csv('df_group_sent_scores.csv', index=False)