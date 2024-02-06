import pandas as pd


if __name__ == '__main__':
    df_prompts = pd.read_csv('../data/anes2020_pilot_prompt_probing.csv')
    pids = df_prompts['pid']
    targets = df_prompts['Prompt1']
    pid2target = dict(zip(pids, targets))

    model_base_names = [
        ('gpt2_{}', 'gpt2'),
        ('gpt2_community_{}', 'gpt2-ft'),
        ('gpt2_community_mp_{}', 'gpt2-ft-mp')
    ]

    def infer_prompt_type(prompt):
        if 'is the' in prompt or 'are the' in prompt:
            prompt_type = 4
        elif 'is a' in prompt or 'are a' in prompt:
            prompt_type = 3
        else:
            prompt_tokens = prompt.split()
            if 'is' in prompt_tokens or 'are' in prompt_tokens:
                prompt_type = 2
            else:
                prompt_type = 1

        return prompt_type

    dfs = []
    for each in model_base_names:
        for i in range(20):
            print(f'model: {each[1]}, community: {i}')
            file_path = f'sampled_output/{each[0].format(i)}.csv'
            df = pd.read_csv(file_path)
            df['prompt_type'] = df['prompt'].apply(infer_prompt_type)
            df['target'] = df['pid'].map(pid2target)
            df['community'] = i
            df['model'] = each[1]
            dfs.append(df)

    df_all = pd.concat(dfs)
    df_unique = df_all.drop_duplicates(subset=['response'])

    print(df_all.shape[0])
    print(df_unique.shape[0])

    print('Creating dict...')
    unique_responses = df_unique['response'].tolist()
    response2id = dict(zip(unique_responses, range(len(unique_responses))))

    print('Mapping....')
    df_all['response_id'] = df_all['response'].map(response2id)
    df_unique['response_id'] = df_unique['response'].map(response2id)

    print('Saving df_responses_all.csv .....')
    df_all.to_csv('df_responses_all.csv', index=False)
    print('Saving df_responses_unique.csv .....')
    df_unique.to_csv('df_responses_unique.csv', index=False)

