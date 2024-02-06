import os
import pandas as pd


if __name__ == '__main__':
    os.makedirs('sampled_output', exist_ok=True)
    model_base_names = [
        'gpt2_{}'
        'gpt2_community_{}',
        'gpt2_community_mp_{}',
    ]

    def infer_prompt_type(prompt):
        if 'is the' in prompt:
            prompt_type = 4
        elif 'is a' in prompt:
            prompt_type = 3
        elif 'is' in prompt:
            prompt_type = 2
        else:
            prompt_type = 1
        return prompt_type

    for each in model_base_names:
        for i in range(20):
            file_path = f'output/{each.format(i)}.csv'
            df = pd.read_csv(file_path)
            seeds = df['seed'].unique().tolist()
            pids = df['pid'].unique().tolist()
            prompt_types = [1, 2, 3, 4]

            dfs = []
            for seed in seeds:
                for pid in pids:
                    for prompt_type in prompt_types:
                        print(f'model: {each}, community: {i}, seed: {seed}, pid: {pid}, prompt_type: {prompt_type}')
                        df1 = df.query(f"seed=={seed} and pid=='{pid}'")
                        prompts = df1['prompt'].unique().tolist()
                        for prompt in prompts:
                            df2 = df1[df1['prompt'] == prompt]
                            df2['response_len'] = df2['response'].apply(lambda x: len(x.split()))
                            df2 = df2.sort_values(by='response_len', ascending=False)
                            n = df2.shape[0]
                            df2 = df2.iloc[:int(n*0.85)]    # sample the top 85% longest responses
                            dfs.append(df2[['seed', 'pid', 'prompt', 'response']])
            df = pd.concat(dfs)
            df.to_csv(f'sampled_output/{each.format(i)}.csv', index=False)
