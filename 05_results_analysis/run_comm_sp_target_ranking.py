import os
import pandas as pd
import pickle
import numpy as np
from scipy.stats import spearmanr, kendalltau


if __name__ == '__main__':
    np.random.seed(2022)
    os.makedirs('results', exist_ok=True)

    comm2political = pickle.load(open('../data/comm2political.pkl', 'rb'))
    df_pred = pd.read_csv('../04_eval_stances/df_group_sent_scores.csv')

    df_survey_results = pd.read_csv('../data/anes2020_pilot_prompt_probing.csv')
    pids = df_survey_results['pid'].tolist()

    print('Community-specific target ranking....')
    models = ['gpt2', 'gpt2-ft', 'gpt2-ft-mp']
    for model in models:
        df = df_pred.query(f"model=='{model}'")
        print(f'**********{model}**********')

        rows = []
        for seed in [0, 1, 2, 3, 4]:
            for prompt_type in [1, 2, 3, 4]:
                for community in range(10):
                    df_ = df.query(f"seed=={seed} and prompt_type=={prompt_type} and community=={community}")

                    dem_ratio, repub_ratio = comm2political[community][0], comm2political[community][1]
                    df_weighted_scores = dem_ratio * df_survey_results['Democrat'] \
                                         + repub_ratio * df_survey_results['Republican']

                    pred_scores = df_['score'].values
                    gt_scores = df_weighted_scores.values
                    spearman_corr, _ = spearmanr(pred_scores, gt_scores)
                    kendall, _ = kendalltau(pred_scores, gt_scores)

                    rows.append([seed, prompt_type, community, spearman_corr, kendall])

        df_within_comm = pd.DataFrame(rows, columns=['seed', 'prompt', 'community', 'spearman_corr', 'kendall'])
        df_within_comm.to_csv(f'results/df_within_comm_{model}.csv', index=False)

        for prompt in [1, 2, 3, 4]:
            avg_spearmans = []
            avg_kendalls = []
            for run in [0, 1, 2, 3, 4]:
                df_within_comm_ = df_within_comm.query(f"seed=={run} and prompt=={prompt}")
                avg_spearman = df_within_comm_['spearman_corr'].mean()
                avg_kendall = df_within_comm_['kendall'].mean()
                avg_spearmans.append(avg_spearman)
                avg_kendalls.append(avg_kendall)
            avg_spearmans = np.array(avg_spearmans)
            avg_kendalls = np.array(avg_kendalls)
            print(prompt,
                  f'Spearman Ranking: {avg_spearmans.mean():.3f}+-{avg_spearmans.std():.3f}',
                  f'Kendall Tau: {avg_kendalls.mean():.3f}+-{avg_kendalls.std():.3f}'
                  )
        print()