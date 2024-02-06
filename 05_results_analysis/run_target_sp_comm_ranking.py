import os
import pandas as pd
import pickle
import numpy as np
from scipy.stats import spearmanr, kendalltau


if __name__ == '__main__':
    model = ['gpt2', 'gpt2-ft', 'gpt2-ft-mp'][2]
    np.random.seed(2022)

    comm2political = pickle.load(open('../data/comm2political.pkl', 'rb'))
    df_pred = pd.read_csv('../04_eval_stances/df_group_sent_scores.csv')

    df_survey_results = pd.read_csv('../data/anes2020_pilot_prompt_probing.csv')
    pids = df_survey_results['pid'].tolist()
    os.makedirs('results', exist_ok=True)

    print('Target-specific community ranking....')
    models = ['gpt2', 'gpt2-ft', 'gpt2-ft-mp']
    for model in models:
        df = df_pred.query(f"model=='{model}'")
        print(f'**********{model}**********')

        rows = []
        for seed in [0, 1, 2, 3, 4]:
            for prompt_type in [1, 2, 3, 4]:
                for pid in pids:
                    df_ = df.query(f"seed=={seed} and prompt_type=={prompt_type} and pid=='{pid}'")
                    df_ = df_.sort_values(by=['community'])

                    gt_scores = []
                    df_survery_results_ = df_survey_results.query(f"pid=='{pid}'").iloc[0]
                    dem_score, repub_score = df_survery_results_['Democrat'], df_survery_results_['Republican']
                    for i in range(20):
                        dem_ratio, repub_ratio = comm2political[i][0], comm2political[i][1]
                        gt_score = dem_ratio * dem_score + repub_ratio * repub_score
                        gt_scores.append(gt_score)

                    pred_scores = df_['score'].values
                    gt_scores = np.array(gt_scores)

                    spearman_corr, _ = spearmanr(a=pred_scores, b=gt_scores)
                    kendall, _ = kendalltau(pred_scores, gt_scores)
                    rows.append([seed, prompt_type, pid, spearman_corr, kendall])

        df_cross_comm = pd.DataFrame(rows, columns=['seed', 'prompt', 'question', 'spearman_corr', 'kendall'])
        df_cross_comm.to_csv(f'results/df_cross_comm_{model}.csv', index=False)

        for prompt in [1, 2, 3, 4]:
            avg_spearmans = []
            avg_kendalls = []
            for run in [0, 1, 2, 3, 4]:
                df_cross_comm_ = df_cross_comm.query(f"seed=={run} and prompt=={prompt}")
                avg_spearman = df_cross_comm_['spearman_corr'].mean()
                avg_kendall = df_cross_comm_['kendall'].mean()
                avg_spearmans.append(avg_spearman)
                avg_kendalls.append(avg_kendall)
            avg_spearmans = np.array(avg_spearmans)
            avg_kendalls = np.array(avg_kendalls)
            print(prompt,
                  f'Spearman Ranking: {avg_spearmans.mean():.3f}+-{avg_spearmans.std():.3f}',
                  f'Kendall Tau: {avg_kendalls.mean():.3f}+-{avg_kendalls.std():.3f}')
        print()