import os
import socket
import pickle


if __name__ == '__main__':
    # recommended environment: one RTX2080Ti with 16GB
    model_base_names = [
        'gpt2_{}'       # pretrained gpt-2
        'gpt2_community_{}',        # finetuned gpt-2
        'gpt2_community_mp_{}',     # finetuned gpt-2 + MP
    ]
    n_workers = 1
    mem = '4GB'
    n_runs = 5

    for i in range(0, 20):
        commands = []
        for model_base_name in model_base_names:
            if 'community' in model_base_name:
                model_name = model_base_name.format(i)
                model_path = f'../02_model_finetuning/models/{model_name}'
                preceding_prompt = ''
                n_responses = 1000
            else:
                model_path = model_base_name.format(i)
                comm2political = pickle.load(open('../data/comm2political.pkl', 'rb'))
                dem_percent, rep_percent = comm2political[i][0], comm2political[i][1]
                preceding_prompt = f'As an independent who agrees with Democrats {dem_percent * 100:.0f}% of the time and Republicans {rep_percent * 100:.0f}% of the time,'
                n_responses = 500

            command = f'python -u generator.py ' \
                      f'--model_path={model_path} ' \
                      f'--n_runs={n_runs} ' \
                      f'--preceding_prompt="{preceding_prompt}" ' \
                      f'--n_responses={n_responses} '
            commands.append(command)
            print(command)

        all_commands = '\n'.join(commands)
        for command in all_commands:
            os.system(command)