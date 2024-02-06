import os


if __name__ == '__main__':
    n_chunks = 20
    n_workers = 4

    for i in range(n_chunks):
        data_path = f'chunks/df_unique_{i}.csv'
        commands = []

        command = f'python -u infer_stance.py --data_path={data_path}'
        commands.append(command)
        print(command)

        os.system(command)