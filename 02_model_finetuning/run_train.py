import os

if __name__ == '__main__':
    # recommended environment: one Tesla A100 with 80GB
    for community in range(20):
        model_name = 'gpt2'
        mp = 0  # mp=0 indicates the baseline; mp=1 indicates our methods

        save_total_limit = 1
        per_device_train_batch_size = 160
        per_device_eval_batch_size = 50
        fp16 = True

        n_workers = 8
        gpu = '0'
        epochs = {1: 5, 0: 10}[mp]  # the corpus for message passing is twice as big as that of the baseline, so 1/2 training epochs

        if mp == 0:
            train_file_path = f'../data/community-texts/tweets_community_{community}.txt'
            output_path = f'models/{model_name}_community_{community}'
        else:
            train_file_path = f'../data/community-texts/tweets_community_mp_{community}.txt'
            output_path = f'models/{model_name}_community_mp_{community}'

        os.system(f'export CUDA_VISIBLE_DEVICES={gpu}')
        command = f'python run_clm.py ' \
                  f'--num_train_epochs={epochs} ' \
                  f'--logging_strategy=epoch ' \
                  f'--save_strategy=epoch ' \
                  f'--save_total_limit={save_total_limit} ' \
                  f'--evaluation_strategy=epoch ' \
                  f'--logging_first_step ' \
                  f'--block_size=128 ' \
                  f'--model_name_or_path={model_name} ' \
                  f'--tokenizer_name={model_name} ' \
                  f'--train_file={train_file_path} ' \
                  f'--validation_split_percentage=2 ' \
                  f'--per_device_train_batch_size={per_device_train_batch_size} ' \
                  f'--per_device_eval_batch_size={per_device_eval_batch_size} ' \
                  f'--do_train ' \
                  f'--do_eval ' \
                  f'--preprocessing_num_workers={n_workers} ' \
                  f'--dataloader_num_workers={n_workers} ' \
                  f'--cache_dir=/scratch2/zihaoh/cache ' \
                  f'--output_dir={output_path} ' \
                  f'--fp16={fp16} ' \

        print(command)
        os.system(f'{command}')