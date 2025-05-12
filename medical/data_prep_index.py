import random
import argparse
from os import makedirs, path
from datasets import load_dataset, concatenate_datasets

def subsample_data(all_data, split_ratio=0.5):
    random.seed(100)
    total_ids = all_data['new_ID']
    subsampled_ids = random.sample(total_ids, int(len(total_ids) * split_ratio))
    return subsampled_ids
    
#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # Parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/medsum_llm_attack')
    parser.add_argument('--data_dir', type=str, default='formatted_data/')
    args = parser.parse_args()
    if not path.exists(args.data_dir):
        makedirs(args.data_dir)
        
    # ----------------------
    # Load & Process Data
    # ----------------------
    print('Downloading and preparing data...')    
    dataset = load_dataset(args.dataset)
    all_data = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    new_ids = range(len(all_data))
    all_data = all_data.add_column("new_ID", new_ids)
    subsample_ids = subsample_data(all_data)

    print('Spliting data...')  
    ## train
    subsample_dataset = all_data.filter(lambda example: example['new_ID'] in subsample_ids)
    dataset_train_val = subsample_dataset.train_test_split(test_size=0.2, seed=100)
    dataset_train = dataset_train_val['train']
    dataset_val = dataset_train_val['test']

    ## shadow_train
    shadow_dataset = all_data.filter(lambda example: example['new_ID'] not in subsample_ids)
    shadow_dataset_train_val = shadow_dataset.train_test_split(test_size=0.2, seed=100)
    shadow_dataset_train = shadow_dataset_train_val['train']
    shadow_dataset_val = shadow_dataset_train_val['test']

    print('Saving Indices...')  
    with open(path.join(args.data_dir, 'train_ids.txt'), 'w') as f:
        for row in dataset_train['new_ID']:
            f.write(str(row) + '\n')
    with open(path.join(args.data_dir, 'val_ids.txt'), 'w') as f:
        for row in dataset_val['new_ID']:
            f.write(str(row) + '\n')
    with open(path.join(args.data_dir, 'shadow_train_ids.txt'), 'w') as f:
        for row in shadow_dataset_train['new_ID']:
            f.write(str(row) + '\n')
    with open(path.join(args.data_dir, 'shadow_val_ids.txt'), 'w') as f:
        for row in shadow_dataset_val['new_ID']:
            f.write(str(row) + '\n')

if __name__ == "__main__":
    main()