import os
import pandas as pd

def get_patient_ids(input_file, output_file):
    df = pd.read_csv(input_file, sep='\t', header=None, dtype=str)
    patient_ids = df.iloc[:, 0]
    id_counts = patient_ids.value_counts().reset_index()
    id_counts.columns = ['ID', 'Count']
    id_counts.to_csv(output_file, sep='\t', header=False, index=False)

def generate_id_files(df, prefix, split_num):
    ids_ns = df[df.iloc[:, 1] == '0'].iloc[:, 0].unique().astype(str)
    ids_ss = df[df.iloc[:, 1] == '1'].iloc[:, 0].unique().astype(str)

    list_spc_ns = ' '.join(ids_ns)
    list_spc_ss = ' '.join(ids_ss)
    list_comma_ns = ','.join(ids_ns)
    list_comma_ss = ','.join(ids_ss)

    dat_content = f"{list_spc_ns}\n{list_spc_ss}"
    with open(f'./{prefix}_ids_s{split_num}.dat', 'w') as f:
        f.write(dat_content)
    with open(f'./{prefix}_ids.dat', 'w') as f:
        f.write(dat_content)
    
    all_ids_list = list(ids_ns) + list(ids_ss)
    with open(f'./{prefix}_ids.lst', 'w') as f:
        f.write('\n'.join(all_ids_list))

    inc_content = f"{list_comma_ns}\n{list_comma_ss}"
    with open(f'./{prefix}_ids_s{split_num}.inc', 'w') as f:
        f.write(inc_content)
    with open(f'./{prefix}_ids.inc', 'w') as f:
        f.write(inc_content)

    py_content = (f"valid_ns = {{{list_comma_ns if list_comma_ns else ''}}}\n"
                  f"valid_ss = {{{list_comma_ss if list_comma_ss else ''}}}")
    with open(f'./{prefix}_ids.py', 'w') as f:
        f.write(py_content)

def make_data_split():
    partitions = {
        'A': 'sepsisexp_timeseries_partition-A.tsv',
        'B': 'sepsisexp_timeseries_partition-B.tsv',
        'C': 'sepsisexp_timeseries_partition-C.tsv',
        'D': 'sepsisexp_timeseries_partition-D.tsv'
    }
        
    split_config = {
        0: {'train': ['A', 'B'], 'dev': 'C', 'test': 'D'},
        1: {'train': ['B', 'C'], 'dev': 'D', 'test': 'A'},
        2: {'train': ['C', 'D'], 'dev': 'A', 'test': 'B'},
        3: {'train': ['D', 'A'], 'dev': 'B', 'test': 'C'}
    }
    split_num = 0  # Default to 0
    config = split_config[split_num]

    train_dfs = [pd.read_csv(partitions[p], sep='\t', header=None, dtype=str) for p in config['train']]
    train_df = pd.concat(train_dfs)
    train_df.sort_values(by=train_df.columns[0], inplace=True)
    train_df.to_csv('train.dat', sep='\t', index=False, header=False)

    dev_df = pd.read_csv(partitions[config['dev']], sep='\t', header=None, dtype=str)
    dev_df.to_csv('dev.dat', sep='\t', index=False, header=False)

    test_df = pd.read_csv(partitions[config['test']], sep='\t', header=None, dtype=str)
    test_df.to_csv('test.dat', sep='\t', index=False, header=False)

    get_patient_ids('train.dat', 'sorted_train_ids.dat')
    get_patient_ids('dev.dat', 'sorted_dev_ids.dat')
    get_patient_ids('test.dat', 'sorted_test_ids.dat')

    generate_id_files(dev_df, 'dev', split_num)
    generate_id_files(test_df, 'test', split_num)

    os.makedirs('data/data_empty', exist_ok=True)
    
    train_dat_path = os.path.abspath('train.dat')
    dev_symlink_path = 'data/data_empty/dev.dat'
    test_symlink_path = 'data/data_empty/test.dat'

    if os.path.lexists(dev_symlink_path):
        os.remove(dev_symlink_path)
    os.symlink(train_dat_path, dev_symlink_path)

    if os.path.lexists(test_symlink_path):
        os.remove(test_symlink_path)
    os.symlink(train_dat_path, test_symlink_path)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

if __name__ == "__main__":    
    make_data_split()