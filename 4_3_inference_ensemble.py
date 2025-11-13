import os
import shutil
import subprocess
import pandas as pd

def process_ensemble_encounter(encnum):
    data_dir = f"data/data_{encnum}"
    os.makedirs(data_dir, exist_ok=True)
    
    for item in os.listdir("data/data_empty"):
        source_item = os.path.join("data/data_empty", item)
        dest_item = os.path.join(data_dir, item)
        if os.path.isdir(source_item):
            if os.path.exists(dest_item):
                shutil.rmtree(dest_item)
            shutil.copytree(source_item, dest_item, symlinks=True)
        else:
            shutil.copy2(source_item, dest_item)

    test_df = pd.read_csv("test.dat", sep='\t', header=None)
    patient_test_df = test_df[test_df[0] == encnum]
    patient_test_df.to_csv(os.path.join(data_dir, "train.dat"), sep='\t', index=False, header=False)

    output_file = os.path.join(data_dir, "ensemble_test.dat")
    command = ["python", "ensemble_predictions.py", str(encnum)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    with open(output_file, "w") as f:
        f.write(result.stdout)

def inference_ensemble():
    with open("test_ids.dat") as f:
        content = f.read()
        enc_nums = [int(num) for num in content.split() if num.strip()]

    for encnum in enc_nums:
        process_ensemble_encounter(encnum)

if __name__ == "__main__":
    inference_ensemble()
