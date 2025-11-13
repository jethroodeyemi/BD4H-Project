import os
import shutil
import subprocess
import pandas as pd

def process_ensemble_inference(encnum, model):
    data_dir = f"data/data_{encnum}"
    generated_file = os.path.join(data_dir, f"generated_{model}.dat")

    if os.path.exists(generated_file):
        return

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

    all_mixed_df = pd.read_csv("all_mixed.dat", sep='\t', header=None)
    patient_data_df = all_mixed_df[all_mixed_df[0] == encnum]
    patient_data_df.to_csv(os.path.join(data_dir, "train.dat"), sep='\t', index=False, header=False)

    predict_command = [
        "python", "predict_ts.py",
        "--cuda",
        "--data", data_dir,
        "--checkpoint", f"models/model_{model}.pt",
        "--outf", generated_file
    ]
    subprocess.run(predict_command, check=True, capture_output=True, text=True)

    test_path = os.path.join(data_dir, "test.dat")
    test_df = pd.read_csv(test_path, sep='\t', header=None)
    generated_df = pd.read_csv(generated_file, sep='\t', header=None)
    
    final_df = pd.concat([test_df.iloc[:, :4], generated_df], axis=1)
    final_df.to_csv(os.path.join(data_dir, "label_pred.dat"), sep='\t', index=False, header=False)

def inference_ensemble_models():
    with open("test_ids.dat") as f:
        content = f.read()
        enc_nums = [int(num) for num in content.split() if num.strip()]
    
    with open("new_ensemble.lst") as f:
        models = [line.strip() for line in f if line.strip()]

    tasks = [(enc, model) for enc in enc_nums for model in models]

    for enc, model in tasks:
        process_ensemble_inference(enc, model)

if __name__ == "__main__":
    inference_ensemble_models()
