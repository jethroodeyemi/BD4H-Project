import os
import shutil
import subprocess
import pandas as pd

def run_inference_for_pair(encnum, model_id):
    data_dir = f"data/data_{encnum}"
    generated_file = os.path.join(data_dir, f"generated_{model_id}.dat")
    
    if os.path.exists(generated_file):
        return
        
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.listdir(data_dir):
        for item in os.listdir("data/data_empty"):
            s = os.path.join("data/data_empty", item)
            d = os.path.join(data_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks=True, ignore=None)
            else:
                shutil.copy2(s, d)

    all_mixed_df = pd.read_csv("all_mixed.dat", sep='\t', header=None)
    patient_data_df = all_mixed_df[all_mixed_df[0] == encnum]
    patient_data_df.to_csv(os.path.join(data_dir, "train.dat"), sep='\t', index=False, header=False)

    predict_command = [
        "python", "predict_ts.py",
        "--cuda",
        "--data", data_dir,
        "--checkpoint", f"models/model_{model_id}.pt",
        "--outf", generated_file
    ]
    subprocess.run(predict_command, check=True, capture_output=True, text=True)

    dev_path = os.path.join(data_dir, "dev.dat")
    dev_df = pd.read_csv(dev_path, sep='\t', header=None)
    generated_df = pd.read_csv(generated_file, sep='\t', header=None)
    
    label_pred_df = pd.concat([dev_df.iloc[:, :4], generated_df], axis=1)
    label_pred_df.to_csv(os.path.join(data_dir, "label_pred.dat"), sep='\t', index=False, header=False)

def inference_pool_models():
    with open("dev_ids.lst") as f:
        enc_nums = [int(line.strip()) for line in f if line.strip()]
    
    model_ids_df = pd.read_csv("sorted_train_ids.dat", sep='\t', header=None)
    model_ids = model_ids_df[0].tolist()

    tasks = [(enc, model) for enc in enc_nums for model in model_ids]
    
    for enc, model in tasks:
        run_inference_for_pair(enc, model)

if __name__ == "__main__":
    inference_pool_models()
