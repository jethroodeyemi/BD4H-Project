import os
import shutil
import subprocess
import pandas as pd

def process_encounter(encnum):
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

    dev_df = pd.read_csv("dev.dat", sep='\t', header=None)
    patient_dev_df = dev_df[dev_df[0] == encnum]
    patient_dev_df.to_csv(os.path.join(data_dir, "train.dat"), sep='\t', index=False, header=False)

    generated_file = os.path.join(data_dir, "generated_all.dat")
    predict_command = [
        "python", "predict_ts.py",
        "--data", data_dir,
        "--checkpoint", "models/model_all_mixed.pt",
        "--cuda",
        "--outf", generated_file
    ]
    subprocess.run(predict_command, check=True, capture_output=True, text=True)

    test_path = os.path.join(data_dir, "test.dat")
    test_df = pd.read_csv(test_path, sep='\t', header=None)
    generated_df = pd.read_csv(generated_file, sep='\t', header=None)
    
    final_df = pd.concat([test_df.iloc[:, :4], generated_df], axis=1)
    final_df.to_csv(os.path.join(data_dir, "label_pred_all.dat"), sep='\t', index=False, header=False)

def inference_full_model():
    with open("test_ids.dat") as f:
        content = f.read()
        enc_nums = [int(num) for num in content.split() if num.strip()]

    for encnum in enc_nums:
        process_encounter(encnum)

if __name__ == "__main__":
    inference_full_model()
