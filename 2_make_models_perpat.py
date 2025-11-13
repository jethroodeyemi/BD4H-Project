import os
import shutil
import subprocess
import pandas as pd

def process_patient(patient_id):
    i = patient_id
    data_dir = f"data/data_{i}"
    model_path = f"models/model_{i}.pt"
    log_path = f"logs/training_sepsis_{i}.log"
    
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    shutil.copytree("data/data_empty", data_dir)

    dev_symlink = os.path.join(data_dir, "dev.dat")
    if os.path.lexists(dev_symlink):
        os.remove(dev_symlink)
    os.symlink(os.path.abspath("dev.dat"), dev_symlink)

    train_df = pd.read_csv("train.dat", sep='\t', header=None)
    patient_train_df = train_df[train_df[0] == i]
    patient_train_df.to_csv(os.path.join(data_dir, "train.dat"), sep='\t', index=False, header=False)

    train_command = [
        "python", "main_ts.py",
        "--data", data_dir,
        "--save", model_path,
        "--cuda",
        "--epochs", "20",
        "--min_epochs", "10"
    ]
    with open(log_path, "w") as log_file:
        subprocess.run(train_command, stdout=log_file, stderr=subprocess.STDOUT, check=True, text=True)

    predict_command = [
        "python", "predict_ts.py",
        "--data", data_dir,
        "--checkpoint", model_path,
        "--cuda",
        "--outf", os.path.join(data_dir, "generated.dat")
    ]
    subprocess.run(predict_command, check=True)

    test_path = os.path.join(data_dir, "test.dat")
    generated_path = os.path.join(data_dir, "generated.dat")
    
    test_df = pd.read_csv(test_path, sep='\t', header=None)
    generated_df = pd.read_csv(generated_path, sep='\t', header=None)
    
    label_pred_df = pd.concat([test_df.iloc[:, :4], generated_df], axis=1)
    label_pred_df.to_csv(os.path.join(data_dir, "label_pred.dat"), sep='\t', index=False, header=False)

def make_models_per_patient():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    ids_df = pd.read_csv("sorted_train_ids.dat", sep='\t', header=None)
    patient_ids = ids_df[0].tolist()

    for patient_id in patient_ids:
        process_patient(patient_id)

if __name__ == "__main__":
    make_models_per_patient()
