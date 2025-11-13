# BD4H-Project

Ensembling Neural Networks for Improved Prediction and Privacy in Early Diagnosis of Sepsis reproduction.

## Installation

Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=1.19.4" "scipy>=1.6.0" "pandas"
```

## Dataset Setup

Download and extract the SepsisExp dataset:
```bash
wget https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/SepsisExp.tar.gz
tar -zxvf SepsisExp.tar.gz
```

## Usage

### 1. Prepare Data Splits
```bash
python 1_make_data.py
```

### 2. Model Training

Train the full model:
```bash
python main_ts.py --data . --save models/model_all_mixed.pt --cuda --epochs 200 2>&1 | tee logs/training_all_mixed.log
```

Train patient-specific models:
```bash
python 2_make_models_perpat.py
```

### 3. Ensemble Creation

Create combined dataset:
```bash
cat train.dat dev.dat test.dat > all_mixed.dat
```

Generate predictions on dev set:
```bash
python 3_inference_poolmodels.py
```

Run the ensemble growing algorithm:
```bash
python grow_ensemble_perrone.py 0 | tee logs/grow_ensemble.log
```

Create ensemble list:
```bash
tail -n1 logs/grow_ensemble.log > new_ensemble.py && sed "s/ /\n/g" new_ensemble.py | sed 's/[^0-9]*//g' | sed -r '/^\s*$/d' > new_ensemble.lst
```

### 4. Generate Test Set Predictions

Generate full model predictions:
```bash
python 4_1_inference_fullmodel.py
```

Generate predictions for each ensemble model:
```bash
python 4_2_inference_ensmodels.py
```

Combine ensemble predictions:
```bash
python 4_3_inference_ensemble.py
```

### 5. Evaluation

**Hypothesis 1: Prediction Accuracy (Table 4)**
```bash
python evaluation/calc_auroc.py
```

**Hypothesis 2: Privacy (Figure 2)**

Attack on full model:
```bash
python evaluation/membership_fullmodel_epsilon_1k.py 0 > full_model_leakage.txt
```

Attack on ensemble model:
```bash
python evaluation/membership_ensemble_epsilon_alltrain_1k.py 0 > ensemble_leakage.txt
```

Generate privacy leakage plot:
```bash
python evaluation/plot_leakage.py
```

**Accuracy/Privacy Trade-off (Figure 3)**
```bash
python evaluation/calc_auroc_laplace_all.py 0 > accuracy_loss_data.txt
python evaluation/plot_accuracy_loss.py
```