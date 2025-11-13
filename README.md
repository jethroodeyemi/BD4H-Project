Installation:
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=1.19.4" "scipy>=1.6.0" "pandas"

Download and extract dataset
wget https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/SepsisExp.tar.gz
tar -zxvf SepsisExp.tar.gz

Prepare Data Splits: python 1_make_data.py

Model Training
Train full model: python main_ts.py --data . --save models/model_all_mixed.pt --cuda --epochs 200 2>&1 | tee logs/training_all_mixed.log
Train Patient-Specific Models: python 2_make_models_perpat.py

Ensemble Creation
Create all_mixed.dat: cat train.dat dev.dat test.dat > all_mixed.dat
Generate Predictions on Dev Set: python inference_poolmodels.py
Run the Ensemble Growing Algorithm: python grow_ensemble_perrone.py 0 | tee logs/grow_ensemble.log
Create Ensemble List: tail -n1 logs/grow_ensemble.log > new_ensemble.py && sed "s/ /\n/g" new_ensemble.py | sed 's/[^0-9]*//g' | sed -r '/^\s*$/d' > new_ensemble.lst

