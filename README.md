Installation:
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=1.19.4" "scipy>=1.6.0" "pandas"

Download and extract dataset
wget https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/SepsisExp.tar.gz
tar -zxvf SepsisExp.tar.gz

Prepare Data Splits: python 1_make_data.py