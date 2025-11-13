# new_ensemble_data.py

# Random seeds for each of the 4 cross-validation splits for reproducibility
seeds = [1111, 2222, 3333, 4444]

# Data from Table 3 of the paper for the "full model"
# The output range of the model (severity score 0-4)
full_ranges = [4.0, 4.0, 4.0, 4.0] 
# Number of training patients for each split
full_sizes = [638, 638, 637, 637] 

# Data from Table 3 of the paper for the "ensemble model"
# The output range of the model (severity score 0-4)
ensemble_ranges = [4.0, 4.0, 4.0, 4.0]
# Number of models in the final ensemble for each split
ensemble_sizes = [37, 20, 65, 40]