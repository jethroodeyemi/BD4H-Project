import os,sys
import numpy as np
import gzip, pickle
import os.path
from inc_trainset import validarr as alltrain_ids
from inc_devset import validarr as alldev_ids
from inc_testset import validarr as alltest_ids
from new_ensemble_data import seeds,full_ranges,full_sizes


def readPrediction(model, candidate):
    timeline = []
    generated = "data/data_"+str(model)+"/generated_"+str(candidate)+".dat"
    with open(generated,'r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            timeline.append(float(values[4]))
    return timeline

def readGroundTruth(model):
    timeline = []
    generated = "data/data_"+str(model)+"/label_pred.dat"
    with open(generated,'r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            timeline.append(float(values[2]))
    return timeline

def readSepticNonSeptic(valid_ids):
    septic = set()
    nonseptic = set()
    with open('all_mixed.dat','r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            encnum = int(values[0])
            if encnum not in valid_ids:
                continue
            if encnum not in nonseptic and encnum not in septic:
                if values[1] == '0':
                    nonseptic.add(encnum)
                else:
                    septic.add(encnum)
    return septic,nonseptic



def readPredictionsAll(model_ids,eval_ids):
    dev_preds_all = {}
    for candidate in model_ids:
        predictions = []
        for model in eval_ids:
            predictions += readPrediction(model,candidate)
        dev_preds_all[candidate] = predictions
    return dev_preds_all

def readPredictionslModelsAll(model_ids):
    predictions_model = {}
    predictions_all = []
    for model in model_ids:
        predictions_model[model] = readPrediction(model,'all')
        predictions_all += predictions_model[model]
    return predictions_model,predictions_all






def readGroundTruthModelsAll(model_ids):
    groundtruth_model = {}
    groundtruth_all = []
    for model in model_ids:
        groundtruth_model[model] = readGroundTruth(model)
        groundtruth_all += groundtruth_model[model]
    return groundtruth_model,groundtruth_all

def meanSquaredError(labels,predictions):
    assert(len(labels) == len(predictions))
    mse = (np.square(np.array(labels) - np.array(predictions))).mean(axis=None)
    return mse

def calcEnsemblePredsAll(all_predictions,ensemble_ids):
    predictions = []
    for model in ensemble_ids:
        predictions.append(np.array(all_predictions[model]))
    return np.mean(predictions,axis=0)

def check_membership(model_ids,predictions,groundtruth,training_mse):
    success = 0
    for model in model_ids:
        model_preds = predictions[model]
        model_groundtruth = groundtruth[model]
        model_mse = meanSquaredError(model_groundtruth,model_preds)
        if model_mse < training_mse:
            success += 1
    return success

def check_membership_laplace(model_ids,predictions,groundtruth,
                             training_mse,scale):
    success = 0
    for model in model_ids:
        preds = np.array(predictions[model])
        noise = np.random.laplace(0.0, scale, len(preds))
        model_preds = np.add(preds,noise)
        model_groundtruth = groundtruth[model]
        model_mse = meanSquaredError(model_groundtruth,model_preds)
        if model_mse < training_mse:
            success += 1
    return success





def existsTimelinesFile(fname):
    return os.path.isfile(fname)

def loadAllTimelines(fname):
    fp = gzip.open(fname,'rb')
    ss_ids,ns_ids,train_preds_all,train_groundtruth_all, \
        train_preds,train_groundtruth, \
            test_preds,test_groundtruth = pickle.load(fp)
    fp.close()
    return ss_ids,ns_ids,train_preds_all, train_groundtruth_all, \
        train_preds,train_groundtruth,\
        test_preds,test_groundtruth

def saveAllTimelines(fname,ss_ids,ns_ids,
                     train_preds_all,train_groundtruth_all,
                     train_preds,train_groundtruth,
                     test_preds,test_groundtruth):
    fp=gzip.open(fname,'wb')
    pickle.dump([ss_ids,ns_ids,train_preds_all,train_groundtruth_all,
                 train_preds,train_groundtruth,
                 test_preds,test_groundtruth
                 ],fp,1)
    fp.close()








def main():
    splitnum = int(sys.argv[1])
    trainset_ids = set(alltrain_ids[splitnum])
    devset_ids = set(alldev_ids[splitnum])
    testset_ids = set(alltest_ids[splitnum])
    splitseed = seeds[splitnum]
    np.random.RandomState(np.random.MT19937(np.random.SeedSequence(splitseed)))
    np.random.seed(splitseed)
    splitrange = full_ranges[splitnum]
    splitsize = full_sizes[splitnum]
    data_filename = "memberdata_fullmodel.pkl.gz"
    if existsTimelinesFile(data_filename):
        ss_ids,ns_ids,train_preds_all,train_groundtruth_all, \
            train_preds,train_groundtruth, \
                test_preds,test_groundtruth = loadAllTimelines(data_filename)
    else:
        ss_ids,ns_ids = readSepticNonSeptic(trainset_ids.union(testset_ids))
        train_preds,train_preds_all = readPredictionslModelsAll(trainset_ids)
        train_groundtruth,train_groundtruth_all = readGroundTruthModelsAll(trainset_ids)
        test_preds,_ = readPredictionslModelsAll(testset_ids)
        test_groundtruth,_ = readGroundTruthModelsAll(testset_ids)
        saveAllTimelines(data_filename,
                         ss_ids,ns_ids,
                         train_preds_all,train_groundtruth_all,
                         train_preds,train_groundtruth,
                         test_preds,test_groundtruth)
    trainset_mse = meanSquaredError(train_groundtruth_all,train_preds_all)
    rounds = 1000
    print("#"*50)
    print("# Full Model Training Loss (MSE):",trainset_mse)
    print("# Ensemble Min. Prediction:",min(train_preds_all))
    print("# Ensemble Max. Prediction:",max(train_preds_all))
    print("#"*50)
    print("# Membership Attack:")
    print("# pos:", len(testset_ids), "sampled",rounds,"times from" ,len(trainset_ids))
    print("# neg:", len(testset_ids))
    print("#"*50)
    length = len(testset_ids)
    epsilons = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]
    for epsilon in epsilons:
        scale = ((splitrange)*(1.0/splitsize))/epsilon
        falsepos = check_membership_laplace(testset_ids,test_preds,
                                        test_groundtruth,trainset_mse,scale)
        trueneg = length - falsepos
        if falsepos == 0:
            falsepos = 1e-9
        recalls = []
        precisions = []
        trueposs = []
        trainlist_ids = list(trainset_ids)
        for i in range(rounds):
            np.random.shuffle(trainlist_ids)
            truepos = check_membership_laplace(trainlist_ids[0:length],train_preds,
                                           train_groundtruth,trainset_mse,scale)
            recalls.append(truepos / length)
            precisions.append(truepos / (truepos+falsepos))
            trueposs.append(truepos)
        tpr = np.mean(trueposs)/length
        fpr = (falsepos/length)
        privacy_leakage = tpr-fpr
        steps = 1
        for pos in range(0,rounds,steps):
            tpr = np.mean(trueposs[pos:pos+steps])/length
            fpr = falsepos/length
            privacy_leakage = tpr-fpr
            print("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f"%(epsilon,scale,np.mean(precisions[pos:pos+steps]),
                    np.mean(trueposs[pos:pos+steps]) / length, (np.mean(trueposs[pos:pos+steps])+trueneg)/(2*length),
                    tpr,fpr,privacy_leakage))

if __name__ == "__main__":
    main()

