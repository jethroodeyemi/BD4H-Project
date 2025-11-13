import os, sys
import numpy as np
import gzip, pickle
import os.path

from inc_trainset import validarr as pool_ids
from dev_ids import valid_ns, valid_ss
dev_ids = valid_ns.union(valid_ss)

def readPrediction(model, candidate):
    timeline = []
    generated = "data/data_"+str(model)+"/generated_"+str(candidate)+".dat"
    with open(generated, 'r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            timeline.append(float(values[4]))
    return timeline

def readGroundTruth(model):
    timeline = []
    generated = "data/data_"+str(model)+"/label_pred.dat"
    with open(generated, 'r') as f:
        while (line := f.readline().rstrip('\n')):
            values = line.split('\t')
            timeline.append(float(values[2]))
    return timeline

def readSepticNonSeptic(valid_ids):
    septic = set()
    nonseptic = set()
    with open('all_mixed.dat', 'r') as f:
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
    return septic, nonseptic



def readPredictionsAll(pool, devids):
    dev_preds_all = {}
    for candidate in pool:
        predictions = []
        for model in devids:
            predictions += readPrediction(model, candidate)
        dev_preds_all[candidate] = predictions
    return dev_preds_all

def readGroundTruthAll(devids):
    ground_truth = []
    for model in devids:
        ground_truth += readGroundTruth(model)
    return ground_truth



def meanSquaredError(labels, predictions):
    mse = (np.square(np.array(labels) - np.array(predictions))).mean(axis=None)
    return mse

def calcEnsemblePredsAll(dev_predictions, ensemble):
    predictions = []
    for model in ensemble:
        predictions.append(np.array(dev_predictions[model[0]]))
    return np.mean(predictions, axis=0)

def calcCovariance(ground_truth, model1, model2):
    covsum = np.dot(np.subtract(model1, ground_truth),
                    np.subtract(model2, ground_truth))
    return covsum / len(ground_truth)



def selectModelPerrone(models_sorted, used_eval, dev_preds, ground_truth, ensemble, ensemble_mse):
    topk_models = []
    for model in models_sorted:
        if model[0] not in used_eval:
            topk_models.append(model)

    ens_size = len(ensemble)
    new_model = None
    
    for model in topk_models:
        model_preds = dev_preds[model[0]]
        model_mse = meanSquaredError(ground_truth, model_preds)
        correlation = 0
        for ensmodel in ensemble:
            ens_model_preds = dev_preds[ensmodel[0]]
            correlation += calcCovariance(ground_truth, model_preds, ens_model_preds)

        perrone_lhs = (2*ens_size + 1) * ensemble_mse
        perrone_rhs = 2 * correlation + model_mse

        if perrone_lhs > perrone_rhs:
            new_model = model
            break
    return new_model



def existsTimelinesFile(fname):
    return os.path.isfile(fname)

def loadAllTimelines(fname):
    fp = gzip.open(fname, 'rb')
    ss_ids, ns_ids, dev_predictions, ground_truth = pickle.load(fp)
    fp.close()
    return ss_ids, ns_ids, dev_predictions, ground_truth

def saveAllTimelines(fname, ss_ids, ns_ids, dev_predictions, ground_truth):
    fp = gzip.open(fname, 'wb')
    pickle.dump([ss_ids, ns_ids, dev_predictions, ground_truth], fp, 1)
    fp.close()



def main():
    splitnum = int(sys.argv[1])

    septic = []
    nonseptic = []
    allpatients = []
    ensemble = []

    eval_models = set(pool_ids[splitnum])
    pred_models = set(dev_ids)

    used_eval = set()
    used_pred = set()

    if existsTimelinesFile('timelines.pkl.gz'):
        ss_ids, ns_ids, dev_predictions, ground_truth = loadAllTimelines('timelines.pkl.gz')
    else:
        ss_ids, ns_ids = readSepticNonSeptic(eval_models)
        dev_predictions = readPredictionsAll(eval_models, dev_ids)
        ground_truth = readGroundTruthAll(dev_ids)
        saveAllTimelines('timelines.pkl.gz', ss_ids, ns_ids, dev_predictions, ground_truth)


    for encnum in ss_ids:
        mse = meanSquaredError(ground_truth, dev_predictions[encnum])
        septic.append((encnum, mse))
        allpatients.append((encnum, mse))

    for encnum in ns_ids:
        mse = meanSquaredError(ground_truth, dev_predictions[encnum])
        nonseptic.append((encnum, mse))
        allpatients.append((encnum, mse))

    septic_sorted = sorted(septic, key=lambda tup: tup[1])
    nonseptic_sorted = sorted(nonseptic, key=lambda tup: tup[1])
    allpatients_sorted = sorted(allpatients, key=lambda tup: tup[1])

    ss = septic_sorted[0]
    ensemble.append(ss)
    used_eval.add(ss[0])
    used_pred.add(ss[1])

    cnt = 1
    num_septic = 1
    num_nonseptic = 0
    last = "septic"
    errors = {}
    ensemble_size = 99

    while cnt < ensemble_size:
        ensemble_preds = calcEnsemblePredsAll(dev_predictions, ensemble)
        ensemble_mse = meanSquaredError(ground_truth, ensemble_preds)

        print(cnt, ensemble_mse, last)
        errors[cnt] = ensemble_mse

        ensemble_model = selectModelPerrone(septic_sorted, used_eval,
                                            dev_predictions, ground_truth,
                                            ensemble, ensemble_mse)
        if ensemble_model != None:
            last = "septic"
            num_septic += 1
            
        if ensemble_model == None:
            ensemble_model = selectModelPerrone(nonseptic_sorted, used_eval,
                                                dev_predictions, ground_truth,
                                                ensemble, ensemble_mse)
            if ensemble_model != None:
                last = "non-septic"
                num_nonseptic += 1

        if ensemble_model == None:
            break

        ensemble.append(ensemble_model)
        used_eval.add(ensemble_model[0])
        cnt += 1

    print("ns =", num_nonseptic, ", ss =", num_septic)
    print(ensemble)

    sys.stdout.write("-----------\n")
    for model in ensemble:
        sys.stdout.write("%s\n" % model[0])

    sys.stdout.write("-----------\n")
    sys.stdout.write("ensemble = {")
    for model in ensemble:
        sys.stdout.write("%s, " % model[0])
    sys.stdout.write('}\n')

    print(sorted(errors.items(), key=lambda item: item[1]))



if __name__ == "__main__":
    main()

