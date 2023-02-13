from other_cleansers.scan import SCAn
from tqdm import tqdm
import numpy as np
import torch

def get_features(data_loader, model):

    class_indices = []
    preds_list = []
    feats = []

    model.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            preds, x_features = model(ins_data, True)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_features[bid].cpu().numpy())
                class_indices.append(ins_target[bid].cpu().numpy())
                preds_list.append(preds[bid])

    return feats, class_indices, preds_list

def cleanser(inspection_set, clean_set, model, num_classes):

    kwargs = {'num_workers': 4, 'pin_memory': True}

    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    # a small clean batch for defensive purpose
    clean_set_loader = torch.utils.data.DataLoader(
        clean_set,
        batch_size=128, shuffle=True, **kwargs)

    feats_inspection, class_indices_inspection, preds_inspection = get_features(inspection_split_loader, model)
    feats_clean, class_indices_clean, _ = get_features(clean_set_loader, model)

    feats_inspection = np.array(feats_inspection)
    class_indices_inspection = np.array(class_indices_inspection)
    preds_inspection = np.array(preds_inspection)

    feats_clean = np.array(feats_clean)
    class_indices_clean = np.array(class_indices_clean)


    scan = SCAn()

    # fit the clean distribution with the small clean split at hand
    gb_model = scan.build_global_model(feats_clean, class_indices_clean, num_classes)

    size_inspection_set = len(feats_inspection)

    feats_all = np.concatenate([feats_inspection, feats_clean])
    class_indices_all = np.concatenate([class_indices_inspection, class_indices_clean])

    # use the global model to divide samples
    lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, num_classes)

    # statistic test for the existence of "two clusters"
    score = scan.calc_final_score(lc_model)
    threshold = np.exp(2)

    suspicious_indices = []
    num_samples = len(inspection_set)

    for target_class in range(num_classes):
        print('[class-%d] outlier score = %f.' % (target_class, score[target_class]))
        if score[target_class] <= threshold: continue # omit classes that pass the single-cluster test
        for i in range(num_samples):
            if class_indices_inspection[i] == target_class and preds_inspection[i] == target_class:
                suspicious_indices.append(i)


    return suspicious_indices