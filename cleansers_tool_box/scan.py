import numpy as np
import torch
from tqdm import tqdm

EPS = 1e-5


class SCAn:
    def __init__(self):
        pass

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index

    def build_global_model(self, reprs, labels, n_classes):
        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']

        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = np.zeros([L, 3])
        u1 = np.zeros([L, M])
        u2 = np.zeros([L, M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = [k, i_sc, np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if (N == 1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        SeG = np.matmul(Se,G)
        for i in range(N):
            vec = X[i]
            dd = np.matmul(SeG, np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N

def get_features(data_loader, model):

    class_indices = []
    feats = []

    model.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            _, x_features = model(ins_data, True)

            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_features[bid].cpu().numpy())
                class_indices.append(ins_target[bid].cpu().numpy())

    return feats, class_indices




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

    feats_inspection, class_indices_inspection = get_features(inspection_split_loader, model)
    feats_clean, class_indices_clean = get_features(clean_set_loader, model)

    feats_inspection = np.array(feats_inspection)
    class_indices_inspection = np.array(class_indices_inspection)

    feats_clean = np.array(feats_clean)
    class_indices_clean = np.array(class_indices_clean)

    # from sklearn.decomposition import PCA
    # projector = PCA(n_components=128)
    # feats_inspection = projector.fit_transform(feats_inspection)
    # feats_clean = projector.fit_transform(feats_clean)


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
    threshold = np.e

    suspicious_indices = []

    for target_class in range(num_classes):

        print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

        if score[target_class] <= threshold: continue

        tar_label = (class_indices_all == target_class)
        all_label = np.arange(len(class_indices_all))
        tar = all_label[tar_label]

        cluster_0_indices = []
        cluster_1_indices = []

        cluster_0_clean = []
        cluster_1_clean = []

        for index, i in enumerate(lc_model['subg'][target_class]):
            if i == 1:
                if tar[index] > size_inspection_set:
                    cluster_1_clean.append(tar[index])
                else:
                    cluster_1_indices.append(tar[index])
            else:
                if tar[index] > size_inspection_set:
                    cluster_0_clean.append(tar[index])
                else:
                    cluster_0_indices.append(tar[index])


        # decide which cluster is the poison cluster, according to clean samples' distribution
        if len(cluster_0_clean) < len(cluster_1_clean): # if most clean samples are in cluster 1
            suspicious_indices += cluster_0_indices
        else:
            suspicious_indices += cluster_1_indices

    return suspicious_indices