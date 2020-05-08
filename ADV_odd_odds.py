import argparse
import math
import os
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import data_loader
import models
import logging
from typing import OrderedDict
from sklearn.metrics import accuracy_score, roc_auc_score


# modified code from https://github.com/yk/icml19_public
# TODO verify and possibly change default arguments to those from torch_example.py
def collect_statistics(
    x_train,
    y_train,
    latent_and_logits_fn=None,
    nb_classes=None,
    weights=None,
    targeted=False,
    noise_eps=8e-3,
    noise_eps_detect=None,
    num_noise_samples=256,
    batch_size=256,
    clip_min=-1.,
    clip_max=1.,
    p_ratio_cutoff=20.,
    clip_alignments=True,
    #    pgd_train=None,
    #    fit_classifier=False,
    #    just_detect=False,
    cache_alignments_dir=None):
    assert len(x_train) == len(y_train)
    # if pgd_train is not None:
    #     assert len(pgd_train) == len(x_train)
    assert latent_and_logits_fn is not None
    assert nb_classes is not None
    assert weights is not None

    def latent_fn(x):
        return to_np(latent_and_logits_fn(to_th(x))[0])

    def logits_fn(x):
        return latent_and_logits_fn(x)[1]

    def to_th(x, dtype=np.float32):
        x = torch.from_numpy(x.astype(dtype))
        x = x.cuda()
        return x

    def to_np(x):
        return x.detach().cpu().numpy()

    try:
        len(noise_eps)
        if isinstance(noise_eps, str):
            raise TypeError()
    except TypeError:
        noise_eps = [noise_eps]

    if noise_eps_detect is None:
        noise_eps_detect = noise_eps

    try:
        len(noise_eps_detect)
        if isinstance(noise_eps_detect, str):
            raise TypeError()
    except TypeError:
        noise_eps_detect = [noise_eps_detect]

    noise_eps_all = set(noise_eps + noise_eps_detect)

    n_batches = math.ceil(x_train.shape[0] / batch_size)

    if len(y_train.shape) == 2:
        y_train = y_train.argmax(-1)

    loss_fn = torch.nn.CrossEntropyLoss(reduce='sum')

    # loss_fn = loss_fn.cuda()

    def get_noise_samples(x, num_samples, noise_eps, clip=False):
        if isinstance(noise_eps, float):
            kind = 'u'
            eps = noise_eps
        else:
            kind, eps = noise_eps[:1], float(noise_eps[1:])

        if isinstance(x, np.ndarray):
            if kind == 'u':
                noise = np.random.uniform(-1.,
                                          1.,
                                          size=(num_samples, ) + x.shape[1:])
            elif kind == 'n':
                noise = np.random.normal(0.,
                                         1.,
                                         size=(num_samples, ) + x.shape[1:])
            elif kind == 's':
                noise = np.random.uniform(-1.,
                                          1.,
                                          size=(num_samples, ) + x.shape[1:])
                noise = np.sign(noise)
            x_noisy = x + noise * eps
            if clip:
                x_noisy = x_noisy.clip(clip_min, clip_max)
        else:
            if kind == 'u':
                noise = x.new_zeros((num_samples, ) + x.shape[1:]).uniform_(
                    -1., 1.)
            elif kind == 'n':
                noise = x.new_zeros((num_samples, ) + x.shape[1:]).normal_(
                    0., 1.)
            elif kind == 's':
                noise = x.new_zeros((num_samples, ) + x.shape[1:]).uniform_(
                    -1., 1.)
                noise.sign_()
            x_noisy = x + noise * eps
            if clip:
                x_noisy.clamp_(clip_min, clip_max)
        return x_noisy

    # def attack_pgd(x, x_pred, targeted=False):
    #     x_pgd = get_noise_samples(x, x.shape[0], pgd_eps, clip=True)

    #     for _ in range(pgd_iters):
    #         x_th = to_th(x_pgd).requires_grad_(True)
    #         x_grads = to_np(
    #             torch.autograd.grad(
    #                 loss_fn(logits_fn(x_th), to_th(x_pred, np.long)),
    #                 [x_th])[0])

    #         x_pgd += pgd_lr * np.sign(x_grads) * (-2. * (targeted - 1 / 2))
    #         x_pgd = x_pgd.clip(x - pgd_eps, x + pgd_eps)
    #         x_pgd = x_pgd.clip(clip_min, clip_max)
    #         if debug:
    #             break
    #     return x_pgd

    def get_latent_and_pred(x):
        # print(f'get_latent_and_pred enter')
        # print(f'x.shape: {x.shape}')
        latent_and_logits = latent_and_logits_fn(to_th(x))
        # print(f'latent_and_logits[0].size(): {latent_and_logits[0].size()}')
        # print(f'latent_and_logits[0][0]]: {latent_and_logits[0][0]}')
        # print(f'latent_and_logits[1].size(): {latent_and_logits[1].size()}')
        # print(f'latent_and_logits[1][0]]: {latent_and_logits[1][0]}')
        l, p = map(to_np, latent_and_logits)
        # print(f'get_latent_and_pred exit')
        return l, p.argmax(-1)

    x_preds_clean = []
    x_train_pgd = []
    x_preds_pgd = []
    latent_clean = []
    latent_pgd = []

    # TODO accept adv examples as argument
    for b in tqdm(range(n_batches),
                  desc='_not_? creating adversarial samples'):
        x_batch = x_train[b * batch_size:(b + 1) * batch_size]
        lc, pc = get_latent_and_pred(x_batch)
        x_preds_clean.append(pc)
        latent_clean.append(lc)

        # if not just_detect:
        #     if pgd_train is not None:
        #         x_pgd = pgd_train[b * batch_size:(b + 1) * batch_size]
        #     else:
        #         if targeted:
        #             x_pgd = np.stack([
        #                 attack_pgd(
        #                     x_batch, np.ones_like(pc) * i, targeted=True)
        #                 for i in range(nb_classes)
        #             ], 1)
        #         else:
        #             x_pgd = attack_pgd(x_batch, pc, targeted=False)
        #     x_train_pgd.append(x_pgd)

        #     if targeted:
        #         pps, lps = [], []
        #         for i in range(x_pgd.shape[1]):
        #             lp, pp = get_latent_and_pred(x_pgd[:, i])
        #             pps.append(pp)
        #             lps.append(lp)
        #         x_preds_pgd.append(np.stack(pps, 1))
        #         latent_pgd.append(np.stack(lps, 1))
        #     else:
        #         lp, pp = get_latent_and_pred(x_pgd)
        #         x_preds_pgd.append(pp)
        #         latent_pgd.append(lp)

    x_preds_clean, latent_clean = map(np.concatenate,
                                      (x_preds_clean, latent_clean))
    # if not just_detect:
    #     x_train_pgd, x_preds_pgd, latent_pgd = map(
    #         np.concatenate, (x_train_pgd, x_preds_pgd, latent_pgd))

    # valid_idcs = []
    # if not just_detect:
    #     for i, (pc, pp,
    #             y) in enumerate(zip(x_preds_clean, x_preds_pgd, y_train)):
    #         if y == pc and pc != pp:
    #             # if y == pc:
    #             valid_idcs.append(i)
    # else:
    #     valid_idcs = list(range(len(x_preds_clean)))
    valid_idcs = list(range(len(x_preds_clean)))

    logging.info('valid idcs ratio: {}'.format(len(valid_idcs) / len(y_train)))
    if targeted:
        for i, xpp in enumerate(x_preds_pgd.T):
            logging.info('pgd success class {}: {}'.format(
                i, (xpp == i).mean()))

    x_train, y_train, x_preds_clean, latent_clean = (a[valid_idcs]
                                                     for a in (x_train,
                                                               y_train,
                                                               x_preds_clean,
                                                               latent_clean))
    # if not just_detect:
    #     x_train_pgd, x_preds_pgd, latent_pgd = (a[valid_idcs]
    #                                             for a in (x_train_pgd,
    #                                                       x_preds_pgd,
    #                                                       latent_pgd))

    weights_np = weights.detach().cpu().numpy()
    big_memory = weights.shape[0] > 20  # TODO figure this out??
    logging.info('BIG MEMORY: {}'.format(big_memory))
    if not big_memory:
        wdiffs = weights[None, :, :] - weights[:, None, :]
        wdiffs_np = weights_np[None, :, :] - weights_np[:, None, :]

    def _compute_neps_alignments(x, lat, pred, idx_wo_pc, neps):
        x, lat = map(to_th, (x, lat))
        if big_memory:
            wdiffs_relevant = weights[pred, None] - weights
        else:
            wdiffs_relevant = wdiffs[:, pred]
        x_noisy = get_noise_samples(x[None],
                                    num_noise_samples,
                                    noise_eps=neps,
                                    clip=clip_alignments)
        lat_noisy, _ = latent_and_logits_fn(x_noisy)
        lat_diffs = lat[None] - lat_noisy
        # print(f'lat_diffs.size(): {lat_diffs.size()}')
        # print(f'wdiffs_relevant.size(): {wdiffs_relevant.size()}')
        return to_np(torch.matmul(lat_diffs,
                                  wdiffs_relevant.transpose(1, 0)))[:,
                                                                    idx_wo_pc]

    def _compute_alignments(x,
                            lat,
                            pred,
                            source=None,
                            noise_eps=noise_eps_all):
        if source is None:
            # print(f'pred: {pred}')
            idx_wo_pc = [i for i in range(nb_classes) if i != pred]
            assert len(idx_wo_pc) == nb_classes - 1
        else:
            idx_wo_pc = source

        alignments = OrderedDict()
        for neps in noise_eps:
            alignments[neps] = _compute_neps_alignments(
                x, lat, pred, idx_wo_pc, neps)
        return alignments, idx_wo_pc

    def _collect_wdiff_stats(x_set,
                             latent_set,
                             x_preds_set,
                             clean,
                             save_alignments_dir=None,
                             load_alignments_dir=None):
        if clean:
            wdiff_stats = {(tc, tc, e): []
                           for tc in range(nb_classes) for e in noise_eps_all}
            name = 'clean'
        else:
            wdiff_stats = {(sc, tc, e): []
                           for sc in range(nb_classes)
                           for tc in range(nb_classes) for e in noise_eps_all
                           if sc != tc}
            name = 'adv'

        def _compute_stats_from_values(v, raw=False):
            if not v.shape:
                return None
            v = v.mean(1)
            # if clean or not fit_classifier:
            if clean:
                if v.shape[0] < 3:
                    return None
                return v.mean(0), v.std(0)
            else:
                return v

        for neps in noise_eps_all:
            neps_keys = {k for k in wdiff_stats.keys() if k[-1] == neps}
            loading = load_alignments_dir
            if loading:
                for k in neps_keys:
                    fn = 'alignments_{}_{}.npy'.format(name, str(k))
                    load_fn = os.path.join(load_alignments_dir, fn)
                    if not os.path.exists(load_fn):
                        loading = False
                        break
                    v = np.load(load_fn)
                    wdiff_stats[k] = _compute_stats_from_values(v)
                logging.info('loading alignments from {} for {}'.format(
                    load_alignments_dir, neps))
            if not loading:
                for x, lc, pc, pcc in tqdm(
                        zip(x_set, latent_set, x_preds_set, x_preds_clean),
                        total=len(x_set),
                        desc='collecting stats for {}'.format(neps)):
                    if len(lc.shape) == 2:
                        alignments = []
                        for i, (xi, lci, pci) in enumerate(zip(x, lc, pc)):
                            if i == pcc:
                                continue
                            alignments_i, _ = _compute_alignments(
                                xi, lci, i, source=pcc, noise_eps=[neps])
                            for e, a in alignments_i.items():
                                wdiff_stats[(pcc, i, e)].append(a)
                    else:
                        alignments, idx_wo_pc = _compute_alignments(
                            x, lc, pc, noise_eps=[neps])
                        for e, a in alignments.items():
                            wdiff_stats[(pcc, pc, e)].append(a)

                saving = save_alignments_dir and not loading
                if saving:
                    logging.info('saving alignments to {} for {}'.format(
                        save_alignments_dir, neps))

                for k in neps_keys:
                    wdsk = wdiff_stats[k]
                    if len(wdsk):
                        wdiff_stats[k] = np.stack(wdsk)
                    else:
                        wdiff_stats[k] = np.array(None)
                    if saving:
                        fn = 'alignments_{}_{}.npy'.format(name, str(k))
                        save_fn = os.path.join(save_alignments_dir, fn)
                        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                        wds = wdiff_stats[k]
                        np.save(save_fn, wds)
                    wdiff_stats[k] = _compute_stats_from_values(wdiff_stats[k])
        return wdiff_stats

    save_alignments_dir_clean = os.path.join(
        cache_alignments_dir, 'clean') if cache_alignments_dir else None
    save_alignments_dir_pgd = os.path.join(
        cache_alignments_dir, 'pgd') if cache_alignments_dir else None
    load_alignments_dir_clean = os.path.join(
        cache_alignments_dir, 'clean') if cache_alignments_dir else None
    load_alignments_dir_pgd = os.path.join(
        cache_alignments_dir, 'pgd') if cache_alignments_dir else None
    if cache_alignments_dir:
        load_alignments_dir_clean, load_alignments_dir_pgd = map(
            lambda s: '{}_{}'.format(s, 'clip'
                                     if clip_alignments else 'noclip'),
            (load_alignments_dir_clean, load_alignments_dir_pgd))
    if cache_alignments_dir:
        save_alignments_dir_clean, save_alignments_dir_pgd = map(
            lambda s: '{}_{}'.format(s, 'clip'
                                     if clip_alignments else 'noclip'),
            (save_alignments_dir_clean, save_alignments_dir_pgd))
    wdiff_stats_clean = _collect_wdiff_stats(
        x_train,
        latent_clean,
        x_preds_clean,
        clean=True,
        save_alignments_dir=save_alignments_dir_clean,
        load_alignments_dir=load_alignments_dir_clean)
    # if not just_detect:
    #     wdiff_stats_pgd = _collect_wdiff_stats(
    #         x_train_pgd,
    #         latent_pgd,
    #         x_preds_pgd,
    #         clean=False,
    #         save_alignments_dir=save_alignments_dir_pgd,
    #         load_alignments_dir=load_alignments_dir_pgd)

    # ??? this is always False...
    # if debug_dict is not None and False:
    #     esizes = OrderedDict((k, []) for k in noise_eps_all)
    #     for k, (mc, sc) in wdiff_stats_clean.items():
    #         mp, sp = wdiff_stats_pgd[k]
    #         esizes[k[-1]].append(np.abs(mp - mc) / ((sp + sc) / 2.))
    #     debug_dict['effect_sizes'] = OrderedDict(
    #         (k, np.array(v)) for k, v in esizes.items())

    wdiff_stats_clean_detect = [
        np.stack([wdiff_stats_clean[(p, p, eps)] for eps in noise_eps_detect])
        for p in range(nb_classes)
    ]
    wdiff_stats_clean_detect = [
        s.transpose((1, 0, 2)) if len(s.shape) == 3 else None
        for s in wdiff_stats_clean_detect
    ]
    # wdiff_stats_pgd_classify = []
    # if not just_detect:
    #     for tc in range(nb_classes):
    #         tc_stats = []
    #         for sc in range(nb_classes):
    #             if sc == tc:
    #                 continue
    #             sc_stats = [
    #                 wdiff_stats_pgd[(sc, tc, eps)] for eps in noise_eps
    #             ]
    #             if sc_stats[0] is None:
    #                 tc_stats.append(None)
    #             else:
    #                 tc_stats.append(np.stack(sc_stats, 1))
    #         wdiff_stats_pgd_classify.append(tc_stats)

    # if fit_classifier:
    #     logging.info('fitting classifier')
    #     for tc in tqdm.trange(nb_classes):
    #         tc_X = []
    #         tc_Y = []
    #         idx_wo_tc = [sc for sc in range(nb_classes) if sc != tc]
    #         for i, sc in enumerate(idx_wo_tc):
    #             sc_data = wdiff_stats_pgd_classify[tc][i]
    #             if sc_data is not None:
    #                 sc_data = sc_data.reshape(sc_data.shape[0], -1)
    #                 for d in sc_data:
    #                     tc_X.append(d.ravel())
    #                     tc_Y.append(sc)
    #         Y_unq = np.unique(tc_Y)
    #         if len(Y_unq) == 0:
    #             lr = SimpleNamespace(predict=lambda x: np.array(tc))
    #         elif len(Y_unq) == 1:
    #             lr = SimpleNamespace(predict=lambda x: np.array(tc_Y[0]))
    #         else:
    #             tc_X = np.stack(tc_X)
    #             tc_Y = np.array(tc_Y)
    #             lr = LogisticRegression(solver='lbfgs',
    #                                     multi_class='multinomial',
    #                                     max_iter=1000)
    #             lr.fit(tc_X, tc_Y)
    #         wdiff_stats_pgd_classify[tc] = lr

    batch = yield

    while batch is not None:
        batch_latent, batch_pred = get_latent_and_pred(batch)
        corrected_pred = []
        detection = []
        for b, latent_b, pred_b in zip(batch, batch_latent, batch_pred):
            b_align, idx_wo_pb = _compute_alignments(b, latent_b, pred_b)
            b_align_det = np.stack([b_align[eps] for eps in noise_eps_detect])
            b_align = np.stack([b_align[eps] for eps in noise_eps])

            wdsc_det_pred_b = wdiff_stats_clean_detect[pred_b]
            if wdsc_det_pred_b is None:
                z_hit = False
            else:
                wdm_det, wds_det = wdsc_det_pred_b
                z_clean = (b_align_det - wdm_det[:, None]) / wds_det[:, None]
                z_clean_mean = z_clean.mean(1)
                z_cutoff = scipy.stats.norm.ppf(p_ratio_cutoff)
                z_hit = z_clean_mean.mean(0).max(-1) > z_cutoff

            # if not just_detect:
            #     if fit_classifier:
            #         lr = wdiff_stats_pgd_classify[pb]
            #         b_align = b_align.mean(1).reshape((1, -1))
            #         lr_pred = lr.predict(b_align)
            #     else:
            #         wdp = wdiff_stats_pgd_classify[pb]
            #         if wdp is None:
            #             z_pgd_mode = None
            #         else:
            #             wdp_not_none_idcs = [
            #                 i for i, w in enumerate(wdp) if w is not None
            #             ]
            #             if len(wdp_not_none_idcs) == 0:
            #                 z_pgd_mode = None
            #             else:
            #                 wdp = np.stack([wdp[i] for i in wdp_not_none_idcs],
            #                                2)
            #                 idx_wo_pb_wdp = [
            #                     idx_wo_pb[i] for i in wdp_not_none_idcs
            #                 ]
            #                 ssidx = np.arange(wdp.shape[-2])
            #                 wdp = wdp[:, :, ssidx, ssidx]
            #                 wdmp, wdsp = wdp
            #                 b_align = b_align[:, :, wdp_not_none_idcs]
            #                 z_pgd = (b_align - wdmp[:, None]) / wdsp[:, None]
            #                 z_pgd_mean = z_pgd.mean(1)
            #                 z_pgd_mode = scipy.stats.mode(
            #                     z_pgd_mean.argmax(-1)).mode[0]
            if z_hit:
                # if not just_detect:
                #     if fit_classifier:
                #         print(lr_pred)
                #         pb = lr_pred.item()
                #     else:
                #         if z_pgd_mode is not None:
                #             pb = idx_wo_pb_wdp[z_pgd_mode]
                detection.append(True)
            else:
                detection.append(False)
            corrected_pred.append(pred_b)
        batch = yield np.stack((corrected_pred, detection), -1)


def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100

    # load networks
    if args.net_type == 'densenet':
        model = models.DenseNet3(100, int(args.num_classes))
        model.load_state_dict(
            torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                 (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
        ])
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(
            torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(
        args.dataset, args.batch_size, in_transform, args.dataroot)
    test_clean_data, test_adv_data, test_noisy_data, test_label = {}, {}, {}, {}
    clean_loaders, adv_loaders, noisy_loaders = {}, {}, {}
    adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2', 'PGD100']
    for adv_type in adv_types:
        test_clean_data[adv_type] = torch.load(
            args.outf + 'clean_data_%s_%s_%s.pth' %
            (args.net_type, args.dataset, adv_type))
        test_adv_data[adv_type] = torch.load(
            args.outf + 'adv_data_%s_%s_%s.pth' %
            (args.net_type, args.dataset, adv_type))
        test_noisy_data[adv_type] = torch.load(
            args.outf + 'noisy_data_%s_%s_%s.pth' %
            (args.net_type, args.dataset, adv_type))
        test_label[adv_type] = torch.load(
            args.outf + 'label_%s_%s_%s.pth' %
            (args.net_type, args.dataset, adv_type))
        clean_loaders[adv_type] = torch.utils.data.DataLoader(
            test_clean_data[adv_type], batch_size=args.batch_size)
        adv_loaders[adv_type] = torch.utils.data.DataLoader(
            test_adv_data[adv_type], batch_size=args.batch_size)
        noisy_loaders[adv_type] = torch.utils.data.DataLoader(
            test_noisy_data[adv_type], batch_size=args.batch_size)

    model.eval()

    # datasets = {}
    method_name = 'odd_odds'
    dir_name = f'{method_name}_{args.net_type}_{args.dataset}'
    exists = os.path.isdir(dir_name)
    if not exists:
        os.mkdir(dir_name)
    results_filename = f'{dir_name}/results_known.txt'
    with open(results_filename, 'w') as results_file:
        for adv_type in adv_types:
            train_split = 0.1
            train_size = int(train_split * len(test_clean_data[adv_type]))
            test_size = len(test_clean_data[adv_type]) - train_size
            X = np.concatenate([
                test_clean_data[adv_type][:train_size],
                test_adv_data[adv_type][:train_size],
                test_noisy_data[adv_type][:train_size],
            ])
            label_y = np.concatenate([
                test_label[adv_type][:train_size],
                test_label[adv_type][:train_size],
                test_label[adv_type][:train_size],
            ])
            adv_y = np.concatenate([
                np.ones(train_size),
                np.zeros(train_size),
                np.ones(train_size),
            ])
            X_test = np.concatenate([
                test_clean_data[adv_type][train_size:],
                test_adv_data[adv_type][train_size:],
                test_noisy_data[adv_type][train_size:],
            ])
            label_y_test = np.concatenate([
                test_label[adv_type][train_size:],
                test_label[adv_type][train_size:],
                test_label[adv_type][train_size:],
            ])
            adv_y_test = np.concatenate([
                np.ones(test_size),
                np.zeros(test_size),
                np.ones(test_size),
            ])

            # "train"
            class_vectors = model.get_class_vectors()
            num_classes = class_vectors.size(0)
            noise_eps = 'n0.01,s0.01,u0.01,n0.02,s0.02,u0.02,s0.03,n0.03,u0.03'
            # noise_eps_detect = 'n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008'
            clip_min = min(X.min(), X_test.min())
            clip_max = max(X.max(), X_test.max())
            predictor = collect_statistics(
                X,
                label_y,  # TODO these are class labels, not adversarial binary labels
                latent_and_logits_fn=model.forward_with_latent,
                nb_classes=args.num_classes,
                weights=model.get_class_vectors(),
                targeted=False,
                noise_eps=noise_eps.split(','),
                # noise_eps_detect=noise_eps_detect.split(','))
                noise_eps_detect=None,
                num_noise_samples=256,
                batch_size=128,
                clip_min=clip_min,
                clip_max=clip_max,
                p_ratio_cutoff=0.999,
                clip_alignments=True,
                cache_alignments_dir=dir_name if exists else None,
            )
            next(predictor)

            # test
            y_pred_list = []
            batch_size = 128
            for i in range(X_test.shape[0] // batch_size + 1):
                X_test_batch = X_test[i * batch_size:(i + 1) * batch_size]
                corrected_batch, y_pred_batch = predictor.send(X_test_batch).T
                print(f'corrected_batch.shape: {corrected_batch.shape}')
                print(f'y_pred_batch.shape: {y_pred_batch.shape}')
                print(f'y_pred_batch: {y_pred_batch}')
                y_pred_list.append(y_pred_batch)
            y_pred = np.concatenate(y_pred_list, axis=0)
            y_scores = y_pred
            acc = accuracy_score(adv_y_test, y_pred)
            auroc = roc_auc_score(adv_y_test, y_scores)
            print(f'Accuracy on {adv_type}: {acc}', file=results_file)
            results[run_n][f'acc_known_{adv_type}'] = acc
            print(f'AUROC on {adv_type}: {auroc}', file=results_file)
            results[run_n][f'auroc_known_{adv_type}'] = auroc

    # TODO unknown variant
    ...

    # TODO unsupervised variant
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        metavar='N',
                        help='batch size for data loader')
    parser.add_argument('--dataset',
                        required=True,
                        help='cifar10 | cifar100 | svhn')
    parser.add_argument('--dataroot',
                        default=str(Path.home() / '.datasets'),
                        help='path to dataset')
    parser.add_argument('--outf',
                        default='./adv_output/',
                        help='folder with data')
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='the # of classes')
    parser.add_argument('--net_type', required=True, help='resnet | densenet')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    # parser.add_argument('--runs', default=5, help='number of runs')
    args = parser.parse_args()
    print(args)
    main()
