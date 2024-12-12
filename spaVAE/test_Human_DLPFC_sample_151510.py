import argparse
import os
from collections import Counter

import numpy as np
import h5py

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score, adjusted_rand_score

from umap import UMAP
import umap.plot

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.distributions.normal import Normal

import dataset as module_dataset
import utils.metric as module_metric
import model.model as module_arch

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.setup import SetupRun, SetupLogger
from utils.util import read_json
from utils.visualization import TensorboardWriter
from utils.util import MetricTracker


def run_testing(run_setup: SetupRun,
                logger_setup: SetupLogger,
                vizualizer_setup,
                dataset_type,
                device: str,
                num_samples=25,
                ):
    # setup logger
    logger = logger_setup("test")

    # setup dataset
    dataset = run_setup.init_obj(name=dataset_type,
                                 module=module_dataset,
                                 )
    # setup data_loader instances
    test_loader = DataLoader(dataset=dataset, batch_size=run_setup['dataloader']['args']['batch_size'])

    # setup model architecture, then print to console
    model = run_setup.init_obj('arch', module_arch)
    logger.info(model)

    # loading the best checkpoint from training
    path_to_checkpoint = os.path.join("experiments", run_setup["outdir"], "checkpoint", "model_best.pth")
    model.load_state_dict(torch.load(path_to_checkpoint, weights_only=False, map_location=torch.device('cpu'))['state_dict'])

    # setup function handles of metrics
    criterion = run_setup.init_funct('loss', module_metric) # Loss is from metric.py
    metrics = [run_setup.init_funct(metric, module_metric) for metric in run_setup['metrics']]

    # run testing process with saving metrics in logs
    model.eval()
    test_metrics = MetricTracker('loss', *[m.__name__ for m in metrics], writer=vizualizer_setup)

    message = ["batch_num", "loss"]
    for met in metrics:
        message.append(met.__name__)

    logger.info(" - ".join(message))

    latents_mean = []
    counts = []
    denoised_counts = []
    pos = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):

            x, y, raw_y, size_factors = data
            x = x.to(device)
            y = y.to(device)
            raw_y = raw_y.to(device)
            size_factors = size_factors.to(device)

            pos += [z.numpy() for z in x]
            counts += [z.numpy() for z in y]

            # Posterior params aren't changed because the test set coincides with the train set here
            output = model(x, y, raw_y, size_factors)
            post_m, post_v = output[2], output[3]
            latents_mean += [z.numpy() for z in post_m]
            
            latent_dist = Normal(post_m, torch.sqrt(post_v))
            latent_samples = []
            for _ in range(num_samples):
                latent_samples += [latent_dist.rsample()]

            mean_samples = []
            for z in latent_samples:
                mean_samples_, _ = model.decoder(z)
                mean_samples += [mean_samples_.numpy()]
            denoised_counts += [z for z in np.array(mean_samples).mean(axis=0)]

            loss = criterion(output)

            output_ = []
            for item in output:
                if type(item) == torch.Tensor:
                    output_ += [item.detach()]
                else:
                    output_ += [item]

            vizualizer_setup.set_step(batch_idx, mode="test")
            test_metrics.update('loss', loss.item() / y.shape[0])

            message = [batch_idx + 1, loss.item() / y.shape[0]]

            for met in metrics:
                test_metrics.update(met.__name__, met(output_))
                message.append(met(output_))

            logger.info(" - ".join(str(x) for x in message))

    latents_mean = np.array(latents_mean)
    denoised_counts = np.array(denoised_counts)
    pos = np.array(pos)
    counts = np.array(counts)

    data_mat = h5py.File(dataset.raw_filename, 'r')
    gt_labels = np.array(data_mat['Y']).astype('U26')
    data_mat.close()

    return test_metrics.result(), latents_mean, denoised_counts, counts, pos, gt_labels


def plot_cluster_latent(latents_mean, pos, gt_labels, checkpoint_dir):
    # Cluster latent variables
    labels = np.array(KMeans(n_clusters=7, n_init=100).fit_predict(latents_mean))

    # Refine the labels wrt neighborhood
    refined = []
    dist = pairwise_distances(pos, metric="euclidean", n_jobs=-1).astype(np.double)
    for i in range(pos.shape[0]):
        nbrs = np.argpartition(dist[i], 6)[:6]
        nbr_lbls = labels[nbrs]
        cnt = Counter(nbr_lbls)
        if cnt[labels[i]] < 3:
            refined += [max(cnt, key=cnt.get)]
        else:
            refined += [labels[i]]
    refined = np.array(refined)

    NMI = np.round(normalized_mutual_info_score(gt_labels, refined), 3)
    print('Normalized Mutual Information', NMI)
    ARI = np.round(adjusted_rand_score(gt_labels, refined), 3)
    print('Adjusted Rand Index', ARI)

    # Plot position wrt clustering
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for i in range(7):
        ax.plot(pos[refined == i][:, 0], pos[refined == i][:, 1], marker='o', linestyle='', ms=2, label=str(i + 1))
    ax.legend()

    plt.show()
    fig.savefig(os.path.join(checkpoint_dir, 'clustered pos.png'), dpi=fig.dpi)


def plot_umap_latent(latents_mean, gt_labels, checkpoint_dir):
    # Reduce the dimension of latent variables
    umap_2d = UMAP(n_components=2, init='random').fit(latents_mean)
    p = umap.plot.points(umap_2d, labels=gt_labels)
    
    umap.plot.show(p)
    p.figure.savefig(os.path.join(checkpoint_dir, 'ground true umap latent.png'))


def plot_denoised_counts(counts, denoised_counts, pos, checkpoint_dir):
    # Plot normalized true and denoised counts
    fig, axs = plt.subplots(5, 2)

    # Normalize counts s.t. color is from 0 to 1
    c_max = np.hstack((counts[:, :5],
                       denoised_counts[:, :5])).max()
    c_min = np.hstack((counts[:, :5],
                       denoised_counts[:, :5])).min()
    counts_norm = (counts[:, :5]  - c_min) / (c_max - c_min)
    denoised_counts_norm = (denoised_counts[:, :5]  - c_min) / (c_max - c_min)

    # Plot normalized counts
    cmap = plt.get_cmap('jet', 11)
    for i, gene in enumerate(["ISG15", "EFHD2", "CAMK2N1", "RPL11", "STMN1"]):
        axs[i, 0].scatter(pos[:, 0], pos[:, 1], c=counts_norm[:, i], cmap=cmap, marker='o', s=0.1)
        axs[i, 0].set_title(gene)

        axs[i, 1].scatter(pos[:, 0], pos[:, 1], c=denoised_counts_norm[:, i], cmap=cmap, marker='o', s=0.1)
        axs[i, 1].set_title(gene + " + denoising")
    for ax in axs.flat:
        ax.label_outer()
    fig.tight_layout()

    # Adjust axis for a colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    # Add colorbar
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ticks=np.linspace(0, 1, 11), 
                 boundaries=np.arange(-0.05, 1.1, .1),
                 cax=cbar_ax)
    
    fig.set_size_inches(4, 8)
    plt.show()
    fig.savefig(os.path.join(checkpoint_dir, 'norm true vs denoised counts.png'), dpi=fig.dpi)


def main(config,
         log_config,
         dataset_type
         ):
    # read configurations, hyperparameters for training and logging
    config = read_json(config)
    log_config = read_json(log_config)

    # set directories where trained model and log will be saved.
    checkpoint_dir = os.path.join("tests", config['outdir'], dataset_type)
    log_dir = os.path.join("tests", config['outdir'], dataset_type)

    run_setup = SetupRun(config=config,
                         checkpoint_dir=checkpoint_dir)

    log_setup = SetupLogger(config=log_config,
                            log_dir=log_dir)

    cfg_trainer = run_setup['trainer']['args']
    vizualizer_setup = TensorboardWriter(log_dir, cfg_trainer['tensorboard'])

    # run testing process
    _, latents_mean, denoised_counts, counts, pos, gt_labels = run_testing(run_setup,
                                                                           log_setup,
                                                                           vizualizer_setup=vizualizer_setup,
                                                                           device="cpu",
                                                                           dataset_type=dataset_type)
    
    plot_cluster_latent(latents_mean, pos, gt_labels, checkpoint_dir)
    plot_umap_latent(latents_mean, gt_labels, checkpoint_dir)
    plot_denoised_counts(counts, denoised_counts, pos, checkpoint_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-l', '--log-config', default="logger_config.json", type=str,
                        help='log config file path (default: logger_config.json)')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--dataset_type', default="dataset_test", type=str, choices=["dataset_test", "dataset"])
    parser.add_argument('--num_samples', default=25, type=int, help='number of latent variables sampled for denoising')
    args = parser.parse_args()

    main(config=args.config,
         log_config=args.log_config,
         dataset_type="dataset_test"
         )