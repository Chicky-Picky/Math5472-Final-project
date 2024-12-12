import argparse
import os
from collections import Counter

import numpy as np
import h5py

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
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
from torch.utils.data.sampler import SubsetRandomSampler

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

    # setup datasets
    train_dataset = run_setup.init_obj(name="dataset",
                                       module=module_dataset,
                                       )
    test_dataset = run_setup.init_obj(name=dataset_type,
                                      module=module_dataset,
                                      )
    
    # setup data_loader instances
    train_indices = np.loadtxt(os.path.join(train_dataset.root, 'train_indices.txt'), dtype=int)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_indices, batch_size=run_setup['dataloader']['args']['batch_size'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=run_setup['dataloader']['args']['batch_size'])

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

    latents_mean_train = []
    latents_mean_test = []
    counts_train = []
    counts_test = []
    denoised_counts_train = []
    denoised_counts_test = []
    pos_train = []
    pos_test = []

    with torch.no_grad():
        enc_mus = []
        enc_vars = []
        for batch_idx, data in enumerate(tqdm(train_loader)):
            x_train, y_train, raw_y_train, size_factors_train = data

            x_train = x_train.to(device)
            y_train = y_train.to(device)
            raw_y_train = raw_y_train.to(device)
            size_factors_train = size_factors_train.to(device)

            counts_train += [z.numpy() for z in y_train]

            output = model(x_train, y_train, raw_y_train, size_factors_train)
            post_m, post_v = output[2], output[3]
            latents_mean_train += [z.numpy() for z in post_m]

            latent_dist = Normal(post_m, torch.sqrt(post_v))
            latent_samples = []
            for _ in range(num_samples):
                latent_samples += [latent_dist.rsample()]

            mean_samples = []
            for z in latent_samples:
                mean_samples_, _ = model.decoder(z)
                mean_samples += [mean_samples_.numpy()]
            denoised_counts_train += [z for z in np.array(mean_samples).mean(axis=0)]

            enc_mu, enc_var = model.encoder(y_train)
            enc_mus += [enc_mu]
            enc_vars += [enc_var]

            pos_train += [x_train]

        pos_train = torch.cat(pos_train, dim=0)

        enc_mus = torch.cat(enc_mus, dim=0)
        enc_vars = torch.cat(enc_vars, dim=0)

        gp_mu = enc_mus[:, :model.GP_dim]
        gp_var = enc_vars[:, :model.GP_dim]

        g_mu = enc_mus[:, model.GP_dim:]
        g_var = enc_vars[:, model.GP_dim:]

        for batch_idx, data in enumerate(tqdm(test_loader)):
            x_test, y_test, raw_y_test, size_factors_test = data

            x_test = x_test.to(device)
            y_test = y_test.to(device)
            raw_y_test = raw_y_test.to(device)
            size_factors_test = size_factors_test.to(device)

            pos_test += [z.numpy() for z in x_test]
            counts_test += [z.numpy() for z in y_test]

            nbr_indices_train = torch.argmin(torch.cdist(x_test, pos_train), dim=1)
            g_mu = g_mu[nbr_indices_train]
            g_var = g_var[nbr_indices_train]

            # Update of the posterior params
            gp_post_m, gp_post_B = [], []
            for l in range(model.GP_dim):
                m_bl, B_bl, _, _ = model.svgp.stochastic_estimate_posterior(x_test=x_test,
                                                                            x_train=pos_train,
                                                                            omega_l=gp_mu[:, l],
                                                                            phi_l=gp_var[:, l])
            
                gp_post_m += [m_bl]
                gp_post_B += [B_bl]

            gp_post_m = torch.stack(gp_post_m, dim=1)
            gp_post_B = torch.stack(gp_post_B, dim=1)
            
            # Posterior params are changed because the test set doesn't coincide with the train set here
            post_m = torch.cat((gp_post_m, g_mu), dim=1)
            post_v = torch.cat((gp_post_B, g_var), dim=1)

            latents_mean_test += [z.numpy() for z in post_m]
            
            latent_dist = Normal(post_m, torch.sqrt(post_v))
            latent_samples = []
            for _ in range(num_samples):
                latent_samples += [latent_dist.rsample()]

            mean_samples = []
            for z in latent_samples:
                mean_samples_, _ = model.decoder(z)
                mean_samples += [mean_samples_.numpy()]
            denoised_counts_test += [z for z in np.array(mean_samples).mean(axis=0)]

            output = model(x_test, y_test, raw_y_test, size_factors_test)
            loss = criterion(output)

            output_ = []
            for item in output:
                if type(item) == torch.Tensor:
                    output_ += [item.detach()]
                else:
                    output_ += [item]

            vizualizer_setup.set_step(batch_idx, mode="test")
            test_metrics.update('loss', loss.item() / y_test.shape[0])

            message = [batch_idx + 1, loss.item() / y_test.shape[0]]

            for met in metrics:
                test_metrics.update(met.__name__, met(output_))
                message.append(met(output_))

            logger.info(" - ".join(str(x) for x in message))

    latents_mean_train = np.array(latents_mean_train)
    latents_mean_test = np.array(latents_mean_test)
    denoised_counts_train = np.array(denoised_counts_train)
    denoised_counts_test = np.array(denoised_counts_test)
    pos_train = np.array(pos_train)
    pos_test = np.array(pos_test)
    counts_train = np.array(counts_train)
    counts_test = np.array(counts_test)

    data_mat = h5py.File(train_dataset.raw_filename, 'r')
    gt_labels_train = np.array(data_mat['Y']).astype('U26')[train_indices]
    data_mat.close()

    data_mat = h5py.File(test_dataset.raw_filename, 'r')
    gt_labels_test = np.array(data_mat['Y']).astype('U26')
    data_mat.close()

    return (test_metrics.result(), latents_mean_train, latents_mean_test,
            denoised_counts_train, denoised_counts_test, counts_train,
            counts_test, pos_train, pos_test, gt_labels_train, gt_labels_test)


def knn_predict_latent(latents_mean_train, latents_mean_test, gt_labels_train, gt_labels_test):
    knn = KNeighborsClassifier(11)
    knn.fit(latents_mean_train, gt_labels_train)
    
    labels = np.array(knn.predict(latents_mean_test))
    print("Accuracy", (labels == gt_labels_test).mean())

def plot_cluster_latent(latents_mean_test, pos_test, gt_labels_test, checkpoint_dir):
    # Cluster latent variables
    labels = np.array(KMeans(n_clusters=7, n_init=100).fit_predict(latents_mean_test))

    # Refine the labels wrt neighborhood
    refined = []
    dist = pairwise_distances(pos_test, metric="euclidean", n_jobs=-1).astype(np.double)
    for i in range(pos_test.shape[0]):
        nbrs = np.argpartition(dist[i], 6)[:6]
        nbr_lbls = labels[nbrs]
        cnt = Counter(nbr_lbls)
        if cnt[labels[i]] < 3:
            refined += [max(cnt, key=cnt.get)]
        else:
            refined += [labels[i]]
    refined = np.array(refined)

    NMI = np.round(normalized_mutual_info_score(gt_labels_test, refined), 3)
    print('Normalized Mutual Information', NMI)
    ARI = np.round(adjusted_rand_score(gt_labels_test, refined), 3)
    print('Adjusted Rand Index', ARI)

    # Plot position wrt clustering
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for i in range(7):
        ax.plot(pos_test[refined == i][:, 0], pos_test[refined == i][:, 1], marker='o', linestyle='', ms=2, label=str(i + 1))
    ax.legend()

    plt.show()
    fig.savefig(os.path.join(checkpoint_dir, 'clustered pos.png'), dpi=fig.dpi)


def plot_umap_latent(latents_mean_train, latents_mean_test, gt_labels_train, gt_labels_test, checkpoint_dir):
    # Reduce the dimension of latent variables
    umap_2d = UMAP(n_components=2, init='random').fit(np.vstack((latents_mean_train, latents_mean_test)))
    p = umap.plot.points(umap_2d, labels=np.hstack((gt_labels_train, gt_labels_test)))

    umap.plot.show(p)
    p.figure.savefig(os.path.join(checkpoint_dir, 'ground true umap latent.png'))


def plot_denoised_counts(counts_train, counts_test, denoised_counts_train, denoised_counts_test, pos_train, pos_test, checkpoint_dir):
    # Plot normalized true and denoised counts
    fig, axs = plt.subplots(5, 2)

    counts = np.vstack((counts_train, counts_test))
    denoised_counts = np.vstack((denoised_counts_train, denoised_counts_test))
    pos = np.vstack((pos_train, pos_test))

    # Normalize counts s.t. color is from 0 to 1
    c_max = np.hstack((counts[:, :5],
                       denoised_counts[:, :5])).max()
    c_min = np.hstack((counts[:, :5],
                       denoised_counts[:, :5])).min()
    counts_train_norm = (counts_train[:, :5]  - c_min) / (c_max - c_min)
    denoised_counts_norm = (denoised_counts[:, :5]  - c_min) / (c_max - c_min)

    # Plot normalized counts
    cmap = plt.get_cmap('jet', 11)
    for i, gene in enumerate(["ISG15", "RPL22", "EFHD2", "CAMK2N1", "RPL11"]):
        axs[i, 0].scatter(pos_train[:, 0], pos_train[:, 1], c=counts_train_norm[:, i], cmap=cmap, marker='o', s=0.1)
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
    (_, latents_mean_train, latents_mean_test,
     denoised_counts_train, denoised_counts_test, counts_train,
     counts_test, pos_train, pos_test, gt_labels_train, gt_labels_test) = run_testing(run_setup,
                                                                                      log_setup,
                                                                                      vizualizer_setup=vizualizer_setup,
                                                                                      device="cpu",
                                                                                      dataset_type=dataset_type)
    
    knn_predict_latent(latents_mean_train, latents_mean_test, gt_labels_train, gt_labels_test)
    plot_cluster_latent(latents_mean_test, pos_test, gt_labels_test, checkpoint_dir)
    plot_umap_latent(latents_mean_train, latents_mean_test, gt_labels_train, gt_labels_test, checkpoint_dir)
    plot_denoised_counts(counts_train, counts_test, denoised_counts_train, denoised_counts_test, pos_train, pos_test, checkpoint_dir)
    

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