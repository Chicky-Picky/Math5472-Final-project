import numpy as np
import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from abc import abstractmethod

from collections import deque

from utils.metric import NBLoss, CE
from model.sparseGP import SVGP
from model.dynamicVAE import PIControl
from model.encoder_decoder import DenseEncoder, DenseDecoder, ConvEncoder


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class spaVAE(BaseModel):
    def __init__(self,
                 in_features: int, # Number of genes
                 GP_dim: int, # Dimension of Gaussian Process prior
                 Normal_dim: int, # Dimension of Gaussian prior
                 encoder_hidden_dims: list, # Hidden dimensions of the layers for Encoder NN
                 decoder_hidden_dims: list, # Hidden dimensions of the layers for Decoder NN
                 train_size: int, # Total number of training samples / observations
                 KL_loss: float, # Desired KL loss for dynamic VAE
                 beta_init: float = 10., # Initial weight of KL loss
                 beta_min: float = 4., # Minimum weight of KL loss
                 beta_max: float = 25., # Maximum weight of KL loss
                 kernel_type: str = 'Cauchy', # Kernel type: Cauchy, EQ
                 dtype: float = torch.float64, # Torch tensors data type
                 dynamicVAE: bool = True, # Flag to use Dynamic VAE for tuning beta
                 noise: float = 0., # Add noise to the input for robustness
                 nn_type: str = 'dense', # Dense or Convolutional Encoder NN
                 encoder_dropout: float = 0., # Dropout of Encoder NN
                 decoder_dropout: float = 0., # Dropout of Decoder NN
                 activation: str = 'elu', # Activation function for Encoder / Decoder NNs
                 norm: str = 'batchnorm', # Normalization method for Encoder / Decoder NNs
                 fixed_inducing_points: bool = True, # Flag to fix the inducing points for Sparse GP
                 inducing_point_steps: int = 6, # Number of inducing points
                 loc_range: float = 20., # Location range after rescaling
                 kernel_scale: float = 20., # Initial kernel scale
                 safety_shift: float = 1e-8, # Little shift for matrix diagonals to ensure invertibility
                 ):

        super().__init__()
        torch.set_default_dtype(dtype)
        
        # Sparse Gaussian Process instance
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points,
                         inducing_point_steps=inducing_point_steps,
                         loc_range=loc_range,
                         kernel_type=kernel_type,
                         kernel_scale=kernel_scale,
                         safety_shift=safety_shift,
                         train_size=train_size,
                         )
        
        # Dynamic VAE beta control
        self.beta = beta_init
        self.dynamicVAE = dynamicVAE
        self.KL_loss = KL_loss
        self.PID = PIControl(Kp=0.01,
                             Ki=-0.005,
                             beta_init=beta_init,
                             beta_min=beta_min,
                             beta_max=beta_max,
                             )
        self.KL_queue = deque() # Queue of last 10 KL terms
        
        # Dimensions
        self.in_features = in_features
        self.GP_dim = GP_dim
        self.Normal_dim = Normal_dim

        # Additional noise for robustness
        self.noise = noise

        # Encoder NN
        if nn_type == 'dense':
            self.encoder = DenseEncoder(input_dim=in_features,
                                        hidden_dims=encoder_hidden_dims,
                                        output_dim=GP_dim + Normal_dim,
                                        activation_type=activation,
                                        dropout=encoder_dropout,
                                        norm_type=norm,
                                        )
        elif nn_type == 'conv':
            self.encoder = ConvEncoder(input_dim=in_features,
                                       hidden_dims=encoder_hidden_dims,
                                       output_dim=GP_dim + Normal_dim,
                                       activation_type=activation,
                                       dropout=encoder_dropout,
                                       norm_type=norm,
                                       )
        else:
            raise Exception("Unknown neural network type")
        
        # Decoder NN
        self.decoder = DenseDecoder(input_dim=GP_dim + Normal_dim,
                                    hidden_dims=decoder_hidden_dims,
                                    output_dim=in_features,
                                    activation_type=activation,
                                    dropout=decoder_dropout,
                                    norm_type=norm,
                                    )


    def forward(self, x, y, raw_y, scale_factors, num_samples=1, update_beta=False):
        b = y.shape[0] # batch size
        enc_mu, enc_var = self.encoder(y)

        gp_mu = enc_mu[:, :self.GP_dim]
        gp_var = enc_var[:, :self.GP_dim]

        g_mu = enc_mu[:, self.GP_dim:]
        g_var = enc_var[:, self.GP_dim:]

        # L_H_l calculation for each latent dimension l
        sum_terms, kl_terms = [], []
        gp_post_m, gp_post_B = [], []
        for l in range(self.GP_dim):
            m_bl, B_bl, mu_hat_l, A_hat_l = self.svgp.stochastic_estimate_posterior(x,
                                                                                    x,
                                                                                    gp_mu[:, l],
                                                                                    gp_var[:, l])

            sum_term_l, kl_term_l = self.svgp.ELBO(x,
                                                   gp_mu[:, l],
                                                   gp_var[:, l],
                                                   mu_hat_l=mu_hat_l,
                                                   A_hat_l=A_hat_l)
            
            sum_terms += [sum_term_l]
            kl_terms += [kl_term_l]
            gp_post_m += [m_bl]
            gp_post_B += [B_bl]

        # Posterior mean and covariance of Sparse GP regression
        gp_post_m = torch.stack(gp_post_m, dim=1)
        gp_post_B = torch.stack(gp_post_B, dim=1)

        # Gaussian Process ELBO (Sum of L_H_l)
        Sum_term = (torch.stack(sum_terms, dim=-1)).sum()
        KL_term = (torch.stack(kl_terms, dim=-1)).sum()
        gp_ELBO = -Sum_term + (b / self.svgp.train_size) * KL_term

        # Gaussian Process Cross-Entropy term
        gp_ce = CE(gp_post_m, gp_post_B, gp_mu, gp_var).sum()
        
        # Gaussian Process KL
        gp_KL = gp_ce + gp_ELBO

        # Gaussian KL
        g_prior_dist = Normal(torch.zeros_like(g_mu), torch.ones_like(g_var))
        g_post_dist = Normal(g_mu, torch.sqrt(g_var))
        g_KL = kl_divergence(g_post_dist, g_prior_dist).sum()

        # Total KL
        total_KL = gp_KL + g_KL

        # Sampling from latent posterior
        post_m = torch.cat((gp_post_m, g_mu), dim=1)
        post_v = torch.cat((gp_post_B, g_var), dim=1)
        latent_dist = Normal(post_m, torch.sqrt(post_v))
        latent_samples = []
        for _ in range(num_samples):
            latent_samples += [latent_dist.rsample()]

        # Calculate reconstruction loss for gene counts
        recon_loss = 0
        for z in latent_samples:
            mu_samples_, theta_samples_ = self.decoder(z)
            recon_loss += NBLoss(raw_y,
                                 mu_samples_,
                                 theta_samples_,
                                 scale_factors,
                                 )
            
        # Take the average of the reconstruction losses over the sampled latent z
        recon_loss = recon_loss / num_samples

        # Additional noise loss
        noise_loss = 0
        if self.noise > 0:
            for _ in range(num_samples):
                noisy_enc_mu, noisy_enc_var = self.encoder(y + torch.randn_like(y) * self.noise)
                gp_mu_noisy = noisy_enc_mu[:, :self.GP_dim]
                gp_var_noisy = noisy_enc_var[:, :self.GP_dim]

                gp_post_m_noisy, gp_post_B_noisy = [], []
                for l in range(self.GP_dim):
                    m_bl, B_bl, _, _ = self.svgp.stochastic_estimate_posterior(x,
                                                                               x,
                                                                               gp_mu_noisy[:, l],
                                                                               gp_var_noisy[:, l])
                    gp_post_m_noisy += [m_bl]
                    gp_post_B_noisy += [B_bl]

                gp_post_m_noisy = torch.stack(gp_post_m_noisy, dim=1)
                gp_post_B_noisy = torch.stack(gp_post_B_noisy, dim=1)

                noise_loss += torch.sum((gp_post_m - gp_post_m_noisy) ** 2)

            # Take the average of the losses
            noise_loss = noise_loss / num_samples

        # Total ELBO
        ELBO = recon_loss + self.beta * total_KL
        if self.noise > 0 :
            ELBO += noise_loss * self.in_features / self.GP_dim

        # Control of beta: weight of KL-loss
        if self.dynamicVAE and update_beta:
            KL = (total_KL / b).item()
            self.KL_queue.append(KL)
            KL_curr = np.mean(self.KL_queue)
            self.beta = self.PID.update(self.KL_loss * (self.GP_dim + self.Normal_dim), KL_curr)
                    
            if len(self.KL_queue) >= 10:
                self.KL_queue.popleft()

        return ELBO, recon_loss, post_m.data.cpu().detach(), post_v.data.cpu().detach()