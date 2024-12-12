import numpy as np
import torch
import torch.nn as nn

from model.kernel import CauchyKernel, ExpQuadKernel
    

def _add_diag_safety_shift(matrix, safety_shift=1e-8):
    return matrix + safety_shift * torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)


class SVGP(nn.Module):

    def __init__(self,
                 fixed_inducing_points, # Flag to fix the inducing points
                 inducing_point_steps, # Number of inducing points (p)
                 loc_range, # Number of inducing points
                 kernel_type, # Kernel type: Cauchy, EQ
                 kernel_scale, # Initial kernel scale
                 safety_shift, # Little shift for matrix diagonals to ensure invertibility
                 train_size, # Total number of training samples / observations
                 ):
        super(SVGP, self).__init__()
        self.train_size = train_size
        self.safety_shift = safety_shift

        # Setting inducing points
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1 + eps):(1. / inducing_point_steps), 0:(1 + eps):(1. / inducing_point_steps)].reshape(2, -1).T * loc_range

        if fixed_inducing_points:
            self.inducing_points = torch.tensor(initial_inducing_points)
        else:
            self.inducing_points = nn.Parameter(torch.tensor(initial_inducing_points))

        # Setting kernel
        if kernel_type == 'Cauchy':
            self.kernel = CauchyKernel(scale=kernel_scale)
        elif kernel_type == 'EQ':
            self.kernel = ExpQuadKernel(scale=kernel_scale)
        else:
            raise Exception("Unknown kernel")
        
    def stochastic_estimate_posterior(self,
                                      x_test, # Test spatial locations: test is only different when denoising
                                      x_train, # Train spatial locations
                                      omega_l, # Inferred (by encoder) latent mean: latent dimension l
                                      phi_l, # Inferred (by encoder) latent variance: latent dimension l
                                      ):
        b = x_train.shape[0] # Batch size

        # Covariance matrices
        K_pp = self.kernel(self.inducing_points, self.inducing_points) # shape (p, p)
        K_pp_inv = torch.linalg.inv(_add_diag_safety_shift(K_pp, self.safety_shift)) # shape (p, p)
        K_xx = self.kernel(x_test, x_test)  # shape (test_size, test_size)
        K_xp = self.kernel(x_test, self.inducing_points)  # (test_size, p)
        K_px = torch.transpose(K_xp, 0, 1)  # (p, test_size)
        K_bp = self.kernel(x_train, self.inducing_points) # shape (b, p)
        K_pb = torch.transpose(K_bp, 0, 1) # shape (p, b)

        Sigma_bl = K_pp + (self.train_size / b) * torch.matmul(K_pb, K_bp / phi_l[:, None])
        Sigma_bl_inv = torch.linalg.inv(_add_diag_safety_shift(Sigma_bl, self.safety_shift))

        # Stochastic estimates
        mu_hat_l = (self.train_size / b) * torch.matmul(torch.matmul(K_pp, torch.matmul(Sigma_bl_inv, K_pb)), omega_l / phi_l)
        A_hat_l = torch.matmul(K_pp, torch.matmul(Sigma_bl_inv, K_pp))
        
        # Calculate m_l, B_l for a test batch
        m_xl = (self.train_size / b) * torch.matmul(K_xp, torch.matmul(Sigma_bl_inv, torch.matmul(K_pb, omega_l / phi_l)))
        B_xl = torch.diagonal(K_xx - torch.matmul(K_xp, torch.matmul(K_pp_inv, K_px)) +
                              torch.matmul(K_xp, torch.matmul(Sigma_bl_inv, K_px)))

        return m_xl, B_xl, mu_hat_l, A_hat_l
        
    def ELBO(self,
             x, # Spatial locations
             omega_l, # Inferred (by encoder) latent mean: latent dimension l
             phi_l, # Inferred (by encoder) latent variance: latent dimension l
             mu_hat_l, # Sparse GP regression outputs mean: latent dimension l
             A_hat_l, # Sparse GP regression outputs covariance: latent dimension l
             ):
        b = x.shape[0] # Batch size
        p = self.inducing_points.shape[0] # Number of inducing points

        # Covariance matrices
        K_pp = self.kernel(self.inducing_points, self.inducing_points) # shape (p, p)
        K_pp_inv = torch.linalg.inv(_add_diag_safety_shift(K_pp, self.safety_shift)) # shape (p, p)
        K_bb = self.kernel(x, x) # shape (b, b)
        K_bp = self.kernel(x, self.inducing_points) # shape (b, p)
        K_pb = torch.transpose(K_bp, 0, 1) # shape (p, b)

        # KL term: using the formula of the KL divergence between two multivariate normal
        K_pp_log_det = torch.logdet(_add_diag_safety_shift(K_pp, self.safety_shift))
        A_hat_l_log_det = torch.logdet(_add_diag_safety_shift(A_hat_l, self.safety_shift))
        KL_term = (K_pp_log_det - A_hat_l_log_det - p + torch.trace(torch.matmul(K_pp_inv, A_hat_l)) +
                   torch.sum(mu_hat_l * torch.matmul(K_pp_inv, mu_hat_l))) / 2
        
        # Sum term: using the formula of the log of N(mean, var)
        posterior_mean = torch.matmul(K_bp, torch.matmul(K_pp_inv, mu_hat_l))
        K_tilde_ii = torch.diagonal(K_bb - torch.matmul(K_bp, torch.matmul(K_pp_inv, K_pb)))
        Lambda = torch.matmul(K_pp_inv, torch.matmul(torch.matmul(K_bp.unsqueeze(2), torch.transpose(K_bp.unsqueeze(2), 1, 2)), K_pp_inv))
        tr_A_hat_l_Lambda = torch.einsum('bii->b', torch.matmul(A_hat_l, Lambda)) # Calculate trace for the batch
        Sum_term = -(((K_tilde_ii + tr_A_hat_l_Lambda) / phi_l).sum() +
                     torch.log(phi_l).sum() + b * np.log(2 * np.pi) +
                     ((omega_l - posterior_mean) ** 2 / phi_l).sum()) / 2.
        
        return Sum_term, KL_term