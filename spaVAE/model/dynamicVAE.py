from math import exp

class PIControl():
    def __init__(self, Kp, Ki, beta_init, beta_min, beta_max):
        self.beta_prev = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.e_prev = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def update(self, KL_expected, KL_curr):
        e = (KL_expected - KL_curr) * 5. # Following the authors' suggestion: the error is enlarged 5 times to allow faster tuning of beta
        dP = self.Kp * (1.0 / (1.0 + exp(e)) - 1.0 / (1.0 + exp(self.e_prev)))
        dI = self.Ki * e

        if self.beta_prev < self.beta_min:
            dI = 0

        dbeta = dP + dI
        
        # Clip beta to force it into the [beta_min, beta_max] range
        beta = min(max(dbeta + self.beta_prev, self.beta_min), self.beta_max)

        self.beta_prev = beta
        self.e_prev = e

        return beta