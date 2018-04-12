/*
	GP posterior
*/

functions {
    matrix joint_cov(real[] x,real alpha,real rho,real delta) {
        int N = size(x);
        matrix[N, N] K;
        real sq_alpha = square(alpha);
        for (i in 1:(N-1)) {
            K[i, i] = sq_alpha + delta;
            for (j in (i + 1):N) {
                K[i, j] = sq_alpha * exp(-0.5 * square((x[i] - x[j])/ rho));
                K[j, i] = K[i, j];
            }
        }
        K[N, N] = sq_alpha + delta;
        return cholesky_decompose(K);
    }
}

data {
    int<lower=1> N;
    real x[N];
    vector[N] y;
    int<lower=1> N_prm;
    real x_prm[N_prm];
    real sig_var;
}

transformed data {
    vector[N] mu;
    int<lower=1> N_tot = N + N_prm;
    real x_tot[N_tot];
    real delta=1e-8;
    
    for (n in 1:N) x_tot[n] = x[n];
    for (n in 1:N_prm) x_tot[N + n] = x_prm[n];
    mu=rep_vector(0,N);
    
    
}

parameters {
    real<lower=0.000001> rho;
    real<lower=0.000001> alpha;
    vector[N_tot] eta;
    real<lower=0.000001> sigma;
}

transformed parameters{
     vector[N_tot] f;
    {
        matrix[N_tot, N_tot] L_K = joint_cov(x_tot,alpha,rho,delta);
        f = L_K * eta;
    }
}


model {    
    rho ~ inv_gamma(5, 5);
    alpha ~ normal(0, 1);
    sigma ~ normal(0,sig_var);
    eta ~ normal(0, 1);
    y ~ normal(f[1:N], sigma);
}

generated quantities {
    vector[N_prm] y_prm;
    for(n in 1:N_prm)
        y_prm[n] = normal_rng(f[N+n], sigma);
}
