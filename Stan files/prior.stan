/* 
Simple GP prior generating samples of functions 
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
        K[N,N] = sq_alpha + delta;
        return cholesky_decompose(K);
    }
}


data {
    int<lower=1> N;
    real x[N];
}

transformed data {
    vector[N] mu;
    real delta=1e-8;
    mu=rep_vector(0,N);
}

parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    vector[N] eta;
}

transformed parameters{
    vector[N] f;
    {
        matrix[N, N] L_K;
        L_K = joint_cov(x,alpha,rho,delta);
        f = mu + L_K*eta;
    }
}

model {    
    rho ~ inv_gamma(5,5);
    alpha ~ normal(0, 1);
    eta ~ normal(0, 1);
}
