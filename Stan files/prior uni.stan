/*
	Prior with Unimodality constraints.
*/
functions {
    matrix joint_cov(real[] x,real alpha,real rho,real delta,int N1) {
        int N = size(x);
        int Nm = N-N1;
        matrix[N, N] K;
        real sq_alpha = square(alpha);
        for (i in 1:(N1-1)) {
            K[i, i] = sq_alpha + delta;
            for (j in (i + 1):N1) {
                K[i, j] = sq_alpha * exp(-0.5 * square((x[i] - x[j])/ rho));
                K[j, i] = K[i, j];
            }
        }
        K[N1, N1] = sq_alpha + delta;
        
        for (i in (N1+1):N) {
            for (j in 1:N1) {
                K[i, j] = sq_alpha * exp(-0.5 * square((x[i] - x[j])/ rho)) * ((x[j] - x[i])/ square(rho)) ;
                K[j, i] = K[i, j];
            }
        }
        
        for (i in (N1+1):(N-1)) {
            K[i, i] = (sq_alpha /square(rho))+ delta;
            for (j in (i+1):N) {
                K[i, j] = sq_alpha * exp(-0.5 * square((x[i] - x[j])/ rho)) * ((1-square((x[i] - x[j])/ rho))/square(rho));
                K[j, i] = K[i, j];
            }
        }
        K[N, N] = (sq_alpha/square(rho)) + delta;
        return cholesky_decompose(K);
    }
    
}


data {
    int<lower=1> N;
    real x[N];
    int<lower=1> N_g;
    real x_g[N_g];
    real v;
}

transformed data {
    int<lower=1> N_tot = N + N_g;
    vector[N_tot] mu;
    real delta=1e-8;
    real x_tot[N_tot];
    real k= 10;
    
    
    for (n in 1:N) x_tot[n] = x[n];
    for (n in 1:N_g) x_tot[N + n] = x_g[n];
    mu=rep_vector(0,N_tot);
    
}

parameters {
    real<lower=0.000001> rho;
    real<lower=0.000001> alpha;
    vector[N_tot] eta;
    real a;
    real b;
}

transformed parameters{
    
    real m[N_g];
    vector[N_tot] f;
    {
        matrix[N_tot, N_tot] L_K;
        L_K = joint_cov(x_tot,alpha,rho,delta,N);
        f = mu + L_K*eta;
        
        for(i in 1:N_g){
            m[i] = a*x_g[i]+b;
        }
        
    }
}

model {    
    rho ~ inv_gamma(5,5);
    alpha ~ normal(0, 1);
    eta ~ normal(0, 1);
    a ~ normal(0,1);
    b ~ normal(0,1);
    for(i in 1:N_g)
        target+=log(1-inv_logit(m[i])-inv_logit(f[N+i]/v) + 2 * inv_logit(m[i])*inv_logit(f[N+i]/v));
}
