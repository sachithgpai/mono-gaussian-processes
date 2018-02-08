/*
	Posterior monotonicity
*/

functions {
    matrix joint_cov(real[] x,real alpha,real rho,real delta,int Nm,int N) {
        int N1 = N-Nm;
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
    vector[N] y;
    int<lower=1> N_g;
    real x_g[N_g];
    int<lower=1> N_prm;
    real x_prm[N_prm];
    real sig_var;
    real v;
}

transformed data {
    int<lower=1> N_tot = N + N_g + N_prm;
    vector[N_tot] mu;
    real delta=1e-4;
    real x_tot[N_tot];
    
    
    for (n in 1:N) x_tot[n] = x[n];
    for (n in 1:N_prm) x_tot[N + n] = x_prm[n];
    for (n in 1:N_g) x_tot[N+N_prm + n] = x_g[n];
    mu=rep_vector(0,N_tot);
    
}

parameters {
    real<lower=0.00001> rho;
    real<lower=0.00001> alpha;
    vector[N_tot] eta;
    real<lower=0.000001> sigma;
    real a;
    real b;
}

transformed parameters{
    real m[N_g];
    vector[N_tot] f;
    {
        matrix[N_tot, N_tot] L_K;
        L_K = joint_cov(x_tot,alpha,rho,delta,N_g,N_tot);
        f = mu + L_K*eta;
        
        for(i in 1:N_g){
            m[i] = a*x_g[i]+b;
        }
    }
}

model {    
    a ~ normal(0,1);
    b ~ normal(0,1);
    rho ~ inv_gamma(5,5);
    alpha ~ normal(0, 1);
    eta ~ normal(0, 1);
    sigma ~ normal(0,sig_var);
    y ~ normal(f[1:N], sigma);
    
    for(i in 1:N_g)
        target+= log((1-inv_logit(m[i]/v))*(1-inv_logit(f[N+N_prm+i]/v)) + inv_logit(m[i]/v)*inv_logit(f[N+N_prm+i]/v));

    
}

generated quantities{
    vector[N_prm] y_prm;
    for(n in 1:N_prm)
        y_prm[n] = normal_rng(f[N+n],sigma);
}
