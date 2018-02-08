/*

'f' used to represent function and 'g' used to as the derivative values and 'm' is the values of support GP

    x      = data_input
    x_g    = psuedo points in input space for derivative evaluation
    m_m    = slope hyperparamter for support GP generating sign of derivative 'm'
    
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
    int m_m[N_g];
}

transformed data {
    int<lower=1> N_tot = N + N_prm + N_g;
    vector[N_tot] mu;

    real delta=1e-4;
    real x_tot[N_tot];
    real x_g_tot[N_g+N_g];
    
    
    for (n in 1:N) x_tot[n] = x[n];
    for (n in 1:N_prm) x_tot[N + n] = x_prm[n];
    for (n in 1:N_g) x_tot[N+N_prm + n] = x_g[n];
    mu=rep_vector(0,N_tot);
    

    for (n in 1:N_g) x_g_tot[n] = x_g[n];
    for (n in 1:N_g) x_g_tot[N_g + n] = x_g[n];
    
}

parameters {
    real<lower=0.00001> rho;
    real<lower=0.00001> alpha;
    real<lower=0.00001> rho_g;
    real<lower=0.00001> alpha_g;
    vector[N_tot] eta;
    vector[N_g+N_g] eta_g;   
    real<lower=0.000001> sigma;
    real a;
    real b;
    
}

transformed parameters{
    vector[N_g+N_g] m;
    vector[N_tot] f;
	vector[N_g+N_g] mu_m;
    {
        matrix[N_tot, N_tot] L_K;
        matrix[(N_g+N_g),(N_g+N_g)] L_K_m;
        
        L_K = joint_cov(x_tot,alpha,rho,delta,N_g,N_tot);
        f = mu + L_K*eta;
        
        
        mu_m=rep_vector(0,N_g+N_g);
        for(i in 1:N_g){
            mu_m[i] = a*x_g[i]+b;
        }
        
        L_K_m = joint_cov(x_g_tot,alpha_g,rho_g,delta,N_g,N_g+N_g);
        m = mu_m + L_K_m*eta_g;
        
    }
}

model {    
	a ~ normal(0,1);
    b ~ normal(0,1);

    rho ~ inv_gamma(5,5);
    alpha ~ normal(0, 1);
    eta ~ normal(0, 1);
    
    rho_g ~ inv_gamma(5,5);
    alpha_g ~ normal(0, 1);
    eta_g ~ normal(0, 1);
    sigma ~ normal(0,sig_var);
    
    y ~ normal(f[1:N], sigma);
    
    
    for(i in 1:N_g){
        target+= log((1-inv_logit(m[i]/v))*(1-inv_logit(f[N+N_prm+i]/v)) + inv_logit(m[i]/v)*inv_logit(f[N+N_prm+i]/v));
        target+= log((1-inv_logit(m_m[i]/v))*(1-inv_logit(m[N_g+i]/v)) + inv_logit(m_m[i]/v)*inv_logit(m[N_g+i]/v));
    }
    
}

generated quantities{
    vector[N_prm] y_prm;
    for(n in 1:N_prm)
        y_prm[n] = normal_rng(f[N+n],sigma);
}
