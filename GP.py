import pystan
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import preprocessing
import pickle
from hashlib import md5
import GPy
import time
import os




def init():
	start_time = time.strftime("%d %b %H:%M", time.gmtime())
	global output_path
	output_path = "./{}\ Output".format(time.ctime())
	if(os.path.isdir(output_path)==False):
		os.makedirs(output_path)
    


def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    
    
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'pkl_cache/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'pkl_cache/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
    
    
################ Start of  Data section ##############################

# using generated toy data can be modified to read a file for input
    
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def func(x):
    return 1-gaussian(x,0,4) #+ gaussian(x,-5,0.4)*0.05

# can be udpated to read from a file later.
def load_data():
	global x,cnt_x
	cnt_x = 10
	x = np.linspace(-5,5,cnt_x)
	
	
	global cnt_data,xdata,ydata
	cnt_data = 7
	xdata = np.linspace(-5,1,cnt_data)
	ydata = preprocessing.scale(func(xdata))
	
	
	global cnt_g,x_g
	cnt_g = 7
	x_g = np.linspace(-5,1,cnt_g)
	
	
	
	sig  = 0.05
	v = 0.001
	
	
	global data_dict
	data_dict = dict (
		N = cnt_data,
		x = xdata,
		y = ydata,
		N_prm = cnt_x,
		x_prm = x,
		N_g = cnt_g,
		x_g = x_g,
		sig_var = sig,
		v = v,
		m_m = np.repeat(1,cnt_g)
	)
	

################# End of data section ###############################


    
def posterior():    
	model = StanModel_cache(model_code=open("./Stan files/posterior.stan").read())
	fit = model.sampling(data=data_dict,iter=1000,chains=1)
	
	val = fit.extract(permuted=True)['y_prm']
	np.save("{}/post_mean".format(output_path),np.mean(val,axis=0))
	np.save("{}/post_std".format(output_path),np.std(val,axis=0))
	np.save("{}/post_q5".format(output_path),np.percentile(val,5,axis=0))
	np.save("{}/post_q95".format(output_path),np.percentile(val,95,axis=0))
	
	
	


def posterior_unimodality():    
	model = StanModel_cache(model_code=open("./Stan files/posterior uni.stan").read())
	fit = model.sampling(data=data_dict,iter=1000,chains=1)
	
	val = fit.extract(permuted=True)['y_prm']
	np.save("{}/uni_post_mean".format(output_path),np.mean(val,axis=0))
	np.save("{}/uni_post_std".format(output_path),np.std(val,axis=0))
	np.save("{}/uni_post_q5".format(output_path),np.percentile(val,5,axis=0))
	np.save("{}/uni_post_q95".format(output_path),np.percentile(val,95,axis=0))
	
	a_b = np.array([np.mean(fit.extract(permuted=True)['a']),np.mean(fit.extract(permuted=True)['b'])])
	np.save("{}/uni_post_a_b".format(output_path),a_b)




def posterior_unimodality_support_GP():    
	model = StanModel_cache(model_code=open("./Stan files/posterior with support GP.stan").read())
	fit = model.sampling(data=data_dict,iter=1000,chains=1)
	
	val = fit.extract(permuted=True)['y_prm']
	np.save("{}/suppGP_post_mean".format(output_path),np.mean(val,axis=0))
	np.save("{}/suppGP_post_std".format(output_path),np.std(val,axis=0))
	np.save("{}/suppGP_post_q5".format(output_path),np.percentile(val,5,axis=0))
	np.save("{}/suppGP_post_q95".format(output_path),np.percentile(val,95,axis=0))
	
	
	derv = fit.extract(permuted=True)['f']
	np.save("{}/suppGP_derv_mean".format(output_path),np.mean(derv[0:,cnt_x+cnt_data:],axis=0))
	np.save("{}/suppGP_derv_std".format(output_path),np.std(derv[0:,cnt_x+cnt_data:],axis=0))
	np.save("{}/suppGP_derv_q5".format(output_path),np.percentile(derv[0:,cnt_x+cnt_data:],5,axis=0))
	np.save("{}/suppGP_derv_q95".format(output_path),np.percentile(derv[0:,cnt_x+cnt_data:],95,axis=0))
	
	param = fit.extract(permuted=True)['m']
	np.save("{}/suppGP_m_mean".format(output_path),np.mean(param[0:,:cnt_x],axis=0))
	np.save("{}/suppGP_m_std".format(output_path),np.std(param[0:,:cnt_x],axis=0))
	np.save("{}/suppGP_m_q5".format(output_path),np.percentile(param[0:,:cnt_x],5,axis=0))
	np.save("{}/suppGP_m_q95".format(output_path),np.percentile(param[0:,:cnt_x],95,axis=0))
	



def posterior_unimodality_support_GP_linear_mean():    
	model = StanModel_cache(model_code=open("./Stan files/posterior with support GP with linear mean.stan").read())
	fit = model.sampling(data=data_dict,iter=1000,chains=1)
	
	val = fit.extract(permuted=True)['y_prm']
	np.save("{}/linsuppGP_post_mean".format(output_path),np.mean(val,axis=0))
	np.save("{}/linsuppGP_post_std".format(output_path),np.std(val,axis=0))
	np.save("{}/linsuppGP_post_q5".format(output_path),np.percentile(val,5,axis=0))
	np.save("{}/linsuppGP_post_q95".format(output_path),np.percentile(val,95,axis=0))
	
	
	derv = fit.extract(permuted=True)['f']
	np.save("{}/linsuppGP_derv_mean".format(output_path),np.mean(derv[0:,cnt_x+cnt_data:],axis=0))
	np.save("{}/linsuppGP_derv_std".format(output_path),np.std(derv[0:,cnt_x+cnt_data:],axis=0))
	np.save("{}/linsuppGP_derv_q5".format(output_path),np.percentile(derv[0:,cnt_x+cnt_data:],5,axis=0))
	np.save("{}/linsuppGP_derv_q95".format(output_path),np.percentile(derv[0:,cnt_x+cnt_data:],95,axis=0))
	
	param = fit.extract(permuted=True)['m']
	np.save("{}/linsuppGP_m_mean".format(output_path),np.mean(param[0:,:cnt_x],axis=0))
	np.save("{}/linsuppGP_m_std".format(output_path),np.std(param[0:,:cnt_x],axis=0))
	np.save("{}/linsuppGP_m_q5".format(output_path),np.percentile(param[0:,:cnt_x],5,axis=0))
	np.save("{}/linsuppGP_m_q95".format(output_path),np.percentile(param[0:,:cnt_x],95,axis=0))
	
	a_b = np.array([np.mean(fit.extract(permuted=True)['a']),np.mean(fit.extract(permuted=True)['b'])])
	np.save("{}/linsuppGP_post_a_b".format(output_path),a_b)


def main():

	init()
	load_data()
	posterior()
	posterior_unimodality()
	posterior_unimodality_support_GP()
	posterior_unimodality_support_GP_linear_mean()





if __name__ == "__main__": main()
