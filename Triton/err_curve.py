##
##	File which computed the rmse error of each of the four models given a data set
##  Command line input:
##		1. Ouput file path
##		2. Number of data points to select from the curve
## 		3. Number of the sampling.


import pystan
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
import pickle
from hashlib import md5
import GPy
import time
import os
import sys


names = ['norm','uni','suppGP','suppGPlin']

def init(path,cnt,sampl):
	global ofile
	global output_path
	output_path = "./%s/"%(path)
	if(os.path.isdir(output_path)==False):
		os.makedirs(output_path)
	
	ofile = open(output_path+'%d_%d.txt'%(cnt,sampl),'w')

    
    
def deinit():
	ofile.close()


def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    
    
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = '../Stan files/pkl_cache/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = '../Stan files/pkl_cache/cached-{}-{}.pkl'.format(model_name, code_hash)
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

# can be udpated to read from a file later.
def load_data(model_type):
	global xdata,cnt_x,f,ydata#,idx
	xdata = np.load('../x40.npy')
	cnt_x = len(xdata)
	f = np.load('../f40.npy')
	ydata = np.load('../y40.npy')
	
	
	## Reads the stan models##
	global gp_model
	if(model_type==0):
		gp_model = StanModel_cache(model_code=open("../Stan files/posterior.stan").read())
	elif(model_type==1):
		gp_model = StanModel_cache(model_code=open("../Stan files/posterior uni.stan").read())
	elif(model_type==2):
		gp_model = StanModel_cache(model_code=open("../Stan files/posterior with support GP.stan").read())
	elif(model_type==3):
		gp_model = StanModel_cache(model_code=open("../Stan files/posterior with support GP with linear mean.stan").read())
	else:
		print('Error: Model_type unknown!!')
		sys.exit()
	
	
	

################# End of data section ###############################



def evaluate(data_dict):
	it = 0
	
	while (it<100):					# Hoping to restart the sim if initialization fails, will make 100 tries
		try:
			fit = gp_model.sampling(data=data_dict,iter=2000,chains=2)
			break
		except RuntimeError:
			print('INIT_FAILED')
			pass
		it = it+1
	
	if(it == 100):
		return -1
	pred = np.mean(fit.extract(permuted=True)['y_prm'],axis=0)
	return metrics.mean_squared_error(pred,f)





def rmse_error_curve(cnt,sampl_no):
    
	sig  = 0.05
	v = 0.001
	idx = np.array(range(cnt_x))
	for i in range(sampl_no):
		np.random.shuffle(idx)
	
	data_dict = dict (
		        N = cnt,
		        x = xdata[idx[0:cnt]],
		        y = ydata[idx[0:cnt]],
		        N_prm = cnt_x,
		        x_prm = xdata,
		        N_g = cnt_x,
		        x_g = xdata,
		        sig_var = sig,
		        v = v,
		        m_m = np.repeat(1,cnt_x)
		    )
    
	ret=evaluate(data_dict)
	return ret
    
#	ofile.write(str(evaluate(post_model,data_dict))+'\n')
#	ofile.write(str(evaluate(uni_model,data_dict))+'\n')
#	ofile.write(str(evaluate(suppGP_model,data_dict))+'\n')
#	ofile.write(str(evaluate(suppGPlin_model,data_dict))+'\n')



def main():
#	path = sys.argv[1]							# Folder to store the files
	cnt = int(sys.argv[1])						# Number of data points to select from curve
	sampl_no = int(int(sys.argv[2])/4) +1		# sampling round number
	model_type = int(sys.argv[2])%4				# for which model is the current simulation going to run.
	rand_seed  = int(sys.argv[3])
	
	print('\n\n\n#######################################\n\n\n')
	print(cnt,sampl_no,model_type)
	np.random.seed(rand_seed)
	ofile = open('./%s_%d_%d.txt'%(names[model_type],cnt,sampl_no),'w')
	
	

	load_data(model_type)
	res = rmse_error_curve(cnt,sampl_no)
	print('RESULT : ',res)
	ofile.write(str(res)+'\n')
	
	ofile.close()	
	print('\n\n\n#######################################\n\n\n')


if __name__ == "__main__": main()
