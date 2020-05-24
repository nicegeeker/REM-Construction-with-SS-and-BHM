import os
import pickle
import matplotlib.pyplot as plt

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
STAN_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, "stan_models")
STAN_DATA_PATH = os.path.join(PROJECT_ROOT_DIR,'data')

def save_fig(fig_id, tight_layout=True, fig_extension="eps", resolution=600):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    

# 将编译好的模型，存储成pickle，供直接使用
def StanData_cache(var, data_name, **kwargs):
    path = os.path.join(STAN_DATA_PATH, data_name + '.pkl')
    with open(path,'wb') as f:
        pickle.dump(var, f)
    print("DATA cached as:" + data_name +'.pkl')
    
    
def StanData_load(data_name):
    path = os.path.join(STAN_DATA_PATH, data_name + '.pkl')
    try:
        sm = pickle.load(open(path, 'rb'))
    except:
        raise FileNotFoundError
    else:
        print("Using cached StanDATA: " + data_name)
    return sm

# 将编译好的模型，存储成pickle，供直接使用
def StanModel_cache(compiled_model, model_name, **kwargs):
    path = os.path.join(STAN_MODEL_PATH, model_name + '.pkl')
    with open(path,'wb') as f:
        pickle.dump(compiled_model, f)
    print("Model cached as:" + model_name +'.pkl')
    
    
def StanModel_load(model_name):
    path = os.path.join(STAN_MODEL_PATH, model_name + '.pkl')
    try:
        sm = pickle.load(open(path, 'rb'))
    except:
        raise FileNotFoundError
    else:
        print("Using cached StanModel:" + model_name)
    return sm