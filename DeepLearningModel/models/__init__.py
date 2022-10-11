from models.AE_Model import AutoEncoder as AE_Model
from models.VAE_Model import VAE as VAE_Model
from models.Bert_Model import BERT as Bert_Model
def load_model(type: str):
    if type=='AE':
        return AE_Model
    elif type=='VAE':
        return VAE_Model
    elif type == 'BERT':
        return Bert_Model
    else:
        raise Exception('Select the correct model type. Currently supporting only T5 and GPT2.')