import numpy as np
from utils.vae import vaeGenerator

def aggregateAverage(collect_SSRep):
    return np.mean(collect_SSRep, axis=0).tolist()


def aggregateVAE(collect_SSRep,original_dim_inputs=10,intermediate_dim=5,latent_dim=10,batch_size=10,epochs=20,samples_count=1000):
    original_dim_inputs = collect_SSRep.shape[1]
    intermediate_dim = np.max([5, original_dim_inputs // 2])
    latent_dim = np.max([2, intermediate_dim // 5])
    samples_count = 10 * collect_SSRep.shape[0]
    output=vaeGenerator(collect_SSRep, original_dim_inputs, intermediate_dim, latent_dim, batch_size, epochs,samples_count)
    return np.mean(output, axis=0).tolist()