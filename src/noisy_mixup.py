import numpy as np
import torch

def _noise(x, numpy_gen: np.random.Generator, torch_gen: torch.Generator, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            add_noise = add_noise_level * numpy_gen.beta(2, 5) * torch.empty(x.shape, dtype=torch.float, device="cuda").normal_(generator=torch_gen)
            #torch.clamp(add_noise, min=-(2*var), max=(2*var), out=add_noise) # clamp
            sparse = torch.empty(x.shape, dtype=torch.float, device="cuda").uniform_(generator=torch_gen)
            add_noise[sparse<sparse_level] = 0
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * numpy_gen.beta(2, 5) * (2*torch.empty(x.shape, dtype=torch.float, device="cuda").uniform_(generator=torch_gen)-1) + 1 
            sparse = torch.empty(x.shape, dtype=torch.float, device="cuda").uniform_(generator=torch_gen)
            mult_noise[sparse<sparse_level] = 1.0

            
    return mult_noise * x + add_noise      

def do_noisy_mixup(x, y, numpy_gen: np.random.Generator, torch_gen: torch.Generator, 
                   jsd=0, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    lam = numpy_gen.beta(alpha, alpha) if alpha > 0.0 else 1.0
    
    if jsd==0:
        index = torch.randperm(x.size()[0], generator=torch_gen, device="cuda")
        x = lam * x + (1 - lam) * x[index]
        x = _noise(x, numpy_gen, torch_gen, 
                   add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
    else:
        kk = 0
        q = int(x.shape[0]/3)
        index = torch.randperm(q, generator=torch_gen, device="cuda")
    
        for i in range(1,4):
            x[kk:kk+q] = lam * x[kk:kk+q] + (1 - lam) * x[kk:kk+q][index]
            x[kk:kk+q] = _noise(x[kk:kk+q], numpy_gen, torch_gen, 
                                add_noise_level=add_noise_level*i, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
            kk += q
     
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
