import torch
from torch.func import jacrev
from jacrev_finite import JacrevFinite

# Assert values are similiar to torch.func.jacrev
input1 = torch.randn((100,100), dtype=torch.float64)
input2 = torch.randn((100,100), dtype=torch.float64)

def function(x,y):
    return x*y

def assertTensorEqual(a, b, abs_tol=1e-9, mean_tol=1e-14):
    mean = a.sub(b).abs().mean().item()
    max = a.sub(b).abs().max().item()
    print(f"Error with actual jacrev // mean error: {mean}, max error: {max}")
    return (max<abs_tol and mean<mean_tol)

jacobian_auto = jacrev(func=function, argnums=0)(input1, input2)
jacobian_finite = JacrevFinite(function=function, num_args=0)(input1, input2)
    
print(assertTensorEqual(jacobian_auto, jacobian_finite))
