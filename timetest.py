import torch
import timeit
from torch.func import jacrev
from JacrevFinite import JacrevFinite

# Function 1
def two(x):
    return 2 * x

input_two = torch.tensor((5, 2), dtype=torch.float64).requires_grad_()

time_jacrev_two = timeit.timeit(lambda: jacrev(two, argnums=0)(input_two), number=100)
time_jacrev_finite_two = timeit.timeit(lambda: JacrevFinite(function=two, num_args=0)(input_two), number=100)
time_grad_two = timeit.timeit(lambda: torch.autograd.grad(two(input_two).sum(), input_two, create_graph=True), number=100)


# Function 2
def f(x, y):
    return x + y

input_f1 = torch.tensor((1, 1), dtype=torch.float64).requires_grad_()
input_f2 = torch.tensor((2, 3), dtype=torch.float64).requires_grad_()

time_jacrev_f = timeit.timeit(lambda: jacrev(f, argnums=1)(input_f1, input_f2), number=100)
time_jacrev_finite_f = timeit.timeit(lambda: JacrevFinite(function=f, num_args=1, delta=1e-10)(input_f1, input_f2), number=100)
time_grad_f = timeit.timeit(lambda: torch.autograd.grad(f(input_f1, input_f2).sum(), input_f2, create_graph=True), number=100)


# Function 3
def g(x, y):
    return x * y

seq1a = [1.0, 2.0, 3.0, 4.0]
seq1b = [5.0, 6.0, 7.0, 8.0]
seq2a = [-1.0, -2.0, -3.0, -4.0]
seq2b = [-5.0, -6.0, -7.0, -8.0]

def wrapper(seq1a, seq1b, seq2a, seq2b):
    seq1 = seq1a + seq1b  # Concatenate
    seq2 = seq2a + seq2b
    return [seq1, seq2]

time_jacrev_g = timeit.timeit(lambda: jacrev(g, argnums=1)(torch.tensor(seq1a + seq1b, dtype=torch.float32), torch.tensor(seq2a + seq2b, dtype=torch.float32)), number=100)
time_jacrev_finite_g = timeit.timeit(lambda: JacrevFinite(function=g, wrapper=wrapper, num_args=3)(seq1a, seq1b, seq2a, seq2b), number=100)

# Print the results
print(f"Time for jacrev (two): {time_jacrev_two:.6f} seconds")
print(f"Time for grad (two): {time_grad_two:.6f} seconds")
print(f"Time for JacrevFinite (two): {time_jacrev_finite_two:.6f} seconds\n")

print(f"Time for jacrev (f): {time_jacrev_f:.6f} seconds")
print(f"Time for grad (f): {time_grad_f:.6f} seconds")
print(f"Time for JacrevFinite (f): {time_jacrev_finite_f:.6f} seconds\n")

print(f"Time for jacrev (g with wrapper): {time_jacrev_g:.6f} seconds")
print(f"Time for JacrevFinite (g with wrapper): {time_jacrev_finite_g:.6f} seconds")

