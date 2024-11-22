import torch
import torch.linalg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def computeWAXPBY(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    solution = alpha * x + beta * y
    w.copy_(solution)
    return solution

def computeDot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.dot(a, b)

def computeSPMV(nx: int, ny:int , nz:int, A: torch.sparse.Tensor, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
    x = x.unsqueeze(1) if x.dim() == 1 else x
    solution = torch.sparse.mm(A, x)
    solution = solution.squeeze() if solution.dim() > 1 else solution
    y.copy_(solution)
    return solution

def greg_symGS(A: torch.sparse.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    dense_A = A.to_dense()
    y_copy = y.clone()
    
    y_copy = y.unsqueeze(1) if y.dim() == 1 else y

    # Get the lower triangular part (including diagonal)
    LA = torch.tril(dense_A)
    # Get the upper triangular part (including diagonal)
    UA = torch.triu(dense_A)
    # Get the diagonal
    DA = torch.diag(torch.diag(dense_A))    
    # Solve LA * x = y
    x, _ = torch.triangular_solve(input = y_copy, A = LA, upper=False)    
    # Compute x1 = y - LA*x + DA*x
    x1 = y_copy - torch.matmul(LA, x) + torch.matmul(DA, x)

    # Solve UA * x = x1
    x, _ = torch.triangular_solve(A= UA, input = x1, upper=True)    
    return x

def computeSymGS(nx: int, ny: int, nz: int, A: torch.sparse.Tensor, r: torch.Tensor, x_return: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Gauss-Seidel preconditioner.
    
    Parameters:
    A (torch.sparse.Tensor): Sparse matrix A
    r (torch.Tensor): Vector r
    
    Returns:
    torch.Tensor: Solution vector x
    """

    # Extract the lower triangular part of A
    LA_indices = A._indices()[:, A._indices()[0] >= A._indices()[1]]
    LA_values = A._values()[A._indices()[0] >= A._indices()[1]]
    LA = torch.sparse_coo_tensor(LA_indices, LA_values, A.size(), device=device)

    # Extract the upper triangular part of A
    UA_indices = A._indices()[:, A._indices()[0] <= A._indices()[1]]
    UA_values = A._values()[A._indices()[0] <= A._indices()[1]]
    UA = torch.sparse_coo_tensor(UA_indices, UA_values, A.size(), device=device)

    # Extract the diagonal part of A
    DA_indices = torch.stack([torch.arange(A.size(0)), torch.arange(A.size(0))])
    DA_values = A._values()[A._indices()[0] == A._indices()[1]]
    DA = torch.sparse_coo_tensor(DA_indices, DA_values, A.size(), device=device)

    # Forward solve LA * x = r
    x = torch.zeros_like(r)
    for i in range(A.size(0)):
        x[i] = (r[i] - torch.sparse.mm(LA, x.unsqueeze(1)).squeeze()[i]) / DA_values[i]

    # Compute x1 = r - LA * x + DA * x
    x1 = r - torch.sparse.mm(LA, x.unsqueeze(1)).squeeze() + torch.sparse.mm(DA, x.unsqueeze(1)).squeeze()

    # Backward solve UA * x = x1
    x = torch.zeros_like(r)
    for i in range(A.size(0) - 1, -1, -1):
        x[i] = (x1[i] - torch.sparse.mm(UA, x.unsqueeze(1)).squeeze()[i]) / DA_values[i]

    x_return.copy_(x)

    return x

# we pass the nx, ny, nz, precondition as arguments to the function just to have the same parameters for all the versions
# this will need to change if we ever have to adapt the code for other matrix types
def computeCG(nx, ny, nz, A: torch.sparse.Tensor, b: torch.Tensor, x: torch.Tensor, precondition: bool) -> torch.Tensor:

    p = x.clone()

    p = p.unsqueeze(1) if p.dim() == 1 else p
    b = b.unsqueeze(1) if b.dim() == 1 else b

    Ap = torch.sparse.mm(A, p)
    r = b - Ap
    r = r.squeeze(1) if r.dim() > 1 else r
    normr = torch.sqrt(torch.dot(r,r))

    iter = 0
    maxiters = 100
    dummy_tensor = torch.zeros_like(x)

    while normr > 1e-16 and iter < maxiters:
        iter += 1
        z = computeSymGS(nx, ny, nz, A, r, dummy_tensor)
        # z = z.unsqueeze(1) if z.dim() == 1 else z
        if iter == 1:
            p = z
            rtz = torch.dot(r,z)
        else:
            oldrtz = rtz
            rtz = torch.dot(r,z)
            beta = rtz/oldrtz
            p = beta*p + z

        p = p.unsqueeze(1) if p.dim() == 1 else p
        Ap = torch.sparse.mm(A, p)
        p = p.squeeze(1) if p.dim() > 1 else p
        Ap = Ap.squeeze(1) if Ap.dim() > 1 else Ap
        pAp = torch.dot(p, Ap)
        alpha = rtz/pAp
        x.add_(alpha*p)
        r.sub_(alpha*Ap)

        normr = torch.sqrt(torch.dot(r,r)).item()
    
def main (A: torch.sparse.Tensor, b: torch.Tensor, x: torch.Tensor, x_exact: torch.Tensor) -> torch.Tensor:

    normr_exact = torch.norm(b - torch.sparse.mm(A, x_exact))
    normr_initial = torch.norm(b - torch.sparse.mm(A, x))

    x_computed = computeCG(A, b, x)

    return x_computed
