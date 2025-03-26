################################################
# The following one may want to change
developer_mode = True
print_elem_not_found_warnings = False
do_tests = False
limit_matrix_size = True
limit_matrix_size_for_cg = True
max_dim_cg = 64
max_dim_size = 64
ault_node = "GH200"
error_tolerance = 1e-5
cg_error_tolerance = 1e-5
num_bench_iterations = 10
################################################
cuda_max_blocks = 65535

################################################

import torch
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this function helps debugging
def print_differeing_vectors(tested: torch.Tensor, control: torch.Tensor, num_outputs: Optional [int]) -> None:

    if num_outputs is None:
        num_outputs = tested.size(0)
    output_ctr = 0

    tested = tested.squeeze() if tested.dim() > 1 else tested
    control = control.squeeze() if control.dim() > 1 else control

    if tested.dtype != control.dtype:
        raise AssertionError(f"Dtype of the vectors is different: {tested.dtype} vs {control.dtype}")

    if tested.size(0) != control.size(0):
        raise AssertionError(f"Size of the vectors is different: {tested.size(0)} vs {control.size(0)}")

    # for i in range(tested.size(0)):
    #     print(f"i: {i} should be: {control[i].item()} but was: {tested[i]}")

    # print the wrong elements next to each other
    print(f"control size: {control.size()}")
    print(f"tested size: {tested.size()}")
    for i in range(tested.size(0)):
        if not torch.isclose(tested[i], control[i],atol=error_tolerance):
            print(f"i: {i} should be: {control[i]} but was: {tested[i]}")
            output_ctr += 1
            if output_ctr >= num_outputs:
                break

def read_dimension(file_path):
    dimensions = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.split(':')
                dimensions[key.strip()] = int(value.strip())
    return dimensions

def getSymGS_rrNorm(nx, ny, nz):
    # The relative residual norm for various grid sizes
    if nx == 2 and ny == 2 and nz == 2:
        return 0.01780163474925223
    elif nx == 4 and ny == 4 and nz == 4:
        return 0.06265609442727489
    elif nx == 8 and ny == 8 and nz == 8:
        return 0.06901259891475327
    elif nx == 16 and ny == 16 and nz == 16:
        return 0.1093031516812343
    elif nx == 24 and ny == 24 and nz == 24:
        return 0.161818728347593
    elif nx == 32 and ny == 32 and nz == 32:
        return 0.2059749871712245
    elif nx == 64 and ny == 64 and nz == 64:
        return 0.3318764588131313
    elif nx == 128 and ny == 64 and nz == 64:
        return 0.3686739441774081
    elif nx == 128 and ny == 128 and nz == 64:
        return 0.415239353783707
    elif nx == 128 and ny == 128 and nz == 128:
        return 0.4761695919404465
    elif nx == 256 and ny == 128 and nz == 128:
        return 0.5161210410022423
    elif nx == 256 and ny == 256 and nz == 128:
        return 0.5649574838245627
    else:
        print(f"The relative residual norm is not implemented for the size {nx}x{ny}x{nz}")
        print("Please add the size run_get_Norm in the testing lib and run it to obtain the relative residual norm")
        print("then add the obtained value to the getSymGS_rrNorm function")
        print("and re-run the benchmark")
        raise ValueError("Relative residual norm not implemented for the given size")

def getSymGS_rrNorm_zero_based(nx, ny, nz):
    if nx == 2 and ny == 2 and nz == 2:
        return 0.03859589326112207
    elif nx == 4 and ny == 4 and nz == 4:
        return 0.1627214749610502
    elif nx == 8 and ny == 8 and nz == 8:
        return 0.1878644061539017
    elif nx == 16 and ny == 16 and nz == 16:
        return 0.1868789912880421
    elif nx == 32 and ny == 32 and nz == 32:
        return 0.2386720453340627
    elif nx == 64 and ny == 64 and nz == 64:
        return 0.3411755350583427
    elif nx == 128 and ny == 64 and nz == 64:
        return 0.375064237277454
    elif nx == 128 and ny == 128 and nz == 64:
        return 0.4186180854459645
    elif nx == 128 and ny == 128 and nz == 128:
        return 0.47348419196090
    elif nx == 256 and ny == 128 and nz == 128:
        return 0.51171735634656
    elif nx == 256 and ny == 256 and nz == 128:
        return 0.5586609181757166
    else:
        print(f"The relative residual norm is not implemented for the size {nx}x{ny}x{nz}")
        print("Please add the size run_get_Norm in the testing lib and run it to obtain the relative residual norm")
        print("then add the obtained value to the getSymGS_rrNorm function")
        print("and re-run the benchmark")
        raise ValueError("Relative residual norm not implemented for the given size")
    
num_its_zerobased_AMGX = {
    (2,2,2): 1,
    (4,4,4): 2,
    (8,8,8): 2,
    (16,16,16): 2,
    (32,32,32): 2,
    (64,64,64): 2,
    (128,64,64): 3
}
