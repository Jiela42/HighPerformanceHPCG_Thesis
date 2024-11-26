################################################
# The following one may want to change
developer_mode = True
print_elem_not_found_warnings = False
do_tests = True
limit_matrix_size = True
limit_matrix_size_for_cg = True
max_dim_cg = 64
max_dim_size = 64
ault_node = "41-44"
error_tolerance = 1e-9
cg_error_tolerance = 1e-5
num_bench_iterations = 10
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