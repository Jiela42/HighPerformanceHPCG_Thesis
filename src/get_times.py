from gpu_timing import gpu_timer
import computeCG
import generations
import torch

num_iterations = 10

sizes =[
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
]

methods = [
    "computeSYMGS",
    "computeSPMV",
    "computeRestriction",
    "computeMG",
    "computeProlongation",
    "computeCG",
]

matrix_types = [
    "3d_27pt"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################################################################
# for each version we need the following
# because each version has it's own name and needs to be called!
#################################################################################################################

for size in sizes:
    for matrix_type in matrix_types:

        timer = gpu_timer(
            version_name = "BaseTorch",
            ault_node = "41-44",
            matrix_type = matrix_type,
            nx = size[0],
            ny = size[1],
            nz = size[2]
        )

        # note: CG and MG include generations of matrices!
        # Also note: in the original HPCG the generations are all done in the main function, non of them in the MG routine.
        
        if "computeCG" in methods:

            for i in range(num_iterations):

                timer.start_CG_timer()
                # computeCG.computeCG(size[0], size[1], size[2])
                timer.stop_CG_timer()
        
        if "computeMG" in methods:

            # initialize the matrix and vectors (not included in measurments)
            A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
            x = torch.zeros(size[0]*size[1]*size[2], device=device)

            for i in range(num_iterations):

                timer.start_CG_timer()
                computeCG.computeMG(size[0], size[1], size[2], A, y, x, 0)
                timer.stop_CG_timer()
        
        if "computeSYMGS" in methods:

            A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
            x = torch.zeros(size[0]*size[1]*size[2], device=device)

            for i in range(num_iterations):

                timer.start_CG_timer()
                computeCG.computeSYMGS(size[0], size[1], size[2], A, y, x)
                timer.stop_CG_timer()
        
        if "computeSPMV" in methods:
                
                # careful! This way we only get "nice" numbers for the vectors, which does not reflect how spmv is used in the routines
                A,y = generations.generate_torch_coo_problem(size[0], size[1], size[2])
                x = torch.zeros(size[0]*size[1]*size[2], device=device)
    
                for i in range(num_iterations):
    
                    timer.start_CG_timer()
                    computeCG.computeSPMV(size[0], size[1], size[2], A, y, x)
                    timer.stop_CG_timer()

        timer.destroy_timers()


#################################################################################################################