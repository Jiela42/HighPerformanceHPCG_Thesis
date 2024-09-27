import torch

version_name = None
ault_node = None
matrix_type = None

def init_timer(version_name, ault_node, matrix_type):

    version_name = version_name
    ault_node = ault_node
    matrix_type = matrix_type


def start_CG_timer():

    global CG_start_event
    global CG_end_event
    global elapsed_CG_time_ms

    if CG_start_event is not None:
        raise ValueError("Timer already started")

    CG_start_event = torch.cuda.Event(enable_timing=True)
    CG_start_event.record()

def stop_CG_timer():
    
    if CG_start_event is None:
        raise ValueError("Timer not started")
    
    CG_end_event.record()
    torch.cuda.synchronize()

    elapsed_CG_time_ms.append(CG_start_event.elapsed_time(CG_end_event))

def destroy_CG_timer():
    
    # write the results to a file
