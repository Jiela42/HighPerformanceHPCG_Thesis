from enum import Enum
from typing import List
import numpy as np
import itertools

class Shape(Enum):
    SQUARE = 1
    CROSS = 2

def get_index(coords, dimension_sizes):
    index = 0
    stride = 1
    for i, coord in enumerate(coords):
        index += coord * stride
        stride *= dimension_sizes[i]
    return int(index)

def is_within_bounds(coords, dimension_sizes):
    return all(0 <= coord < size for coord, size in zip(coords, dimension_sizes))

def generate_offsets(dimension, halo, shape):
    if shape == Shape.CROSS:
        offsets = []
        for offset in itertools.product(range(-halo, halo + 1), repeat=dimension):
            if sum(abs(x) for x in offset) <= halo:
                offsets.append(offset)
        return offsets
    elif shape == Shape.SQUARE:
        return list(itertools.product(range(-halo, halo + 1), repeat=dimension))

def store_matrix(matrix, filename):

    np.savetxt(filename, matrix, fmt="%5g")

def get_matrix_type_string(dimension, shape, halo, dimension_sizes):
    shape_str = shape.name
    halo_str = str(halo)
    dimensions_str = "x".join(map(str, dimension_sizes))

    return f"matrix_{shape_str}_halo={halo_str}_{dimensions_str}"

def generate_stencil_matrix(dimension: int, shape: Shape, halo: int, dimension_sizes: List[int]) -> np.ndarray:
    assert len(dimension_sizes) == dimension

    # make the matrix array
    matrix_size = 1

    for size in dimension_sizes:
        matrix_size *= size
    
    matrix = np.zeros((matrix_size, matrix_size))

    # generate the stencil

    offsets = generate_offsets(dimension, halo, shape)
    print(f"Offsets: {offsets}")
    num_neighbors = len(offsets) - 1 # number of neighbors excluding the center point
    print(f"Number of neighbors: {num_neighbors}")

    for coords in itertools.product(*[range(size) for size in dimension_sizes]):
        i = get_index(coords, dimension_sizes)
        for offset in offsets:
            neighbor_coords = [coord + off for coord, off in zip(coords, offset)]
            if is_within_bounds(neighbor_coords, dimension_sizes):
                j = get_index(neighbor_coords, dimension_sizes)
                # print(f"Center point: {coords}, Neighbor: {neighbor_coords}")
                # print(f"Center index: {i}, Neighbor index: {j}")
                if i == j:
                    # print(f"Center point: {coords}")
                    matrix[i, j] = num_neighbors
                else:
                    matrix[i, j] = -1.0

    import os

    # Ensure the directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "example_stencil_matrices")

    # Print the absolute path of the directory
    # print(f"Matrix will be stored in: {os.path.abspath(output_dir)}")


    # filename = get_matrix_type_string(dimension, shape, halo, dimension_sizes)

    # store_matrix(matrix, "example_stencil_matrices/" + filename + ".txt")

    return matrix
        

# first let's check out our nice 3d27pt stencil
# generate_stencil_matrix(3, Shape.SQUARE, 1, [3, 3, 3])
# generate_stencil_matrix(2, Shape.CROSS, 1, [2, 2])
# generate_stencil_matrix(2, Shape.CROSS, 1, [4, 4])
# generate_stencil_matrix(2, Shape.SQUARE, 2, [4, 4])
# generate_stencil_matrix(2, Shape.CROSS, 2, [4, 4])
# generate_stencil_matrix(4, Shape.CROSS, 1, [4, 4, 4, 4])
# generate_stencil_matrix(4, Shape.SQUARE, 1, [4, 4, 4, 4])
# generate_stencil_matrix(4, Shape.SQUARE, 2, [8, 8, 8, 8])
# generate_stencil_matrix(4, Shape.CROSS, 2, [4, 4, 4, 4])
# generate_stencil_matrix(4, Shape.CROSS, 1, [2, 2, 2, 2])