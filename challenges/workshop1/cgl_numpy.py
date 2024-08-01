"""
Conway's Game of Life simulator in numpy.
"""


import shutil
import sys
import time

import numpy as np

COLS, ROWS = shutil.get_terminal_size()


def main(
    width: int = COLS,
    height: int = 2 * ROWS // 3,
    iterations: int = 1000,
    seed: int = 10,
    display: bool = True,
):
    width *= 2
    height *= 4
    print(f"size: {width}x{height}")

    print("initialising state...")
    np.random.seed(seed)
    state = np.random.randint(
        low=0,
        high=2, # not included
        size=(height, width),
        dtype=np.uint8,
    )

    print("simulating automaton...")
    start_time = time.perf_counter()
    result = simulate(
        init_state=state,
        iterations=iterations,
    )
    end_time = time.perf_counter()
    print("simulation complete!")
    print("result shape", result.shape)
    print(f"time taken {end_time - start_time:.4f} seconds")

    if display: print_as_braille(result)


def simulate(
    init_state: np.typing.ArrayLike,    # uint8[width]
    iterations: int,
) -> np.typing.NDArray:                 # uint8[height, width]
    width, height = init_state.shape

    grid = np.zeros((width+2, height+2), dtype=np.uint8)
    grid[1:-1, 1:-1] = init_state
    neighbors = np.zeros((width+2, height+2), dtype=np.uint8)
    twos = np.zeros((width+2, height+2), dtype=np.bool)
    threes = np.zeros((width+2, height+2), dtype=np.bool)

    for _ in range(iterations):
        # copy edges to opposite edges so that we can wrap around
        grid[-1, :] = grid[1, :]
        grid[:, -1] = grid[:, 1]
        grid[0, :] = grid[-2, :]
        grid[:, 0] = grid[:, -2]
        # and corners
        grid[0, 0] = grid[-2, -2]
        grid[-1, -1] = grid[1, 1]
        grid[0, -1] = grid[-2, 1]
        grid[-1, 0] = grid[1, -2]

        # update neighbors array
        neighbors[1:-1, 1:-1]  = grid[ :-2,  :-2]
        neighbors[1:-1, 1:-1] += grid[ :-2, 1:-1]
        neighbors[1:-1, 1:-1] += grid[ :-2, 2:  ]
        neighbors[1:-1, 1:-1] += grid[1:-1,  :-2]
        neighbors[1:-1, 1:-1] += grid[1:-1, 2:  ]
        neighbors[1:-1, 1:-1] += grid[2:  ,  :-2]
        neighbors[1:-1, 1:-1] += grid[2:  , 1:-1]
        neighbors[1:-1, 1:-1] += grid[2:  , 2:  ]

        # update grid
        np.equal(neighbors, 2, out=twos)
        np.equal(neighbors, 3, out=threes)
        grid &= twos
        grid |= threes

    return grid[1:-1, 1:-1]


def print_as_braille(a):
    H, W = a.shape
    h, w = (H // 4, W // 2)
    c = (   a
            .reshape(h, 4, w, 2)     # split rows into 4s and cols into 2s
            .transpose([1, 3, 0, 2]) # put the 4x2s in the first two dims
            .reshape(8, h, w)        # collapse them into one dimension
        )
    # pack the numbers into an array of bytes
    b = ( c[0]      | c[1] << 3
        | c[2] << 1 | c[3] << 4
        | c[4] << 2 | c[5] << 5
        | c[6] << 6 | c[7] << 7
        )

    for i in range(h):
        for j in range(w):
            ch = chr(0x2800+int(b[i, j]))
            sys.stdout.write(ch)
        sys.stdout.write("\n")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
