"""
Conway's Game of Life simulator in numpy.
"""


import shutil
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
# import numpy as np

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
    key = jax.random.key(seed)
    key, key_init = jax.random.split(key)
    state = jax.random.randint(
        key=key_init,
        minval=0,
        maxval=2, # not included
        shape=(height, width),
        dtype=jnp.uint8,
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


@partial(jax.jit, static_argnames=['iterations'])
def simulate(
    init_state: jax.Array,    # uint8[width]
    iterations: int,
) -> jax.Array:                 # uint8[height, width]
    width, height = init_state.shape

    grid = jnp.pad(init_state, 1, mode='wrap')

    def step(_, grid):
        neighbors = (grid[ :-2,  :-2] + grid[ :-2, 1:-1] + grid[ :-2, 2:  ] +
                     grid[1:-1,  :-2]                    + grid[1:-1, 2:  ] +
                     grid[2:  ,  :-2] + grid[2:  , 1:-1] + grid[2:  , 2:  ])

        # update grid
        grid = (neighbors == 3) | (grid[1:-1, 1:-1] & (neighbors == 2))

        # wrap edges to opposite edges
        return jnp.pad(grid, 1, mode='wrap')

    grid = jax.lax.fori_loop(0, iterations, step, grid)

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
