"""
Elementary cellular automata simulator in numpy.
"""


import itertools
import pathlib
import time
from typing import Literal

import numpy as np
from PIL import Image
import tqdm


def main(
    rule: int = 110,
    width: int = 32,
    height: int = 32,
    init: Literal["random", "middle"] = "middle",
    seed: int = 42,
    animate: bool = True,
    fps: None | float = None,
    save_image: None | pathlib.Path = None,
    upscale: int = 1,
):
    print(f"rule: {rule}")
    print(f"bits: {rule:08b}")
    print("Wolfram table:")
    print(" 1 1 1   1 1 0   1 0 1   1 0 0   0 1 1   0 1 0   0 0 1   0 0 0")
    print("   " + "       ".join(f'{rule:08b}'))

    print("initialising state...")
    match init:
        case "middle":
            state = np.zeros(width, dtype=np.uint8)
            state[width//2] = 1
        case "random":
            np.random.seed(seed)
            state = np.random.randint(
                low=0,
                high=2, # not included
                size=(width,),
                dtype=np.uint8,
            )
    print("initial state:", state)

    print("simulating automaton...")
    start_time = time.perf_counter()
    history = simulate(
        rule=rule,
        init_state=state,
        height=height,
    )
    end_time = time.perf_counter()
    print("simulation complete!")
    print("result shape", history.shape)
    print(f"time taken {end_time - start_time:.4f} seconds")

    if animate:
        print("rendering...")
        for row in history:
            print(''.join(["█░"[s]*2 for s in row]))
            if fps is not None: time.sleep(1/fps)

    if save_image is not None:
        print("rendering to", save_image, "...")
        history_greyscale = 255 * (1-history)
        history_upscaled = (history_greyscale
            .repeat(upscale, axis=0)
            .repeat(upscale, axis=1)
        )
        Image.fromarray(history_upscaled).save(save_image)


def simulate(
    rule: int,
    init_state: np.typing.ArrayLike,    # uint8[width]
    height: int,
) -> np.typing.NDArray:                 # uint8[height, width]
    # parse rule
    rule_uint8 = np.uint8(rule)
    rule_bits = np.unpackbits(rule_uint8, bitorder='little')
    rule_table = rule_bits.reshape(2,2,2)

    # parse initial state
    init_state = np.asarray(init_state, dtype=np.uint8)
    (width,) = init_state.shape

    # accumulate output into this array
    # extra width is to implement wraparound with slicing
    history = np.zeros((height, width+2), dtype=np.uint8)

    # first row
    history[0, 1:-1] = init_state
    history[0, 0] = init_state[-1]
    history[0, -1] = init_state[0]

    # remaining rows
    for step in tqdm.trange(1, height):
        # apply rules
        history[step, 1:-1] = rule_table[
            history[step-1, 0:-2],
            history[step-1, 1:-1],
            history[step-1, 2:],
        ]
        # sync edges
        history[step, 0] = history[step, -2]
        history[step, -1] = history[step, 1]

    # return a view of the array without the width padding
    return history[:, 1:-1]


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
