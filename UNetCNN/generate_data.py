# IMPORTS
import numpy as np
from scipy.ndimage import binary_dilation

def generate_sequence(size, timesteps, n_seeds, spread_prob, rng=None,
                      neighborhood_size = 3):
    """
    Returns array shape (timesteps, H, W) with binary masks (0/1).
    
    Arguments:
        size {int} -- Width and height of image.
        timesteps {int} -- No. of steps in the sequence.
        n_seeds {int} -- No. random pixels to initially set to 1.
        spread_prob {float} -- Probability that neighboring pixel is
                            set to 1 in next time-step.
        rng {np.random._generator.Generator} -- Random number generator.
                                                (Default None)
        neighborhood_size {int} -- How big the neighborhood kernel
                                   should be. (Default 3)

    Returns:
        {np.array} -- Binary mask sequences of shape (time-steps, size, size).
    """
    if rng is None: # Initialize random number generator.
        rng = np.random.default_rng()  
    masks = []  # To store all masks.

    # Get initial mask.
    mask = np.zeros((size, size), dtype=np.uint8) # Initialize all 0 mask.
    seed_coords = rng.integers(low=0, high=size,  # Get random coordinate
                               size=(n_seeds, 2)) # pairs.
    for (x, y) in seed_coords: mask[x, y] = 1     # Set them to 1.

    # Get next time-steps.
    structure = np.ones((neighborhood_size, neighborhood_size), 
                        dtype=bool) # Create a 3 by 3 kernel of 1s.
    # For each time-step ...
    for _ in range(timesteps): 
        masks.append(mask.copy()) # Add previous time-step to list.
        # Get binary masks of neighborhoods around 1s.
        dilated = binary_dilation(mask, structure=structure)
        # Let x% of neighborhood pixels where value is currently 0
        # be considered to be new growth.
        new_growth = ((dilated & (mask == 0))
                      & (rng.random(mask.shape) < spread_prob))#
        # Add growth.
        mask = mask | new_growth 

    return np.stack(masks)  # (time-steps, H, W)

def generate_dataset(n_sequences=1000, size=128, timesteps=6, seed=42,
                     neighborhood_size = 3, n_seeds_range=(1,4), 
                     spread_prob_range=(0.15, 0.45)):
    """ Generates sequences and returns tuple (X_pairs, Y_pairs).

    Each pair is (mask_t, mask_t+1) with both X_pairs and Y_pairs
    having shape (N_pairs, 1, H, W).

    Arguments:
        n_sequences {int} -- No. of sequences to generate.
        size {int} -- Width and height of image.
        timesteps {int} -- No. of steps per sequence.
        seed {int} -- Random number generator seed.
        n_seeds_range {tuple} -- Range of integers from which no. of
                                 starting pixels are to be chosen.
        spread_prob_range {tuple} -- Range of from within which spread
                                     probability is to be chosen.
        neighborhood_size {int} -- How big the neighborhood kernel
                                   should be. (Default 3)
    """
    rng = np.random.default_rng(seed) # random number generator.
    X_list = [] # Time-step t masks.
    Y_list = [] # Time-step t + 1 masks.
    # For given no. of sequences ...
    for _ in range(n_sequences):
        # Get a random no. of initial 1 pixels from given range.
        n_seeds = int(rng.integers(n_seeds_range[0], n_seeds_range[1]+1))
        # Get a random spread probability from given range.
        spread_prob = float(rng.uniform(spread_prob_range[0], spread_prob_range[1]))
        # Generate a sequence.
        seq = generate_sequence(size=size, timesteps=timesteps, rng=rng,
                                n_seeds=n_seeds, spread_prob=spread_prob,
                                neighborhood_size=neighborhood_size)
        # Create pairs (t, t + 1) from generated sequence.
        for t in range(seq.shape[0] - 1):
            X_list.append(seq[t:t+1].astype(np.float32))  # (1, H, W)
            Y_list.append(seq[t+1:t+2].astype(np.float32))
    X = np.stack(X_list)  # (N_pairs, 1, H, W)
    Y = np.stack(Y_list)
    return X, Y