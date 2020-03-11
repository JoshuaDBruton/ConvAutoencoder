import numpy as np
import matplotlib.pyplot as plt

example = np.load("/media/joshua/TOSHIBA EXT 2/trainingData/trainingDataBoards400.npy")
check = np.load("/media/joshua/TOSHIBA EXT 2/trainingData/trainingDataMoves399.npy")

print(example.shape)
print(check.shape)
print(check[1])

# Currently boards are kept in sequential order in (seemingly) groups of 20,000
# We can pair them off in twos as (train, test) and then shuffle and totally
# ignore the trainingDataMoves*.npy files
