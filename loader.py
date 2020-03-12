from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class SnakeGameDataset(Dataset):

    def __init__(self, root_dir="/media/joshua/TOSHIBA EXT 2/trainingData/", transform=None, prefix="trainingDataBoards1.npy"):
        self.root_dir = root_dir
        self.transform = transform
        self.prefix = prefix

        files = []
        for i, f in enumerate(os.listdir(self.root_dir)):
            if os.path.isfile(os.path.join(self.root_dir,f)) and self.prefix in f and "trainingDataBoards400" not in f:
                files.append(os.path.join(self.root_dir, f))
            if i+1<len(os.listdir(self.root_dir)):
                if (i+1)%100==0:
                    print('[' + str(i+1) + '] ' + str(np.round(((i+1)/len(os.listdir(self.root_dir)))*100,2)) + '%', end='\r')
            else:
                print('[' + str(i+1) + '] ' + str(np.round(((i+1)/len(os.listdir(self.root_dir)))*100,2)) + '%', end='\n')

        print("Reading data from " + str(len(files)) + " files")
        # If you do not know the final size beforehand you need to
        # go through the chunks once first to check their sizes
        rows = 0
        cols = None
        dtype = None
        print("Measuring...")
        for i, data_file in enumerate(files):
            data = np.load(data_file)
            rows += data.shape[0]
            cols = data.shape[1]
            dtype = data.dtype

        print("Reading...")
        # Once the size is known create memmap and write chunks
        self.merged = np.memmap('merged.buffer', dtype=dtype, mode='w+', shape=(rows, cols))
        idx = 0
        for i, data_file in enumerate(files):
            data = np.load(data_file)
            self.merged[idx:idx + len(data)] = data
            idx += len(data)
        print("Done.\n")
        self.merged = np.memmap.reshape(self.merged, (self.merged.shape[0]//2,2,52,52))

    def __len__(self):
        return self.merged.shape[0]

    def __getitem__(self, idx):
        x = self.merged[idx,0]
        y = self.merged[idx, 1]

        sample = (np.array(np.reshape(x, (52,52))), np.array(np.reshape(y, (52,52))))

        if self.transform:
            sample = self.transform(sample)

        return sample
