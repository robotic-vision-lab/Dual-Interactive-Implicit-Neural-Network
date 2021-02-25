import os
import torch

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

#get all directories of the data
#data is store as
#train/
#    170410-001-m-r9iu-df44-low-res-result/
#      170410-001-m-r9iu-df44-low-res-result_normalized.npz
#      landmarks3d.txt
def get_ids(dir, num=None):
    partition = {}
    partition['train'] = []
    partition['test'] = []
    for root, directories, files in os.walk(os.path.join(dir,'train')):
        for filename in files:
            if filename.lower().endswith('.npz'):
                partition['train'].append(os.path.join(root,filename))
                #print(os.path.join(root,filename))
    for root, directories, files in os.walk(os.path.join(dir,'test')):
        for filename in files:
            if filename.lower().endswith('.npz'):
                partition['test'].append(os.path.join(root,filename))
    return partition

class SRDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dir=os.path.join('.','data')):
        '''
        Inputs: dir(str) - directory to the data folder
        '''
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        partial_scan, full_scan = get_scans(ID)
        
        return torch.from_numpy(partial_scan), torch.from_numpy(full_scan)

if __name__ == '__main__':
    print(list(os.walk(os.path.join('.','data'))))
    print((os.path.abspath(os.path.curdir)))