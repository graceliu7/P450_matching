import ESM_functions
import torch

class IterP450(torch.utils.data.IterableDataset):
    def __init__(self, batch_size=16, split=0.9):
        self.batch_size = batch_size
        self.split_value = split
        id_list, seq_list, name_list, species_list = ESM_functions.load_data()
        self.data = ESM_functions.data_filter(id_list, seq_list, name_list, species_list)
        self.NCPR_embed, self.NCPR_df = NCPR_functions.load_NCPR()
        self.data = ESM_functions.filter_with_NCPR(self.data, self.NCPR_df)
        self.train_test_split(split)
    
    def train_test_split(self, split):
        self.train = []
        self.test = []
        self.species_dict = {}
        i=0
        np.random.shuffle(self.data)
        for prot in self.data:
            species = prot[2]
            if species in self.species_dict.values():
                self.test.append(prot)
            else:
                self.train.append(prot)
                self.species_dict[i] = species
                i+=1
        num_missing = int(split*len(self.data) - len(self.train))
        np.random.shuffle(self.test)
        for i in range(num_missing):
            self.train.append(self.test.pop())
    
    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test

    def get_species_dict(self):
        return self.species_dict
    
    def iter_train(self):
        self.train_batch = ESM_functions.batch_data(self.train, batch_size=self.batch_size)
        for batch in self.train_batch:
            batch = np.array(batch)
            batch_tokens = batch[:,1]
            batch_strs = batch[:,0]
            yield batch_tokens, batch_strs
    
    def iter_test(self):
        self.test_batch = ESM_functions.batch_data(self.test, self.batch_size)
        for batch in self.test_batch:
            batch = np.array(batch)
            batch_tokens = batch[:,1]
            batch_strs = batch[:,0]
            yield batch_tokens, batch_strs
            
    def __len__(self):
        return len(self.data)