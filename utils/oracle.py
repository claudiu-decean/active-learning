from torch.utils.data import Dataset

class Oracle(Dataset):
    def __init__(self, base_dataset, budget=None):
        self.base_dataset = base_dataset
        self.is_labelled = {i: False for i in range(len(base_dataset))}
        self.labelled_data = []
        if budget is None:
            self.budget = len(base_dataset)
        else:
            self.budget = budget


    def __getitem__(self, index):
        return self.base_dataset[self.labelled_data[index]]

    def __len__(self):
        return len(self.labelled_data)
    
    def label(self, data_to_label):
        for base_index in data_to_label:
            if not self.is_labelled[base_index] and self.get_labelled_data_num() < self.budget:
                self.is_labelled[base_index] = True
                self.labelled_data.append(base_index)
    
    def get_total_data_num(self):
        return len(self.base_dataset)
    
    def get_labelled_data_num(self):
        return len(self.labelled_data)
    
    def get_unlabelled_data_num(self):
        return self.get_total_data_num() - self.get_labelled_data_num()
    
    def get_image(self, index):
        return self.base_dataset[index][0]