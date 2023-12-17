import abc
import random

import numpy as np

import torch
import torch.nn.utils.rnn as rnnutils

import configs.defaults as defaults

"""
Pull a batch of a given size and location from a given dataset
- dataset: the dataset in question, a list of tuples from our data processing
- batch_size: int, the size of one batch
- batch_index: int, the index of the bath to retrieve (batch #0, 1, 2, ...)
- is_multilabel: boolean, whether this dataset is multilabel or not
- num_labels: int, the number of labels in this dataset
@return: batch_tokens, batch_type_ids, batch_attn_mask, batch_labels
"""
def get_batch(dataset, batch_size, batch_index, is_multilabel, num_labels=2):
    
    batch = dataset[batch_index * batch_size: min((batch_index + 1) * batch_size, len(dataset))]

    # pad inputs with padding value from defaults
    batch_tokens = rnnutils.pad_sequence([torch.tensor(b[0]) for b in batch],
                                         padding_value=defaults.PADDING_IDX,
                                         batch_first=True)

    # pad token types with 0
    batch_token_types = rnnutils.pad_sequence([torch.tensor(b[1]) for b in batch],
                                              padding_value=0, batch_first=True)

    # pad attention mask with 0 (this being the whole point of an attention mask)
    batch_attn_mask = rnnutils.pad_sequence([torch.tensor(b[2]) for b in batch],
                                            padding_value=0, batch_first=True)

    # label tensor is a list of class indices if single-label & a "one-hot" (...k-hot?) encoding if multi-label
    if is_multilabel:
        batch_y = torch.zeros(len(batch), num_labels)

        # set the elements of the label tensor to 1 appropriately
        # this will totally ignore anything after b[3], which is fine for here
        for i, b in enumerate(batch):
            if isinstance(b[3], int):
                batch_y[i][b[3]] = 1
            else:
                for label in b[3]:
                    batch_y[i][label] = 1
    else:
        batch_y = torch.tensor([b[3][0] for b in batch])

    return batch_tokens, batch_token_types, batch_attn_mask, batch_y


"""
Pull a batch of a given size and location from a given dataset, for the Multi case
- dataset: the dataset in question, a list of tuples from our data processing
- batch_size: int, the size of one batch
- batch_index: int, the index of the bath to retrieve (batch #0, 1, 2, ...)
- is_multilabel: a list of booleans, whether each task is multilabel or not
- num_labels: a list of ints, the number of labels for each task in this dataset
@return: batch_tokens, batch_type_ids, batch_attn_mask, batch_labels (a list of tensors now)
"""
def get_batch_multitask(dataset, batch_size, batch_index, is_multilabel, num_labels):

    batch = dataset[batch_index * batch_size: min((batch_index + 1) * batch_size, len(dataset))]

    # pad inputs with padding value from defaults
    batch_tokens = rnnutils.pad_sequence([torch.tensor(b[0]) for b in batch],
                                         padding_value=defaults.PADDING_IDX,
                                         batch_first=True)

    # pad token types with 0
    batch_token_types = rnnutils.pad_sequence([torch.tensor(b[1]) for b in batch],
                                              padding_value=0, batch_first=True)

    # pad attention mask with 0 (this being the whole point of an attention mask)
    batch_attn_mask = rnnutils.pad_sequence([torch.tensor(b[2]) for b in batch],
                                            padding_value=0, batch_first=True)

    # label tensor is a list of class indices if single-label or a "one-hot" encoding if multi-label
    labels = []
    for j, im in enumerate(is_multilabel):
        if im:
            batch_y = torch.zeros(len(batch), num_labels[j])

            # set the elements of the label tensor to 1 appropriately
            # this will totally ignore anything after b[3], which is fine for here
            for i, b in enumerate(batch):
                if isinstance(b[3+j], int):
                    batch_y[i][b[3+j]] = 1
                else:
                    for label in b[3+j]:
                        batch_y[i][label] = 1
        else:
            batch_y = torch.tensor([b[3+j][0] for b in batch])

        labels.append(batch_y)

    return batch_tokens, batch_token_types, batch_attn_mask, labels

"""
A class that takes in one or more datasets and creates batches.
The batches may be shuffled or incorporate one or more datasets.
"""
class BatchGenerator:
    
    def __init__(self):
        pass

    """
    Resets any internal metrics for the next epoch, or calculates new metrics (e.g., proportions) for a new epoch
    """
    @abc.abstractmethod
    def init_epoch(self):
        pass

    """
    A generator method that returns one or more batches at a time.
    """        
    @abc.abstractmethod
    def get_batches(self):
        pass


"""
A batch generator intended to work with a single dataset.
"""
class SimpleBatchGenerator(BatchGenerator):
    
    """
    Create the batch generator
    - datasets: a list of lists of data points (so should be one dataset wrapped in an extra list)
    - batch_size: int, the desired size of one batch
    - device: the torch device to use
    - is_multilabel: boolean, whether this dataset is multilabel
    - num_labels: int, the number of labels this dataset has
    - shuffle: boolean, whether to shuffle the data (not required for dev/eval)
    """
    def __init__(self, datasets, batch_size, device, is_multilabel, num_labels, shuffle=True):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # this should be a single Boolean
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels

    """
    No-op; this generator has nothing to reset or calculate.
    """
    def init_epoch(self):
        pass

    """
    Length function
    @return: an int, the length of this batch generator
    """
    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / self.batch_size))

    """
    Generator that returns one batch at a time for one epoch and shuffles data.
    """
    def get_batches(self):
        # sort the dataset from longest to shortest
        dataset = sorted(self.datasets[0], key=lambda x: len(x[0]), reverse=True)

        batch_idxes = np.arange(0, len(self))
        if self.shuffle:
            # shuffle the batches, but be sure to put the longest batch first
            # (shuffle batches so that batches still generally have all similar-length inputs)
            batch_idxes = np.insert(np.random.permutation(batch_idxes[1:]), 0, 0)

        for i in batch_idxes:
            # yield one batch at a time, where a batch is...
            # (token_ids, token_type_ids, attention_mask, golds)
            # golds is a list of [task_1_labels, task_2_labels, ...]
            batch_tokens, batch_token_types, batch_attn_mask, batch_y = get_batch(dataset, self.batch_size, i,
                                                                                  self.is_multilabel, self.num_labels)

            yield 0, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                  batch_attn_mask.to(self.device), batch_y.to(self.device)


"""
A batch generator for multiple datasets, which alternates between batches at every step.
"""
class RoundRobinBatchGenerator(BatchGenerator):
    
    """
    Create the batch generator
    - datasets: a list of lists of data points
    - batch_size: int, the desired size of one batch
    - device: the torch device to use
    - is_multilabel: list of booleans, whether each dataset is multilabel
    - num_labels: lost of ints, the number of labels each dataset has
    - shuffle: boolean, whether to shuffle the data (not required for dev/eval)
    """
    def __init__(self, datasets, batch_size, device, is_multilabel, num_labels, shuffle=True):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # this should be a list of Booleans
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels

    """
    No-op; this generator has nothing to reset or calculate.
    """
    def init_epoch(self):
        pass

    """
    Length function
    @return: an int, the length of this batch generator
    """
    def __len__(self):
        return int(np.ceil(min([len(d) for d in self.datasets]) / self.batch_size) * len(self.datasets))

    """
    Generator that returns one batch at a time for one epoch. Shuffles data. Subsamples longer datasets.
    """
    def get_batches(self):
       
        # subsample longer datasets to match the shortest dataset exactly
        shortest_data_len = min([len(d) for d in self.datasets])
        datasets = [random.sample(dataset, shortest_data_len) for dataset in self.datasets]

        # sort them all from longest to shortest
        datasets = [sorted(dataset, key=lambda x: len(x[0]), reverse=True) for dataset in datasets]

        # create a unique batch idx permutation for each dataset if shuffling
        batch_idxes = [np.arange(0, len(self) / len(datasets)) for _ in datasets]

        if self.shuffle:
            # shuffle the batches, but be sure to put the longest batch first
            # (shuffle batches so that batches still generally have all similar-length inputs)
            batch_idxes = [np.insert(np.random.permutation(bi[1:]), 0, 0) for bi in batch_idxes]

        # for each timestep...
        for i in range(len(batch_idxes[0])):
            # for each dataset...
            for j, d in enumerate(datasets):
                # yield one batch from one dataset at a time
                batch_tokens, batch_token_types, batch_attn_mask, batch_y = get_batch(d, self.batch_size,
                                                                                      int(batch_idxes[j][i]),
                                                                                      self.is_multilabel[j],
                                                                                      self.num_labels[j])
                yield j, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                      batch_attn_mask.to(self.device), batch_y.to(self.device)

"""
A batch generator for one dataset with multiple tasks.
"""
class SimultaneousBatchGenerator(BatchGenerator):
   
    """
    Create the batch generator
    - dataset: a list of data points
    - batch_size: int, the desired size of one batch
    - device: the torch device to use
    - is_multilabel: list of booleans, whether each task is multilabel
    - num_labels: lost of ints, the number of labels each task has
    - shuffle: boolean, whether to shuffle the data (not required for dev/eval)
    """
    def __init__(self, dataset, batch_size, device, is_multilabel, num_labels, shuffle=True):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        # this should be a list of Booleans
        self.is_multilabel = is_multilabel
        self.num_labels = num_labels

    """
    No-op; this generator has nothing to reset or calculate.
    """
    def init_epoch(self):
        pass

    """
    Length function
    @return: an int, the length of this batch generator
    """
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    """
    Generator that returns one batch at a time for one epoch. Shuffles data.
    """
    def get_batches(self):
        # sort the data from longest to shortest
        dataset = sorted(self.dataset, key=lambda x: len(x[0]), reverse=True)

        # create a batch idx permutation for each dataset if shuffling
        batch_idxes = np.arange(0, len(self))

        if self.shuffle:
            # shuffle the batches, but be sure to put the longest batch first
            # (shuffle batches, not inputs, so that batches still generally have all similar-length inputs)
            batch_idxes = np.insert(np.random.permutation(batch_idxes[1:]), 0, 0)

        # for each timestep...
        for i in range(len(batch_idxes)):
            # yield one batch from the dataset at a time
            batch_tokens, batch_token_types, batch_attn_mask, batch_y = get_batch_multitask(dataset, self.batch_size,
                                                                                            int(batch_idxes[i]),
                                                                                            self.is_multilabel,
                                                                                            self.num_labels)
            yield None, batch_tokens.to(self.device), batch_token_types.to(self.device), \
                  batch_attn_mask.to(self.device), [y.to(self.device) for y in batch_y]
