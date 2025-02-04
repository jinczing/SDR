from pytorch_metric_learning.samplers.m_per_class_sampler import MPerClassSampler
import torch
from torch.utils.data.sampler import Sampler
from pytorch_metric_learning.utils import common_functions as c_f
import numpy as np

# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassSamplerDeter(MPerClassSampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000, matching_table=None):
        super(MPerClassSamplerDeter, self).__init__(labels, m, batch_size, int(length_before_new_iter))
        self.shuffled_idx_list = None
        self.matching_table = matching_table

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        if self.shuffled_idx_list is None:
            for _ in range(num_iters):

                c_f.NUMPY_RANDOM.shuffle(self.labels)
                if self.batch_size is None:
                    curr_label_set = self.labels
                else:
                    curr_label_set = self.labels[: self.batch_size // self.m_per_class]
                for label in curr_label_set:
                    t = self.labels_to_indices[label]
                    m = self.matching_table[label]
                    m_ids = np.where(m)[0]
                    if m_ids.size != 0:
                        m = np.random.choice(m_ids, 1)[0]
                        t = np.concatenate((t, self.labels_to_indices[m]))
                    idx_list[i : i + self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
                    i += self.m_per_class
            self.shuffled_idx_list = idx_list
        return iter(self.shuffled_idx_list)

class MPerClassSamplerTrain(MPerClassSampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000, matching_table=None):
        super(MPerClassSamplerTrain, self).__init__(labels, m, batch_size, int(length_before_new_iter))
        self.matching_table = matching_table

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()

        for _ in range(num_iters):

            c_f.NUMPY_RANDOM.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                m = self.matching_table[label]
                m_ids = np.where(m)[0]
                if m_ids.size != 0:
                    m = np.random.choice(m_ids, 1)[0]
                    t = np.concatenate((t, self.labels_to_indices[m]))
                idx_list[i : i + self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class

        return iter(idx_list)
