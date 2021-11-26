import torch

from pytorch_metric_learning import miners, losses, reducers
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.miners.base_miner import BaseTupleMiner
from pytorch_metric_learning.utils import common_functions as c_f

def get_all_pairs_indices(labels, ref_labels=None, matching_table=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    if matching_table is None:
        matches = (labels1 == labels2).byte()
    else:
        # print(matching_table.shape)
        t1, t2 = torch.meshgrid(labels1.squeeze(), labels2.squeeze())
        inds = torch.cat([t1.unsqueeze(-1), t2.unsqueeze(-1)], dim=-1).view(-1, 2)
        inds = inds[:, 0]*matching_table.size(0) + inds[:, 1]
        matching_table = matching_table.flatten()
        matches = matching_table[inds].byte()
        # print(matches.sum())
        matches = matches.view(labels.size(0), labels.size(0))
        labels1 = labels.unsqueeze(1)
        labels2 = ref_labels.unsqueeze(0)
        matches = (matches.bool() | (labels1 == labels2)).byte()
        # print(matches)
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx

def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n

def pos_pairs_from_tuple(indices_tuple):
    return indices_tuple[:2]

def neg_pairs_from_tuple(indices_tuple):
    return indices_tuple[2:]

class MultiSimilarityMinerWithMatchingTable(miners.MultiSimilarityMiner):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(epsilon, **kwargs)

        # self.matching_table = kwargs['matching_table']

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None, matching_table=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = self.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels, matching_table)
        self.output_assertion(mining_output)
        return mining_output

    def mine(self, embeddings, labels, ref_emb, ref_labels, matching_table):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = get_all_pairs_indices(labels, ref_labels, matching_table)

        if len(a1) == 0 or len(a2) == 0:
            empty = torch.tensor([], device=labels.device, dtype=torch.long)
            return empty.clone(), empty.clone(), empty.clone(), empty.clone()

        mat_neg_sorting = mat
        mat_pos_sorting = mat.clone()

        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings is ref_emb:
            mat_pos_sorting.fill_diagonal_(pos_ignore)
            mat_neg_sorting.fill_diagonal_(neg_ignore)

        pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)

        if self.distance.is_inverted:
            hard_pos_idx = torch.where(
                pos_sorted - self.epsilon < neg_sorted[:, -1].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted + self.epsilon > pos_sorted[:, 0].unsqueeze(1)
            )
        else:
            hard_pos_idx = torch.where(
                pos_sorted + self.epsilon > neg_sorted[:, 0].unsqueeze(1)
            )
            hard_neg_idx = torch.where(
                neg_sorted - self.epsilon < pos_sorted[:, -1].unsqueeze(1)
            )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]

        return a1, p, a2, n

class ContrastiveLossWithMatchingTable(losses.ContrastiveLoss):
    def __init__(self, pos_margin=0, neg_margin=1, **kwargs):
        super().__init__(pos_margin, neg_margin, **kwargs)

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        pos_pairs = pos_pairs_from_tuple(indices_tuple)
        neg_pairs = neg_pairs_from_tuple(indices_tuple)
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, labels, indices_tuple)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
