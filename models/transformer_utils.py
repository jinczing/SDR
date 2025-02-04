from typing import Tuple
import torch
from transformers import PreTrainedTokenizer
# import synonyms
import random
import time

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, synonyms=None, seg_lens=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    device = inputs.device

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability, device=labels.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    if (~masked_indices).all():
        masked_indices = ~masked_indices  # If we choose to not learn anything - learn everything
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if seg_lens is not None: # whole word masking
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool()
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool()
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        random_synonyms = torch.randint(args.max_synonyms, labels.shape, dtype=torch.long, device=device)
        random_synonyms.masked_fill_(~indices_replaced | ~masked_indices, value=-1)
        timer = time.time()
        # inputs: bxs
        # synonyms: bx5xs
        # seg_lens: bx[seg]
        for b in range(labels.size(0)):
            prev = 1
            print(seg_lens[b])
            print(labels[b])
            for seg_len in seg_lens[b]:
                random_synonyms[b, prev:prev+seg_len] = random_synonyms[b, prev].clone()
                prev = prev+seg_len
                if prev > labels.size(-1):
                    break
        synonyms = torch.cat([synonyms, labels.unsqueeze(-1)], dim=-1) # bxsx(5+1)
        print(synonyms.shape, random_synonyms.shape)
        inputs = torch.gather(synonyms, 1, random_synonyms.unsqueeze(-1).repeat(1, 1, args.max_synonyms+1))[:, :, 0]
        
        indices_random = indices_random & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        inputs[indices_random] = random_words[indices_random]
        print('whole word masking time:', time.time()-timer)
    else:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

