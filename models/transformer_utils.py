from typing import Tuple
import torch
from transformers import PreTrainedTokenizer
import synonyms

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, word_segs=None: list[list[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    cc = OpenCC('t2s')

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    
    if word_segs is not None:
        probility_matrix = torch.full(labels.shape, args.mlm_probability, device=labels.device)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)
    else:
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

    if word_segs is not None: # whole word masking
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
        for i, word_seg in enumerate(word_segs):
            prev_len = 1
            for j, word in enumerate(word_seg):
                if masked_indices[i][prev_len] and indices_replaced[i][prev_len]:
                    syn = random.choice(synonyms.nearby(word, 5)[0][1:])[:len(word)]
                    inputs[i][prev_len:prev_len+len(syn)] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(syn))
                elif indices_random[i][prev_len]:
                    inputs[i][prev_len:prev_len+len(word)] = random_words[i][prev_len:prev_len+len(word)]
                prev_len += len(word)
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

