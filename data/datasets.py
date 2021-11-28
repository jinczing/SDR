import ast
from data.data_utils import get_gt_seeds_titles, raw_data_link
# import nltk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import json
import csv
import sys
from models.reco.recos_utils import index_amp
from opencc import OpenCC
# from ltp import LTP
# import synonyms


# nltk.download("punkt")


class WikipediaTextDatasetParagraphsSentences(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="train"):
        self.hparams = hparams
        cached_features_file = os.path.join(
            f"data/datasets/cached_proccessed/{dataset_name}",
            f"bs_{block_size}_{dataset_name}_{type(self).__name__}_tokenizer_{str(type(tokenizer)).split('.')[-1][:-2]}_mode_{mode}",
        )
        self.cached_features_file = cached_features_file
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)

        raw_data_path = self.download_raw(dataset_name)

        all_articles = self.save_load_splitted_dataset(mode, cached_features_file, raw_data_path)

        self.hparams = hparams

        max_article_len,max_sentences, max_sent_len = int(1e6), 16, 10000
        block_size = min(block_size, tokenizer.max_len_sentences_pair) if tokenizer is not None else block_size
        self.block_size = block_size
        self.tokenizer = tokenizer

        self.sent_tokenizer = lambda s: [(i+'。').strip() for i in s.split('。') if i is not '']

        if self.hparams.language is 'chinese':
            self.t2s = OpenCC('t2s').convert
            self.ensure_ascii = False
            # self.ltp = LTP()
        else:
            self.t2s = lambda x:x
            # self.sent_tokenizer = nltk.sent_tokenize
            self.ensure_ascii = True

        if os.path.exists(cached_features_file) and (self.hparams is None or not self.hparams.overwrite_data_cache):
            print("\nLoading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples, self.indices_map = pickle.load(handle)
        else:
            print("\nCreating features from dataset file at ", cached_features_file)

            self.examples = []
            self.indices_map = []

            for idx_article, article in enumerate(tqdm(all_articles)):
                this_sample_sections = []
                title, sections = article[0], ast.literal_eval(article[1])
                valid_sections_count = 0
                for section_idx, section in enumerate(sections):
                    this_sections_sentences = []
                    if section[1] == "":
                        continue
                    valid_sentences_count = 0
                    title_with_base_title = "{}:{}".format(title, section[0])
                    for sent_idx, sent in enumerate(self.sent_tokenizer(section[1][:max_article_len])[:max_sentences]):
                        if self.hparams.language == "chinese":
                            txt = self.t2s(sent[:max_sent_len])[:block_size]
                        else:
                            txt = json.dumps(sent[:max_sent_len])[:block_size]
                        tokenized_desc = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))[:block_size]
                        print(txt)
                        print(tokenized_desc)
                        tup = (
                                tokenized_desc,
                                len(tokenized_desc),
                                idx_article,
                                valid_sections_count,
                                valid_sentences_count,
                                sent[:max_sent_len],
                            )

                        # generate alterative synonyms tokens
                        # segs, _ = self.ltp.seg([txt])
                        # segs = segs[0]
                        # syns = []
                        # seg_inds = []
                        # for seg in segs:
                        #     s, _ = synonyms.nearby(seg, self.hparams.max_synonyms+1)
                        #     s = s[1:]
                        #     if len(s) == 0 or seg.isalnum():
                        #         s = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seg))
                        #         s = list(map(lambda x: [x]*self.hparams.max_synonyms, s))
                        #         syns += s
                        #     else:
                        #         for i in range(len(s)):
                        #             ss = s[i][:len(seg)]
                        #             if len(ss) < len(seg):
                        #                 ss = ss + seg[-(len(seg)-len(ss)):]
                        #             if ss.isalnum():
                        #                 ss = seg
                        #             s[i] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ss))
                        #         syns += np.array(s).transpose((1, 0)).tolist()

                        #     seg_inds.append(len(seg))

                        # separate word for whole word masking
                        if self.hparams.base_model_name == 'macbert':
                            tup = tup # + (None, None) # + (np.array(syns).transpose((1, 0)), np.array(seg_inds))
                        else:
                            tup = tup # + (None, None)
                        this_sections_sentences.append(tup,)
                        self.indices_map.append((idx_article, valid_sections_count, valid_sentences_count))
                        valid_sentences_count += 1
                    this_sample_sections.append((this_sections_sentences, title_with_base_title))
                    valid_sections_count += 1
                self.examples.append((this_sample_sections, title))

            print("\nSaving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.examples, self.indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.labels = [idx_article for idx_article, _, _ in self.indices_map]

        # prepare for ground truth matching table
        if self.hparams.use_matching_table:
            inds_table = [int(tup[1]) for tup in self.examples]
            # print(inds_table)
            path = self.hparams.matching_file_path
            matching_table = torch.zeros(len(self.examples), len(self.examples)).bool()
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for i, r in enumerate(reader):
                    if not i:
                        continue
                    l = list(map(int, r))
                    if l[0] not in inds_table or l[1] not in inds_table:
                        continue
                    # print(l[0], l[1], inds_table.index(l[0]), inds_table.index(l[1]))
                    matching_table[inds_table.index(l[0]), inds_table.index(l[1])] = True
                    matching_table[inds_table.index(l[1]), inds_table.index(l[0])] = True
            self.matching_table = matching_table
        else:
            self.matching_table = None

    def save_load_splitted_dataset(self, mode, cached_features_file, raw_data_path):
        proccessed_path = f"{cached_features_file}_EXAMPLES"
        if not os.path.exists(proccessed_path):
            all_articles = self.read_all_articles(raw_data_path)
            indices = list(range(len(all_articles)))
            if mode != "test":
                state = np.random.get_state() # ensure disjoint indices between train and val set
                np.random.seed(9487)
                train_indices = sorted(
                    np.random.choice(indices, replace=False, size=int(len(all_articles) * self.hparams.train_val_ratio))
                )
                np.random.set_state(state)
                val_indices = np.setdiff1d(list(range(len(all_articles))), train_indices)
                indices = train_indices if mode == "train" else val_indices

            articles = []
            for i in indices:
                articles.append(all_articles[i])
            all_articles = articles
            pickle.dump(all_articles, open(proccessed_path, "wb"))
            print(f"\nsaved dataset at {proccessed_path}")
        else:
            all_articles = pickle.load(open(proccessed_path, "rb"))
        setattr(self.hparams, f"{mode}_data_file", proccessed_path)
        return all_articles

    def read_all_articles(self, raw_data_path):
        csv.field_size_limit(sys.maxsize)
        with open(raw_data_path, newline="") as f:
            reader = csv.reader(f)
            all_articles = list(reader)
        return all_articles[1:]

    def download_raw(self, dataset_name):
        raw_data_path = f"data/datasets/{dataset_name}/raw_data"
        # if 'agricultures' in dataset_name:
        # 	raw_data_path += '.csv'
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        if not os.path.exists(raw_data_path):
            os.system(f"wget -O {raw_data_path} {raw_data_link(dataset_name)}")
        return raw_data_path

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, item):
        idx_article, idx_section, idx_sentence = self.indices_map[item]
        sent = self.examples[idx_article][0][idx_section][0][idx_sentence]
        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sent[0]), dtype=torch.long,)[
                : self.hparams.limit_tokens
            ],
            self.examples[idx_article][1],
            self.examples[idx_article][0][idx_section][1],
            sent[1],
            idx_article,
            idx_section,
            idx_sentence,
            item,
            self.labels[item],
            self.matching_table, # 9
            # torch.tensor(sent[-1]) # segs, 10
            # torch.tensor([self.tokenizer.build_inputs_with_special_tokens(i) for i in sent[-2]])[
            #     :, : self.hparams.limit_tokens
            # ].permute(1, 0), # words x syns, 11
        )

class WikipediaTextDatasetParagraphsSentencesTest(WikipediaTextDatasetParagraphsSentences):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="test"):
        super().__init__(tokenizer, hparams, dataset_name, block_size, mode=mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sections = []
        for idx_section, section in enumerate(self.examples[item][0]):
            sentences = []
            for idx_sentence, sentence in enumerate(section[0]):
                sentences.append(
                    (
                        torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sentence[0]), dtype=torch.long,),
                        self.examples[item][1],
                        section[1],
                        sentence[1],
                        item,
                        idx_section,
                        idx_sentence,
                        item,
                        self.labels[item],
                        self.matching_table,
                    )
                )
            sections.append(sentences)
        return sections

# tokenized_desc,
# len(tokenized_desc),
# idx_article,
# valid_sections_count,
# valid_sentences_count,
# sent[:max_sent_len],