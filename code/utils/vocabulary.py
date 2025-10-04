import os
import pdb
import csv
import json 
import torch
from collections import Counter, OrderedDict
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False, add_double_eos=False, add_cls_token=False, add_s=False, add_cls_token_last=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_cls_token:
            return ['<CLS>'] + symbols + ['<S>']
        elif add_cls_token_last:
            return ['<S>'] + symbols + ['<CLS>']
        elif add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        elif add_s:
            return symbols + ['<S>']
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_csqa(self, path, num_classes=5, verbose=False, add_eos=False, add_double_eos=False, add_cls_token=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                example = json.loads(line.strip())
                question = example["question"]["stem"]
                assert len(example["question"]["choices"]) == num_classes
                # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
                question = "Q: " + question
                question_toks = self.tokenize(question, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                for i, choice in enumerate(example["question"]["choices"]):
                    src = "A: " + choice["text"]
                    assert (ord(choice["label"]) - ord("A")) == i
                    src_bin = self.tokenize(src, add_eos=add_eos)
                    question_toks.extend(src_bin)
                self.counter.update(question_toks)
                sents.append(question_toks)
        return sents

    def count_sst2(self, path, verbose=False, add_eos=False, add_double_eos=False, add_cls_token=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            for line in tsv_file:
                if not line[1] in ['0', '1']: 
                    # print('* Ignore ', line)
                    continue
                sentence, label = line[0], int(line[1])
                assert label in [0,1]
                sentence_toks = self.tokenize(sentence, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                self.counter.update(sentence_toks)
                sents.append(sentence_toks)
        return sents

    def count_stsb(self, path, verbose=False, add_eos=False, add_double_eos=False, add_cls_token=False):
        """为STSB数据集计数词汇"""
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 10:  # STSB格式检查
                    continue
                # STSB格式: genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
                sentence1, sentence2 = line[7], line[8]
                try:
                    score = float(line[9])
                    if score < 0 or score > 5:  # STSB分数范围检查
                        continue
                except (ValueError, IndexError):
                    continue
                
                # 将两个句子用分隔符连接，类似BERT的输入格式
                combined_sentence = sentence1 + " [SEP] " + sentence2
                sentence_toks = self.tokenize(combined_sentence, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                self.counter.update(sentence_toks)
                sents.append(sentence_toks)
        return sents

    def count_qqp(self, path, verbose=False, add_eos=False, add_double_eos=False, add_cls_token=False):
        """为QQP数据集计数词汇"""
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 6:  # QQP格式检查: id, qid1, qid2, question1, question2, is_duplicate
                    continue
                try:
                    question1, question2 = line[3], line[4]
                    label = int(line[5])
                    if label not in [0, 1]:  # 二分类标签检查
                        continue
                except (ValueError, IndexError):
                    continue
                
                # 将两个问题用分隔符连接，类似BERT的输入格式
                combined_question = question1 + " [SEP] " + question2
                question_toks = self.tokenize(combined_question, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                self.counter.update(question_toks)
                sents.append(question_toks)
        return sents

    def count_qnli(self, path, verbose=False, add_eos=False, add_double_eos=False, add_cls_token=False):
        """为QNLI数据集计数词汇"""
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 4:  # QNLI格式检查: index, question, sentence, label
                    continue
                try:
                    question, sentence = line[1], line[2]
                    label_str = line[3]
                    # QNLI标签通常是 'entailment' 或 'not_entailment'
                    if label_str == 'entailment':
                        label = 1
                    elif label_str == 'not_entailment':
                        label = 0
                    else:
                        # 如果是数字格式
                        label = int(label_str)
                        if label not in [0, 1]:
                            continue
                except (ValueError, IndexError):
                    continue
                
                # 将问题和句子用分隔符连接，类似BERT的输入格式
                combined_text = question + " [SEP] " + sentence
                text_toks = self.tokenize(combined_text, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                self.counter.update(text_toks)
                sents.append(text_toks)
        return sents

    def count_rte(self, path, verbose=False, add_eos=False, add_double_eos=False, add_cls_token=False):
        """为RTE数据集计数词汇"""
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 4:  # RTE格式检查: index, sentence1, sentence2, label
                    continue
                try:
                    sentence1, sentence2 = line[1], line[2]
                    label_str = line[3]
                    # RTE标签通常是 'entailment' 或 'not_entailment'
                    if label_str == 'entailment':
                        label = 1
                    elif label_str == 'not_entailment':
                        label = 0
                    else:
                        # 如果是数字格式
                        label = int(label_str)
                        if label not in [0, 1]:
                            continue
                except (ValueError, IndexError):
                    continue
                
                # 将两个句子用分隔符连接，类似BERT的输入格式
                combined_text = sentence1 + " [SEP] " + sentence2
                text_toks = self.tokenize(combined_text, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                self.counter.update(text_toks)
                sents.append(text_toks)
        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_csqa_file(self, path, ordered=False, num_classes=5, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = [[] for i in range(num_classes)]
        labels = []

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                example = json.loads(line.strip())
                if "answerKey" in example:
                    label = ord(example["answerKey"]) - ord("A")
                    labels.append(label)
                question = example["question"]["stem"]
                assert len(example["question"]["choices"]) == num_classes
                # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
                question = "Q: " + question
                question_bin = self.tokenize(question,  add_eos=add_eos,
                        add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                for i, choice in enumerate(example["question"]["choices"]):
                    src = " A: " + choice["text"]
                    assert (ord(choice["label"]) - ord("A")) == i
                    src_bin = question_bin + self.tokenize(src, add_s=True)
                    encoded[i].append(self.convert_to_tensor(src_bin))

        labels = torch.LongTensor(labels)

        # pdb.set_trace()

        # if ordered:
        #     for idx in range(num_classes):
        #         encoded[idx] = pad_sequence(encoded[idx])

        # encoded = pad_sequence(encoded)
        # print(encoded.shape)

        return [encoded, labels]

    def encode_sst2_file(self, path, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            for line in tsv_file: 
                if not line[1] in ['0', '1']: 
                    print('* Ignore ', line)
                    continue
                sentence, label = line[0], int(line[1])
                assert label in [0,1]
                sentence_toks = self.tokenize(sentence, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                encoded.append(self.convert_to_tensor(sentence_toks))
                labels.append(label)

        labels = torch.LongTensor(labels)
        return [encoded, labels]

    def encode_sst2_file_v2(self, path, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token_last=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            for line in tsv_file: 
                if not line[1] in ['0', '1']: 
                    print('* Ignore ', line)
                    continue
                sentence, label = line[0], int(line[1])
                assert label in [0,1]
                sentence_toks = self.tokenize(sentence, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token_last=add_cls_token_last)
                encoded.append(self.convert_to_tensor(sentence_toks))
                labels.append(label)

        labels = torch.LongTensor(labels)
        return [encoded, labels]

    def encode_stsb_file(self, path, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token=False):
        """编码STSB数据集文件"""
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 10:  # STSB格式检查
                    continue
                # STSB格式: genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
                sentence1, sentence2 = line[7], line[8]
                try:
                    score = float(line[9])
                    if score < 0 or score > 5:  # STSB分数范围检查
                        continue
                except (ValueError, IndexError):
                    continue
                
                # 将两个句子用分隔符连接，类似BERT的输入格式
                combined_sentence = sentence1 + " [SEP] " + sentence2
                sentence_toks = self.tokenize(combined_sentence, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                encoded.append(self.convert_to_tensor(sentence_toks))
                labels.append(score)

        labels = torch.FloatTensor(labels)  # STSB使用浮点数标签
        return [encoded, labels]

    def encode_qqp_file(self, path, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token=False):
        """编码QQP数据集文件"""
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 6:  # QQP格式检查: id, qid1, qid2, question1, question2, is_duplicate
                    continue
                try:
                    question1, question2 = line[3], line[4]
                    label = int(line[5])
                    if label not in [0, 1]:  # 二分类标签检查
                        continue
                except (ValueError, IndexError):
                    continue
                
                # 将两个问题用分隔符连接，类似BERT的输入格式
                combined_question = question1 + " [SEP] " + question2
                question_toks = self.tokenize(combined_question, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                encoded.append(self.convert_to_tensor(question_toks))
                labels.append(label)

        labels = torch.LongTensor(labels)  # QQP使用整数标签（0或1）
        return [encoded, labels]

    def encode_qnli_file(self, path, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token=False):
        """编码QNLI数据集文件"""
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 4:  # QNLI格式检查: index, question, sentence, label
                    continue
                try:
                    question, sentence = line[1], line[2]
                    label_str = line[3]
                    # QNLI标签通常是 'entailment' 或 'not_entailment'
                    if label_str == 'entailment':
                        label = 1
                    elif label_str == 'not_entailment':
                        label = 0
                    else:
                        # 如果是数字格式
                        label = int(label_str)
                        if label not in [0, 1]:
                            continue
                except (ValueError, IndexError):
                    continue
                
                # 将问题和句子用分隔符连接，类似BERT的输入格式
                combined_text = question + " [SEP] " + sentence
                text_toks = self.tokenize(combined_text, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                encoded.append(self.convert_to_tensor(text_toks))
                labels.append(label)

        labels = torch.LongTensor(labels)  # QNLI使用整数标签（0或1）
        return [encoded, labels]

    def encode_rte_file(self, path, verbose=False, add_eos=False,
            add_double_eos=False, add_cls_token=False):
        """编码RTE数据集文件"""
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        labels = []
        with open(path, 'r', encoding='utf-8') as f:
            tsv_file = csv.reader(f, delimiter="\t")
            next(tsv_file, None)  # 跳过标题行
            for line in tsv_file:
                if len(line) < 4:  # RTE格式检查: index, sentence1, sentence2, label
                    continue
                try:
                    sentence1, sentence2 = line[1], line[2]
                    label_str = line[3]
                    # RTE标签通常是 'entailment' 或 'not_entailment'
                    if label_str == 'entailment':
                        label = 1
                    elif label_str == 'not_entailment':
                        label = 0
                    else:
                        # 如果是数字格式
                        label = int(label_str)
                        if label not in [0, 1]:
                            continue
                except (ValueError, IndexError):
                    continue
                
                # 将两个句子用分隔符连接，类似BERT的输入格式
                combined_text = sentence1 + " [SEP] " + sentence2
                text_toks = self.tokenize(combined_text, add_eos=add_eos, add_double_eos=add_double_eos, add_cls_token=add_cls_token)
                encoded.append(self.convert_to_tensor(text_toks))
                labels.append(label)

        labels = torch.LongTensor(labels)  # RTE使用整数标签（0或1）
        return [encoded, labels]

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            print(sym)
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
