# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import math
import operator

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSingleTokenClassification, BertForSingleTokenClassificationFollowTemporal, TemporalModelJoint, TemporalModelJointWithLikelihood, BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForSingleTokenClassificationWithPooler, TemporalModelArguments
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, target_idx, soft_label_indices, soft_label_values, mlm_labels):
        self.guid = guid
        self.text = text
        self.target_idx = target_idx
        self.soft_label_indices = soft_label_indices
        self.soft_label_values = soft_label_values
        self.mlm_labels = mlm_labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, lm_labels, target_idx, soft_label_indices, soft_label_values, adjustments, non_zero_adjustments):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_labels = lm_labels
        self.target_idx = target_idx
        self.soft_label_indices = soft_label_indices
        self.soft_label_values = soft_label_values
        self.adjustments = adjustments
        self.non_zero_adjustments = non_zero_adjustments


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class TemporalVerbProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        f = open(os.path.join(data_dir, "train.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        examples = self._create_examples(lines, "train")
        return examples

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test.formatted.txt"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for i in range(0, len(lines)):
            guid = "%s-%s" % (set_type, i)
            group = lines[i].split("\t")
            text = group[0]
            target_idx = int(group[1])
            soft_label_indices = [int(x) for x in group[2].split()]
            """No COMP"""
            # if 62 in soft_label_indices or 60 in soft_label_indices:
            #     continue
            soft_label_values = [float(x) for x in group[3].split()]
            """NO SOFT LOSS"""
            # max_pos = int(np.argmax(np.array(soft_label_values)))
            # for ii in range(0, len(soft_label_values)):
            #     if max_pos == ii:
            #         soft_label_values[ii] = 1.0
            #     else:
            #         soft_label_values[ii] = 0.0

            mlm_labels = [int(x) for x in group[4].split()]

            examples.append(
                InputExample(
                    guid=guid, text=text, target_idx=target_idx,
                    soft_label_indices=soft_label_indices, soft_label_values=soft_label_values, mlm_labels=mlm_labels
                )
            )

        return examples


def random_word(tokens, target_id, tokenizer, non_mask_ids):
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        if i in non_mask_ids:
            """SKIPPING ALL NON MASK IDS"""
            output_label.append(-1)
            continue
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                """CHANGED: NEVER PREDICT [UNK]"""
                # output_label.append(tokenizer.vocab["[UNK]"])
                output_label.append(-1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    tmp_prob = random.random()
    # 80% on temporal words
    # Yet never compute loss until feedback
    output_label[target_id] = -1
    if tmp_prob < 0.8:
        tokens[target_id] = "[MASK]"
    assert len(tokens) == len(output_label)

    return tokens, output_label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = example.text.split()
        second_sent_start = -1
        second_sent_end = -1
        sep_count = 0
        for i, t in enumerate(tokens):
            if t == "[SEP]":
                sep_count += 1
                if sep_count == 3:
                    second_sent_start = i + 1
                if sep_count == 4:
                    second_sent_end = i
        assert second_sent_start >= 0
        assert second_sent_end >= 0

        lm_labels = example.mlm_labels

        if len(tokens) > max_seq_length:
            # Never delete any token
            continue

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * max_seq_length
        for i in range(second_sent_start, second_sent_end):
            segment_ids[i] = 1
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_labels) == max_seq_length

        target_idx = example.target_idx
        if target_idx == -1:
            target_idx = 0
            # example.soft_label_indices = [101] + list(range(50, 61))
            # example.soft_label_values = [1.0] + [0.0] * 11

        assert len(example.soft_label_indices) == 12
        assert len(example.soft_label_values) == 12

        """BEFORE FIX"""
        # adjustment_map = {0: 0.0, 1: 5.513921678845794, 2: 2.9973924911894723, 3: 1.8219968547618164, 4: 2.1861525889607014, 5: 2.233134514107059, 6: 0.7213863026862188, 7: 0.965479002624672, 8: 0.3715687611428456, 9: 0.5214643914021938, 10: 4.1779158509944105, 11: 0.5365740791348295, 12: 3.327310522470683, 13: 0.8811519797033287, 14: 1.2482150273926544, 15: 5.935389824132418, 16: 0.34051446722281276, 17: 3.1115438794196364, 18: 0.8312453900922734, 19: 0.8011958715771033, 20: 0.7960490988572191, 21: 0.8088813127336778, 22: 0.8540103654187053, 23: 1.2754739969338527, 24: 1.1814683276191291, 25: 1.5238313695190264, 26: 1.8970807997526538, 27: 1.229159131252539, 28: 1.3878916018447542, 29: 1.4407917265940688, 30: 1.2325610593099754, 31: 1.3322306223933753, 32: 1.9984393447962328, 33: 1.7169724057520404, 34: 1.7702945301542776, 35: 1.7747037319728438, 36: 1.721743502203931, 37: 0.5484479282536151, 38: 0.306760611497803, 39: 0.6376513923550018, 40: 0.9328928363276994, 41: 0.7022089531302801, 42: 1.922457582739841, 61: 1.3634633240482823, 62: 0.7562417171932185, 43: 4.474797013786715, 44: 3.53716267149103, 45: 3.1609600246357568, 46: 3.245282273203312, 47: 0.43192080811906136, 48: 0.7228450078123281, 49: 1.6355850338103632, 50: 0.32559365917678734, 51: 2.0494409503843465, 52: 3.0384591564996595, 53: 3.003305582761998, 54: 4.305194805194805, 55: 5.241106719367589, 56: 0.8861555134109997, 57: 0.43170139377727723, 58: 2.5653333333333332, 59: 0.2687886922697639, 60: 2.7797167138810197}
        """AFTER FIX"""
        # adjustment_map = {0: 0.0, 1: 5.279436241932201, 2: 2.200417234374922, 3: 1.3462279504490062, 4: 1.7079067353097266, 5: 1.2437578106707516, 6: 0.7074982869815412, 7: 0.4155354428148789, 8: 0.5189779844786042, 9: 2.0966355184630383, 10: 4.331683847450319, 11: 0.5374200356450761, 12: 3.4171826565555836, 13: 0.8818740399385561, 14: 1.2741157278240216, 15: 5.912066014015942, 16: 0.33699405000133403, 17: 3.0993052001318087, 18: 0.8210446352951665, 19: 0.7990979672629, 20: 0.7952210141538284, 21: 0.802515798627185, 22: 0.864184184340734, 23: 1.270256834652276, 24: 1.1880282617517663, 25: 1.5241452785171494, 26: 1.9617420863676505, 27: 1.2484593964110007, 28: 1.3836506137984137, 29: 1.4301377219743707, 30: 1.2516372532086764, 31: 1.3244082126689376, 32: 1.984676858149819, 33: 1.708181986800006, 34: 1.7674644553552865, 35: 1.7272169604901224, 36: 1.7255449208381475, 37: 0.5493663838581674, 38: 0.3065854666711053, 39: 0.641413423255058, 40: 0.9248867055264269, 41: 0.7103861238361687, 42: 1.9347789744131207, 61: 1.3474239248991997, 62: 0.7500170542529896, 43: 4.196568274538037, 44: 3.2345781773116546, 45: 2.611574074074074, 46: 0.4186892303124768, 47: 1.137194844382015, 48: 1.0291372410546138, 49: 0.3299950935990338, 50: 1.7325110219448159, 51: 4.491472748754302, 52: 3.017079753590045, 53: 2.747570265300762, 54: 4.194666437512532, 55: 0.8626092693386113, 56: 0.6886951287188314, 57: 0.8448867734025675, 58: 0.27139023558540526, 59: 2.713359273670558, 60: 4.586426132982555}
        """AFTER FIX PARTIAL MASK"""
        # adjustment_map = {0: 0.0, 1: 5.139154718921255, 2: 2.175401630262741, 3: 1.3535403894920846, 4: 1.715999094817832, 5: 1.2432131429282136, 6: 0.7122090245896994, 7: 0.41471737787772917, 8: 0.5181438834342963, 9: 2.107569854755778, 10: 4.27797090753724, 11: 0.5371718836875835, 12: 3.440188660091675, 13: 0.8882692265040759, 14: 1.2652616946971786, 15: 6.029452420381294, 16: 0.3371949735405189, 17: 3.109719771299499, 18: 0.8159510452338927, 19: 0.8107838518278588, 20: 0.8072376799113156, 21: 0.813429074170838, 22: 0.8485949337098412, 23: 1.260212765348826, 24: 1.1966503330110063, 25: 1.529425242664519, 26: 1.930374810491539, 27: 1.2414972791857002, 28: 1.372023494384892, 29: 1.42242211538772, 30: 1.2292551989307767, 31: 1.3346562128886843, 32: 1.9823433330783526, 33: 1.7314107401119285, 34: 1.8039374657068685, 35: 1.7711140771234684, 36: 1.698508348851286, 37: 0.5465089738055188, 38: 0.30526086037673095, 39: 0.6353175982314234, 40: 0.9357621225283934, 41: 0.7057996868502785, 42: 1.940046565774156, 61: 1.3540274033312405, 62: 0.7523444714251514, 43: 4.151113023934199, 44: 3.2360715017987287, 45: 2.6406875047532132, 46: 0.4178719368806668, 47: 1.1499387315780758, 48: 1.027223080154546, 49: 0.33039122772963075, 50: 1.7055899400726988, 51: 4.500751801731736, 52: 2.9977080647471706, 53: 2.7709491922975067, 54: 4.19895663140998, 55: 0.8533223820258753, 56: 0.6854729650686694, 57: 0.8466495590156164, 58: 0.2726695703968431, 59: 2.706594239048094, 60: 4.57463618762101}
        """AFTER FIX SINGLE SENT"""
        adjustment_map = {0: 0.0, 1: 5.1982408979437835, 2: 2.2102608835148088, 3: 1.3463481070252905, 4: 1.7188246806154948, 5: 1.2401767606445968, 6: 0.7068293566627067, 7: 0.41569694535429474, 8: 0.5195649688952053, 9: 2.0865585749306677, 10: 4.292306495375728, 11: 0.5376932907270399, 12: 3.41592625843601, 13: 0.8742220708069072, 14: 1.2801277328862875, 15: 5.789988062286007, 16: 0.33583952080863544, 17: 3.20521851534473, 18: 0.8164686687212576, 19: 0.8034298654230173, 20: 0.818251761216166, 21: 0.8135760368663594, 22: 0.8599130666884877, 23: 1.241829973411364, 24: 1.1959328555364377, 25: 1.5136700664640963, 26: 1.932615806320251, 27: 1.2356865687520646, 28: 1.3794768573937883, 29: 1.458768508725542, 30: 1.270645305235278, 31: 1.344817289615688, 32: 1.9113412062238273, 33: 1.6712736993967943, 34: 1.7922432836305753, 35: 1.7890903047450728, 36: 1.678168392876154, 37: 0.5458357861202764, 38: 0.305507813770815, 39: 0.6387429629951229, 40: 0.9358825113495884, 41: 0.7089576264092873, 42: 1.9337460484720759, 61: 1.3474741170423672, 62: 0.7517563683287021, 43: 4.249665903016418, 44: 3.188790402005551, 45: 2.612761711929809, 46: 0.41837998106367635, 47: 1.133568859607244, 48: 1.029846689452165, 49: 0.3298345437404265, 50: 1.7359575754030923, 51: 4.605783157459135, 52: 3.0371497794620117, 53: 2.763808016281305, 54: 4.246048289039431, 55: 0.8547326829609427, 56: 0.6921748240857414, 57: 0.8419992077570915, 58: 0.2716553590380459, 59: 2.710183491324353, 60: 4.5255021753216695}

        """NO WEIGHT ADJUSTMENT"""
        # for key in adjustment_map:
        #     adjustment_map[key] = 1.0
        # adjustment_map[0] = 0.0

        actual_label = -1
        max_val = 0.0
        for it in range(0, len(example.soft_label_indices)):
            if example.soft_label_values[it] > max_val:
                actual_label = example.soft_label_indices[it]
                max_val = example.soft_label_values[it]

        non_zero_adjustments = adjustment_map[actual_label]

        if target_idx == 0 or target_idx == -1:
            actual_label = 0

        adjustments = adjustment_map[actual_label]


        if ex_index < 100:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("LM label: %s" % " ".join(str(x) for x in lm_labels))
            logger.info("soft_label_a: %s" % " ".join(str(x) for x in example.soft_label_indices))
            logger.info("soft_values_a: %s" % " ".join(str(x) for x in example.soft_label_values))
            logger.info("target index a: %s" % str(target_idx))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                lm_labels=lm_labels,
                target_idx=target_idx,
                soft_label_indices=example.soft_label_indices,
                soft_label_values=example.soft_label_values,
                adjustments=adjustments,
                non_zero_adjustments=non_zero_adjustments,
            )
        )
    return features


def simple_accuracy(preds, labels, tolerances=None):
    correct = 0.0
    if tolerances is not None:
        for i, v in enumerate(preds):
            if preds[i] == labels[i]:
            # if labels[i] - tolerances[i] <= preds[i] <= labels[i] + tolerances[i]:
                correct += 1.0
        return correct / float(len(preds))
    else:
        return 0.0


def compute_f1(p, r):
    return 2 * p * r / (p + r)


def compute_metrics(task_name, preds, labels, additional=None):
    if task_name == "tempoalverb":
        return simple_accuracy(preds, labels)
    else:
        raise KeyError(task_name)


def soft_cross_entropy_loss(logits, soft_indices, soft_values, lm_loss, adjustments=None, non_zero_adjustments=None):
    soft_target = torch.zeros(logits.size(0), logits.size(1)).cuda()
    for x in range(soft_target.size(0)):
        soft_target[x, soft_indices[x]] = soft_values[x]
    loss = -soft_target * torch.log(nn.functional.softmax(logits, -1))
    loss = torch.sum(loss, -1)
    lm_loss = lm_loss.view(-1, 128)

    non_zero_adjustments = non_zero_adjustments.view(-1, 1).repeat(1, 128)
    if adjustments is not None:
        loss = loss * adjustments
        lm_loss = lm_loss * non_zero_adjustments

    lm_loss = lm_loss.view(-1).sum() / torch.nonzero(lm_loss.view(-1).data).size(0)

    mean_loss = loss.mean()

    if lm_loss is not None:
        return mean_loss + lm_loss, mean_loss.item()
    else:
        return mean_loss.item()


def compute_distance(logits, target):
    vocab_indices = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        1: [10, 11, 12, 13, 14, 15, 16, 17],
        2: [18, 19, 20, 21, 22, 23, 24],
        3: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        4: [37, 38, 39, 40],
        5: [41, 42, 61, 62],
        6: [43, 44, 45, 46, 47, 48, 49, 50, 51],
        7: [52, 53, 54, 55, 56, 57, 58, 59, 60],
    }
    reverse_map = {}
    index_map = {}
    avg_dist_map = {}
    count_map = {}
    for i in vocab_indices:
        for j, idx in enumerate(vocab_indices[i]):
            reverse_map[idx] = i
            index_map[idx] = j
    for i in range(0, logits.shape[0]):
        label_id = int(np.argmax(target[i]))
        if label_id not in reverse_map:
            continue
        group_index = reverse_map[label_id]

        scores_in_order = []
        for gi in vocab_indices[group_index]:
            scores_in_order.append(logits[i][gi])
        predicted_relative_label_id = int(np.argmax(np.array(scores_in_order)))
        dist = float(abs(index_map[label_id] - predicted_relative_label_id))
        if group_index not in avg_dist_map:
            avg_dist_map[group_index] = 0.0
            count_map[group_index] = 0.0
        avg_dist_map[group_index] += dist
        count_map[group_index] += 1.0

    return avg_dist_map, count_map


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=48,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "temporalverb": TemporalVerbProcessor,
    }

    output_modes = {
        "temporalverb": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        # n_gpu = 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = TemporalModelArguments.from_pretrained(args.bert_model, cache_dir=cache_dir)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps,
                             e=1e-4)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_lm_labels = torch.tensor([f.lm_labels for f in train_features], dtype=torch.long)
        all_target_idxs = torch.tensor([f.target_idx for f in train_features], dtype=torch.long)
        all_soft_label_indices = torch.tensor([f.soft_label_indices for f in train_features], dtype=torch.long)
        all_soft_label_values = torch.tensor([f.soft_label_values for f in train_features], dtype=torch.float)
        all_adjustments = torch.tensor([f.adjustments for f in train_features], dtype=torch.float)
        all_non_zero_adjustments = torch.tensor([f.non_zero_adjustments for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lm_labels,
            all_target_idxs, all_soft_label_indices, all_soft_label_values, all_adjustments, all_non_zero_adjustments
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        output_loss_file = os.path.join(args.output_dir, "loss_log.txt")
        f_loss = open(output_loss_file, "a")
        epoch_loss = 0.0
        epoch_label_loss = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            middle_loss = 0.0
            middle_label_loss = 0.0
            middle_rel_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                if step == 0:
                    epoch_loss = 0.0
                    epoch_label_loss = 0.0

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_labels, target_ids, soft_label_indices, soft_label_values, adjustments, non_zero_adjustments = batch

                lm_loss, cls = model(
                    input_ids, segment_ids, input_mask, lm_labels, target_ids,
                )

                loss, non_lm_loss = soft_cross_entropy_loss(
                    cls.view(-1, 30522),
                    soft_label_indices, soft_label_values,
                    lm_loss, adjustments, non_zero_adjustments,
                )

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                tr_loss += loss.item()
                middle_loss += loss.item()
                middle_label_loss += non_lm_loss
                epoch_loss += loss.item()
                epoch_label_loss += non_lm_loss

                if step % 100 == 0:
                    f_loss.write(("Total Loss: " + str(middle_loss)) + "\n")
                    f_loss.write(("Label Loss: " + str(middle_label_loss)) + "\n")
                    f_loss.write(("Rel Loss: " + str(middle_rel_loss)) + "\n")
                    f_loss.flush()
                    middle_loss = 0.0
                    middle_label_loss = 0.0
                    middle_rel_loss = 0.0
                nb_tr_examples += input_ids.size(0) * 2
                nb_tr_steps += 1
                if step % 1000 == 0:
                    actual_dir = args.output_dir + "_epoch_" + str(_)
                    if not os.path.exists(actual_dir):
                        os.makedirs(actual_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(actual_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(actual_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = TemporalModelArguments(config)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = TemporalModelArguments.from_pretrained(args.bert_model)
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_lm_labels = torch.tensor([f.lm_labels for f in eval_features], dtype=torch.long)
        all_target_idxs = torch.tensor([f.target_idx for f in eval_features], dtype=torch.long)
        all_soft_label_indices = torch.tensor([f.soft_label_indices for f in eval_features], dtype=torch.long)
        all_soft_label_values = torch.tensor([f.soft_label_values for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lm_labels,
            all_target_idxs, all_soft_label_indices, all_soft_label_values,
        )

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        output_file = os.path.join(args.output_dir, "bert_logits.txt")
        f_out = open(output_file, "w")
        total_loss = []
        lm_total_loss = []
        prediction_distance = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_labels, target_ids, soft_label_indices, soft_label_values = batch

            with torch.no_grad():
                lm_loss, cls = model(
                    input_ids, segment_ids, input_mask, lm_labels, target_ids
                )
                cls = cls.view(-1, 30522)
                avg_loss = soft_cross_entropy_loss(cls, soft_label_indices, soft_label_values, None)
                soft_target = torch.zeros(cls.size(0), cls.size(1)).cuda()
                for x in range(soft_target.size(0)):
                    soft_target[x, soft_label_indices[x]] = soft_label_values[x]

            prediction_distance.append(compute_distance(cls.cpu().numpy(), soft_target.cpu().numpy()))
            lm_total_loss.append(lm_loss.item())
            total_loss.append(avg_loss)

        f_out.write("Temporal Loss\n")
        f_out.write(str(np.mean(np.array(total_loss))) + "\n")
        f_out.write("LM Loss\n")
        f_out.write(str(np.mean(np.array(lm_total_loss))) + "\n")
        f_out.write("Label Distance\n")
        mm_total = {}
        mm_count = {}
        for mmt, mmc in prediction_distance:
            for key in mmt:
                if key not in mm_total:
                    mm_total[key] = 0.0
                    mm_count[key] = 0.0
                mm_total[key] += mmt[key]
                mm_count[key] += mmc[key]
        for key in mm_total:
            mm_total[key] /= mm_count[key]

        f_out.write(str(mm_total) + "\n")


if __name__ == "__main__":
    main()
