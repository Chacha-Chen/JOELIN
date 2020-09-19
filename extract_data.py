#!/usr/bin/env python3

import os
import csv
import time
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from pprint import pprint
from transformers import (
    BertTokenizerFast, BertPreTrainedModel, BertModel, BertConfig,
    AutoTokenizer, AutoModel, AutoConfig,
    RobertaConfig, RobertaModel,
    AdamW, get_linear_schedule_with_warmup)
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.loadData import loadData, loadNewData
from preprocessing.processText import getTextProcessingFuncList
from preprocessing.utils import (
    make_dir_if_not_exists, format_time, log_list, plot_train_loss,
    saveToJSONFile, loadFromJSONFile)
from model import (MultiTaskBertForCovidEntityClassification,
                   MultiTaskBertForCovidEntityClassificationNew)

import logging
import h5py
import json

EVENT_LIST = ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']
pd.set_option('display.max_columns', None)

################### util ####################
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="Path to the output directory", type=str, default='./results/debug_chacha')
    parser.add_argument("-rt", "--retrain", help="True if the model needs to be retrained", action="store_false", default=True)
    parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=32)
    parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=8)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=2e-5)
    parser.add_argument("-d", "--device", help="Device for running the code", type=str, default="cuda")
    parser.add_argument("-pm", "--pretrained_model", help="pretrained model version", type=str, default="bert-base-cased")
    parser.add_argument("-w", "--weighting", help="weighting for classes, 10 means 0.1:1, 5 means 0.2:1", type=int, default=None)
    parser.add_argument("-fl", "--f1_loss", help="using F1 loss", type=str_to_bool, default=False)
    parser.add_argument("-bu", "--batch_size_update", type=int, default=-1)

    # Add Data Clean Options
    parser.add_argument("-ca", "--clean_all", action="store_true", default=False)
    # parser.add_argument("--replace_tags", action="store_true", default=False)

    # NOTE Placeholder, dependent if we want to move them out from preprocessing
    # parser.add_argument(
    #     '--replace_usernames', default=False, help='Replace usernames with filler',
    #     action="store_true")
    # parser.add_argument('--replace_urls', default=False, help='Replace URLs with filler',
    #                     action="store_true")
    # parser.add_argument('--replace_multiple_usernames', default=False,
    #                     help='Replace "@user @user" with "2 <username_filler>"', action="store_true")
    # parser.add_argument('--replace_multiple_urls', default=False,
    #                     help='Replace "http://... http://.." with "2 <url_filler>"',
    #                     action="store_true")

    parser.add_argument(
        '--force_lower_case', default=False, action="store_true",
        help='Convert text to lower case (not included in clean_all)')
    parser.add_argument(
        '--asciify_emojis', default=False, help='Asciifyi emojis', action="store_true")
    parser.add_argument(
        '--standardize_punctuation', default=False,
        help='Standardize (asciifyi, action="store_true") special punctuation',
        action="store_true")
    parser.add_argument(
        '--remove_unicode_symbols', default=False,
        help='After preprocessing remove characters which belong to unicode category "So"',
        action="store_true")
    parser.add_argument(
        '--remove_accented_characters', default=False,
        help='Remove accents/asciify everything. Probably not recommended.',
        action="store_true")
    parser.add_argument(
        '--replace_tags', default=False,
        help='After preprocessing remove characters which belong to unicode category "So"',
        action="store_true")

    parser.add_argument(
        '--new_data', default=True,
        help='set to True if processing new data',
        action="store_true")

    return parser.parse_args()

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{} is not a valid boolean value'.format(value))

def extract_data(event):
    args =  parse_arg()
    if args.clean_all:
        version = "clean"
    else:
        version = "normal"

    pretrained_bert_version = args.pretrained_model
    #subtask_list = ['age', 'close_contact', 'employer', 'gender_male', 'gender_female', 'name', 'recent_travel', 'relation', 'when', 'where']

    # data loading
    #train_dataloader = torch.load("temp/train_dataloader.bin")
    #valid_dataloader = torch.load("temp/valid_dataloader.bin")
    #test_dataloader = torch.load("temp/test_dataloader.bin")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_version)
    tokenizer.add_tokens(["<E>", "</E>", "<URL>", "@USER"])
    entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]

    input_text_processing_func_list = getTextProcessingFuncList(args)
    if args.new_data:
        (train_dataloader, valid_dataloader, test_dataloader, subtask_list) = loadNewData(
            event, entity_start_token_id, tokenizer,
            batch_size=args.batch_size, train_ratio=0.6, dev_ratio=0.15,
            shuffle_train_data_flg=False, num_workers=0,
            input_text_processing_func_list=input_text_processing_func_list)

    #print(train_dataloader.dataset)
    #print(valid_dataloader.dataset)
    #print(test_dataloader.dataset)

    #torch.save(train_dataloader, "temp/train_dataloader.bin")
    #torch.save(valid_dataloader, "temp/valid_dataloader.bin")
    #torch.save(test_dataloader, "temp/test_dataloader.bin")

    # extract batch
    folder_path = os.path.join("temp", version)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    if args.new_data:
        # for dataloader, phase in zip(
        #     [train_dataloader, valid_dataloader, test_dataloader],
        #     ["train", "valid", "test"]
        # ):
        for dataloader, phase in zip(
            [train_dataloader],
            ["new"]
        ):
            all_input_ids = []
            all_entity_start_positions = []
            all_labels = []
            all_data = []
            for batch in dataloader:
                input_ids = batch["input_ids"].cpu().numpy()
                print(input_ids)
                print()
                print(tokenizer.decode(input_ids[0]))
                print()
                print(batch["batch_data"][0])
                quit()
                entity_start_positions = batch["entity_start_positions"].numpy()[:, 1]
                labels = np.vstack([batch["gold_labels"][subtask].numpy() for subtask in subtask_list]).T
                all_data.extend(batch["batch_data"])

                for input_id in input_ids:
                    input_id = input_id[input_id!=0]
                    all_input_ids.append(input_id)
                all_entity_start_positions.append(entity_start_positions)
                all_labels.append(labels)

            all_entity_start_positions = np.hstack(all_entity_start_positions)
            all_labels = np.vstack(all_labels)

            print(phase)
            print(f"input_ids.shape = {len(all_input_ids)}")
            print(f"entity_start_positions.shape = {all_entity_start_positions.shape}")
            print(f"labels.shape = {all_labels.shape}")

            # save
            with h5py.File(os.path.join(folder_path, f"{event}_{phase}.h5"), 'w') as outfile:
                outfile.create_dataset("entity_start_positions", data=all_entity_start_positions)
                outfile.create_dataset("labels", data=all_labels)

            table = pd.DataFrame({"input_ids":all_input_ids}, index=np.arange(len(all_input_ids)))
            table.to_parquet(os.path.join(folder_path, f"{event}_{phase}.parquet"))

            with open(os.path.join(folder_path, f"{event}_{phase}_data.json"), 'w', encoding='utf-8') as outfile:
                json.dump(all_data, outfile, indent=2)

            with open(os.path.join(folder_path, f"{event}_subtask.json"), 'w', encoding='utf-8') as outfile:
                json.dump(subtask_list, outfile, indent=2)


def main():
    for event in EVENT_LIST:
        print(event)
        extract_data(event)

if __name__ == "__main__":
    main()


