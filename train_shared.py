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
from prediction import new_data_predict
from transformers import (
    AutoTokenizer, AutoConfig,
    AdamW, get_linear_schedule_with_warmup)
import torch
from model import MultiTaskBertForCovidEntityClassificationShare
from preprocessing.utils import (
    make_dir_if_not_exists, format_time, log_list, plot_train_loss,
    saveToJSONFile)
from torch.utils import data as torch_data
from prediction import prediction_to_submission
from preprocessing.loadData import loadData, loadNewData
import logging
import json
import h5py

EVENT_LIST = ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']
pd.set_option('display.max_columns', None)

################### util ####################
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="Path to the output directory", type=str, default='./results/global_1')
    parser.add_argument("-rt", "--retrain", help="True if the model needs to be retrained", action="store_false", default=True)
    parser.add_argument("-bs", "--batch_size", help="Train batch size for BERT model", type=int, default=16)
    parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=30)
    parser.add_argument("-E", "--embedding_type",
                        help=("Type of Embedding, 0 for last, 1 for Sum L4 and 2 for concat L4,"
                              "3 for multihead concat"),
                        type=int, default=0)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=2e-5)
    parser.add_argument("-d", "--device", help="Device for running the code", type=str, default="cuda")
    parser.add_argument("-pm", "--pretrained_model", help="pretrained model version", type=str, default="digitalepidemiologylab/covid-twitter-bert")
    parser.add_argument("-w", "--weighting", help="weighting for classes, 10 means 0.1:1, 5 means 0.2:1", type=int, default=10)
    parser.add_argument("-fl", "--f1_loss", help="using F1 loss", type=str_to_bool, default=False)
    parser.add_argument("-bu", "--batch_size_update", type=int, default=32)
    parser.add_argument("-new", "--new", type=str_to_bool, default=True)

    # Add Data Clean Options
    parser.add_argument("-ca", "--clean_all", action="store_true", default=True)

    return parser.parse_args()

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{} is not a valid boolean value'.format(value))

################### training functions ####################
# dataset
# testing/val script
def evaluation(model, dataloader, device, threshold=0.5, save_sample_flg=False):
    model.eval()

    total_preds, total_labels, total_batch_data = prepare_for_prediction(
        model, dataloader, device)

    if type(threshold) in {float, np.float32, np.float64}:
        prediction = (total_preds > threshold).astype(int)
    else:
        prediction = np.vstack(
            [(total_preds[:,subtask_idx] > threshold[subtask_idx]).astype(int)
             for subtask_idx in range(len(model.subtasks))]).T

    if save_sample_flg:
        save_sample_to_file(prediction, total_labels, total_batch_data,
                            model.subtasks, "shared")
    
    # Calculating metrics
    precision = np.array(
        [metrics.precision_score(total_labels[:,idx], prediction[:,idx], zero_division=0)
         for idx in range(total_labels.shape[1])])
    recall = np.array(
        [metrics.recall_score(total_labels[:,idx], prediction[:,idx], zero_division=0)
         for idx in range(total_labels.shape[1])])
    f1 = np.array(
        [metrics.f1_score(total_labels[:,idx], prediction[:,idx], zero_division=0)
         for idx in range(total_labels.shape[1])])
    # f1_micro = np.array(
    #     [metrics.f1_score(total_labels[:,idx], prediction[:,idx], zero_division=0, average='micro')
    #      for idx in range(total_labels.shape[1])])
    confusion_matrix = np.array(
        [metrics.confusion_matrix(total_labels[:,idx], prediction[:,idx], labels=[0, 1]).ravel()
         for idx in range(total_labels.shape[1])])
    # if confusion_matrix.size!=36:
    #     print('not 36')
    classification_report = [
        metrics.classification_report(total_labels[:, idx], prediction[:,idx], output_dict=True, zero_division=0)
        for idx in range(total_labels.shape[1])]
    
    return precision, recall, f1, prediction, confusion_matrix, classification_report

def save_sample_to_file(total_preds, total_labels, total_batch_data,
                        subtask_list, event):

    save_dir = os.path.join('./test-samples', event)
    make_dir_if_not_exists(save_dir)

    assert len(subtask_list) == total_preds.shape[1]
    for subtask_idx, subtask in enumerate(subtask_list):
        filename_dict = {
            (0, 0): os.path.join(save_dir, f"{subtask}-TN.jsonl"),
            (0, 1): os.path.join(save_dir, f"{subtask}-FP.jsonl"),
            (1, 0): os.path.join(save_dir, f"{subtask}-FN.jsonl"),
            (1, 1): os.path.join(save_dir, f"{subtask}-TP.jsonl"),
        }

        batch_data_dict = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
        for data_idx, (label, pred) in enumerate(
                zip(total_labels[:, subtask_idx], total_preds[:, subtask_idx])):
            batch_data_dict[(label, pred)].append(total_batch_data[data_idx])

        for label_pred_tuple, batch_data_list in batch_data_dict.items():
            saveToJSONFile(batch_data_list, filename_dict[label_pred_tuple])
    return True 

def prepare_for_prediction(model, dataloader, device):
    
    total_preds = []
    total_labels = []
    total_batch_data = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_dict = {"input_ids": batch[0].to(device),
                          "entity_start_positions": batch[1].to(device),
                          "y": batch[2].to(device)}

            logits, _ = model(**input_dict)
            
            # Post-model subtask information aggregation.
            preds = torch.sigmoid(logits)
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
            total_labels.append(batch[2].cpu().numpy())
            #total_batch_data += batch['batch_data']
        
    total_preds = np.vstack(total_preds)
    total_labels = np.vstack(total_labels)

    return total_preds, total_labels, total_batch_data

# prediction script
# NOTE We didn't use it
# def make_prediction(model, dataloader, device, threshold=0.5):
#     # run model and predict without having "y" label
#     # only return the prediction
    
#     model.eval()
#     dev_logits = []
#     for step, batch in enumerate(dataloader):

#         input_dict = {"input_ids": batch["input_ids"].to(device),
#                       "entity_start_positions": batch["entity_start_positions"].to(device)}
        
#         logits, _ = model(**input_dict)
        
#         # Post-model subtask information aggregation.
#         logits = list(logits.detach().cpu().numpy())
#         dev_logits += logits
    
#     dev_logits = np.array(dev_logits)
    
#     # Assessment on the results according to labels and logits.  
#     if type(threshold) == float:
#         prediction = (dev_logits > threshold).astype(int)
#     else:
#         prediction = np.vstack([(dev_logits[:,subtask_idx] > threshold[subtask_idx]).astype(int) for subtask_idx in range(len(model.subtasks))])
    
    
#     return prediction


def result_to_tsv(results, model_config, taskname, output_dir):
    # results = loadFromJSONFile(results_file)
    # model_config = loadFromJSONFile(model_config_file)
    # We will save the classifier results and model config for each subtask in this dictionary
    all_subtasks_results_and_model_configs = dict()
    all_task_results_and_model_configs = dict()
    all_task_question_tags = dict()
    tested_tasks = list()
    for key in results:
        if key not in ["best_dev_threshold", "best_dev_F1s", "dev_t_F1_P_Rs"]:
            tested_tasks.append(key)
            results[key]["best_dev_threshold"] = results["best_dev_threshold"][key]
            results[key]["best_dev_F1"] = results["best_dev_F1s"][key]
            results[key]["dev_t_F1_P_Rs"] = results["dev_t_F1_P_Rs"][key]
            all_subtasks_results_and_model_configs[key] = results[key], model_config
    all_task_results_and_model_configs[taskname] = all_subtasks_results_and_model_configs
    all_task_question_tags[taskname] = tested_tasks

    # Read the results for each task and save them in csv file
    # results_tsv_save_file = os.path.join("results", "all_experiments_multitask_bert_entity_classifier_results.tsv")
    # NOTE: After fixing the USER and URL tags

    results_tsv_save_file = os.path.join(output_dir, "result.tsv")
    with open(results_tsv_save_file, "a") as tsv_out:
        writer = csv.writer(tsv_out, delimiter='\t')
        # header = ["Event", "Sub-task",  "model name", "accuracy", "CM", "pos. F1", "dev_threshold", "dev_N",
        #           "dev_F1", "dev_P", "dev_R", "dev_TP", "dev_FP", "dev_FN", "N", "F1", "P", "R", "TP", "FP", "FN"]
        # writer.writerow(hesader)
        for taskname, question_tags in all_task_question_tags.items():
            current_task_results_and_model_configs = all_task_results_and_model_configs[taskname]
            for question_tag in question_tags:
                results_sub, model_config = current_task_results_and_model_configs[question_tag]
                # Extract results_sub
                classification_report = results_sub["Classification Report"]
                positive_f1_classification_report = classification_report['1.0']['f1-score']
                accuracy = classification_report['accuracy']
                CM = results_sub["CM"]
                # Best threshold and dev F1
                best_dev_threshold = results_sub["best_dev_threshold"]
                dev_t_F1_P_Rs = results_sub["dev_t_F1_P_Rs"]
                best_dev_threshold_index = [info[0] for info in dev_t_F1_P_Rs].index(best_dev_threshold)
                # Each entry in dev_t_F1_P_Rs is of the format t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN
                t, dev_F1, dev_P, dev_R, dev_N, dev_TP, dev_FP, dev_FN = dev_t_F1_P_Rs[best_dev_threshold_index]
                # Alan's metrics
                F1 = results_sub["F1"]
                P = results_sub["P"]
                R = results_sub["R"]
                TP = results_sub["TP"]
                FP = results_sub["FP"]
                FN = results_sub["FN"]
                N = results_sub["N"]
                # Extract model config
                model_name = model_config["model"]

                row = [taskname, question_tag, model_name, accuracy, CM,
                       positive_f1_classification_report, best_dev_threshold,
                       dev_N, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN,
                       N, F1, P, R, TP, FP, FN]
                writer.writerow(row)

def cal_micro_f1(confusion_matrix):
    TP, FP, FN = 0.0, 0.0, 0.0
    micro_f1_list = []
    len_subtasks = confusion_matrix.shape[0]
    for i in range(len_subtasks):
        cur_confusion_matrix = confusion_matrix[i,:]
        TN_tmp, FP_tmp, FN_tmp, TP_tmp = cur_confusion_matrix.ravel()
        TP += TP_tmp
        FP += FP_tmp
        FN += FN_tmp
    if TP + FP == 0:
        P = 0.0
    else:
        P = TP / (TP + FP)

    if TP + FN == 0:
        R = 0.0
    else:
        R = TP / (TP + FN)
    if P + R == 0:
        F1 = 0.0
    else:
        F1 = 2.0 * P * R / (P + R)
        # micro_f1_list.append(F1)
    return F1

def post_processing(args, model, valid_dataloader, test_dataloader, output_dir, device, verbose=False):
    # Save the model name in the model_config file
    model_config = dict()
    results = dict()
    model_config["model"] = "MultiTaskBertForCovidEntityClassificationShare"
    model_config["epochs"] = args.n_epochs
    # model_config["device"] = device
    
    # Check different thresholds
    #thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = np.arange(0, 1, 0.05)[1:] ## TODO change
    #thresholds = np.arange(0, 1, 0.2)[1:]
    # Evaluate on different thresholds in dev set
    dev_results = []
    for i, t in enumerate(thresholds):
        print(f"\x1b[2K\rPost Processing {i} / {thresholds.shape[0]}"
              f" [{100.0*i/thresholds.shape[0]}%]", end="")
        dev_results.append(evaluation(model, valid_dataloader, device, t))
    print()
    
    # Finding out the best threshold using dev set
    # Getting the f1 scores on different thresholds and in every subtasks
    dev_f1_scores = np.array([dev_results[t_idx][2] for t_idx in range(len(thresholds))]) # (thresholds_idx, subtask_idx)
    # Calculating the best thresholds indices on different subtasks
    best_dev_thresholds_idx = np.argmax(dev_f1_scores, axis=0)
    
    # assert best_dev_thresholds_idx.size == len(model.subtasks, "Expected subtask size: "+str(len(model.subtasks))+" with calculated "+str(best_dev_thresholds_idx.size)+".")
    
    best_dev_F1s = {}
    best_dev_thresholds = {}
    dev_subtasks_t_F1_P_Rs = {subtask: list() for subtask in model.subtasks}
    for subtask_idx in range(best_dev_thresholds_idx.size):
        # Find the subtasks using index
        subtask = model.subtasks[subtask_idx]
        # Find the thresholds of that task using index
        best_thresholds_idx = best_dev_thresholds_idx[subtask_idx]
        
        # Find bests
        # Find the best F1 of that task using index
        best_dev_F1s[subtask] = dev_f1_scores[best_thresholds_idx, subtask_idx]
        # Find the best threshold of that task using index
        best_dev_thresholds[subtask] = thresholds[best_thresholds_idx]
        
        # Log all results in output formats
        for t_idx in range(len(thresholds)):
            dev_P = dev_results[t_idx][0][subtask_idx]
            dev_R = dev_results[t_idx][1][subtask_idx]
            dev_F1 = dev_results[t_idx][2][subtask_idx]
            dev_TN, dev_FP, dev_FN, dev_TP = dev_results[t_idx][4][subtask_idx]
            
            dev_subtasks_t_F1_P_Rs[subtask].append((thresholds[t_idx], dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN)) # Copy-pasted from original code

    results["best_dev_threshold"] = best_dev_thresholds
    results["best_dev_F1s"] = best_dev_F1s
    results["dev_t_F1_P_Rs"] = dev_subtasks_t_F1_P_Rs
    
    # Apply to testset
    # TODO: Squad style and raw style scores are not evaluated here yet.
    
    logging.info("Testing on test dataset")
    # Turn into list
    best_thresholds = [best_dev_thresholds[subtask] for subtask in model.subtasks]
    # Getting test results
    test_result = evaluation(model, test_dataloader, device, best_thresholds,
                             save_sample_flg=False)
    
    for subtask_idx in range(len(best_thresholds)):
        subtask = model.subtasks[subtask_idx]
        results[subtask] = {}
        P = test_result[0][subtask_idx]
        R = test_result[1][subtask_idx]
        F1 = test_result[2][subtask_idx]
        TN, FP, FN, TP = test_result[4][subtask_idx]
        classification_report = test_result[5][subtask_idx]
        
        results[subtask]["CM"] = [TN, FP, FN, TP]  # Storing it as list of lists instead of numpy.ndarray
        results[subtask]["Classification Report"] = classification_report
    
        results[subtask]["F1"] = F1
        results[subtask]["P"] = P
        results[subtask]["R"] = R
        results[subtask]["TN"] = TN
        results[subtask]["TP"] = TP
        results[subtask]["FP"] = FP
        results[subtask]["FN"] = FN
        N = TP + FN
        results[subtask]["N"] = N
        
        if verbose:
            print(results)
            logging.info("New evaluation scores:")
            logging.info(f"F1: {F1}")
            logging.info(f"Precision: {P}")
            logging.info(f"Recall: {R}")
            logging.info(f"True Positive: {TP}")
            logging.info(f"False Positive: {FP}")
            logging.info(f"False Negative: {FN}")

    # Save model_config and results
    model_config_file = os.path.join(output_dir, "model_config.json")
    results_file = os.path.join(output_dir, "results.json")
    logging.info(f"Saving model config at {model_config_file}")
    saveToJSONFile(model_config, model_config_file)
    logging.info(f"Saving results at {results_file}")
    saveToJSONFile(results, results_file) 
    return results, model_config
    
def output_precision_recall_f1(precision, recall, f1, subtask_list=None):
    data = np.vstack([precision, recall, f1])
    if subtask_list is None:
        subtask_list = [f"Label {i}" for i in np.arange(f1.shape[0])]
    table = pd.DataFrame(data, columns=subtask_list, index=["Precision", "Recall", "F1"])
    print(table)
    return table

def h5_load(filename, data_list, dtype=None, verbose=False):
    with h5py.File(filename, 'r') as infile:
        data = []
        for data_name in data_list:
            if dtype is not None:
                temp = np.empty(infile[data_name].shape, dtype=dtype)
            else:
                temp = np.empty(infile[data_name].shape, dtype=infile[data_name].dtype)
            infile[data_name].read_direct(temp)
            data.append(temp)
          
        if verbose:
            print("\n".join(
                "{} = {} [{}]".format(data_name, str(real_data.shape), str(real_data.dtype))
                for data_name, real_data in zip(data_list, data)
            ))
            print()
        return data

class SharedDataset(torch_data.Dataset):
    def __init__(self, x, pos, y):
        self.x = x
        self.pos = pos
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.pos[index], self.y[index]

def my_collate_fn(data):
    length = max(d[0].shape[0] for d in data)
    x = np.empty([len(data), length], dtype=np.int64)
    x.fill(0)
    for i, d in enumerate(data):
        l = d[0].shape[0]
        x[i, 0:l] = d[0]
    
    y = np.vstack([d[2] for d in data])

    # build pos
    pos = np.hstack([d[1] for d in data])
    pos_index = np.arange(pos.shape[0])
    pos = np.vstack([pos_index, pos]).T

    # turn to torch tensor
    x = torch.LongTensor(x)
    y = torch.FloatTensor(y)
    pos = torch.LongTensor(pos)
    
    return x, pos, y

def train(logging, args):
    event_output_dir = os.path.join(args.output_dir)
    make_dir_if_not_exists(event_output_dir)
    
    # parameter setting
    max_len = 100       # TODO: compute the statistic of the length
    # subtask_num = 5     # might need to re-assign value after loading the data
    # pretrained_bert_version = "bert-base-cased"
    # pretrained_bert_version = "digitalepidemiologylab/covid-twitter-bert"
    # pretrained_bert_version = "roberta-base"
    pretrained_bert_version = args.pretrained_model

    if torch.cuda.is_available():
        device = torch.device(args.device)
        logging.info(f"Using {args.device} -- GPU{torch.cuda.get_device_name(0)} to train")
    else:
        device = torch.device("cpu")
        logging.info(f"Using CPU to train")

    # load data
    if args.clean_all:
        if args.new:
            data_folder = os.path.join("temp", "clean")
        else:
            data_folder = os.path.join("temp", "normal")
    else:
        if args.new:
            data_folder = os.path.join("temp", "normal")
        else:
            data_folder = os.path.join("temp", "normal")
    subtask_list = []
    data = {
        "train":{"input_ids":[], "entity_start_positions":[], "labels":[]},
        "valid":{"input_ids":[], "entity_start_positions":[], "labels":[]},
        "test":{"input_ids":[], "entity_start_positions":[], "labels":[]},
        "new":{"input_ids":[], "entity_start_positions":[], "labels":[]},
    }
    for event in EVENT_LIST:
        print(f"loading {event} data")

        # load subtask list
        with open(os.path.join(data_folder, f"{event}_subtask.json"), 'r', encoding='utf-8') as infile:
            event_subtask_list = json.load(infile)
            subtask_list.extend(f"{event}_{t}" for t in event_subtask_list)
        
        # load input_ids, positions, and labels
        for phase in ["train", "valid", "test", "new"]:
            table = pd.read_parquet(os.path.join(data_folder, f"{event}_{phase}.parquet"))
            data[phase]["input_ids"].extend(table["input_ids"].to_list())
            
            entity_start_positions, labels = h5_load(os.path.join(data_folder, f"{event}_{phase}.h5"), data_list=["entity_start_positions", "labels"])
            data[phase]["entity_start_positions"].append(entity_start_positions)
            data[phase]["labels"].append(labels)

    for phase in ["train", "valid", "test", "new"]:
        data[phase]["entity_start_positions"] = np.hstack(data[phase]["entity_start_positions"])
        num_row = sum([l.shape[0] for l in data[phase]["labels"]])
        num_col = sum([l.shape[1] for l in data[phase]["labels"]])
        new_labels = np.zeros([num_row, num_col], dtype=np.int32)
        print(f"{phase} new_labels.shape = {new_labels.shape}")

        offset_row = 0
        offset_col = 0
        for l in data[phase]["labels"]:
            new_labels[offset_row:offset_row+l.shape[0], offset_col:offset_col+l.shape[1]] = l
            offset_row += l.shape[0]
            offset_col += l.shape[1]

        data[phase]["labels"] = new_labels

    print(subtask_list)

    # build dataloader
    train_dataloader = torch_data.DataLoader(
        SharedDataset(data["train"]["input_ids"], data["train"]["entity_start_positions"], data["train"]["labels"]),
        num_workers=2,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=my_collate_fn,
    )
    valid_dataloader = torch_data.DataLoader(
        SharedDataset(data["valid"]["input_ids"], data["valid"]["entity_start_positions"], data["valid"]["labels"]),
        num_workers=2,
        batch_size=128,
        shuffle=False,
        collate_fn=my_collate_fn,
    )
    test_dataloader = torch_data.DataLoader(
        SharedDataset(data["test"]["input_ids"], data["test"]["entity_start_positions"], data["test"]["labels"]),
        num_workers=2,
        batch_size=128,
        shuffle=False,
        collate_fn=my_collate_fn,
    )
    new_dataloader = torch_data.DataLoader(
        SharedDataset(data["new"]["input_ids"], data["new"]["entity_start_positions"], data["new"]["labels"]),
        num_workers=2,
        batch_size=128,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    # data loading
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_version)
    tokenizer.add_tokens(["<E>", "</E>", "<URL>", "@USER"])
    tokenizer.save_pretrained(event_output_dir)
    entity_start_token_id = tokenizer.convert_tokens_to_ids(["<E>"])[0]

    output_hidden_states = True if args.embedding_type > 0 else False
    config = AutoConfig.from_pretrained(
        pretrained_bert_version, output_hidden_states=output_hidden_states)
    config.subtasks   = subtask_list
    config.device     = args.device
    config.f1_loss    = args.f1_loss ##f1 loss flag
    config.weighting  = args.weighting
    config.embedding_type = args.embedding_type
    model = MultiTaskBertForCovidEntityClassificationShare(pretrained_bert_version, config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(args.device)  ## TODO old model move classifier

    # init optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_model = None
    best_score = 0.0
    best_epoch = 0
    training_stats = []
    epoch_train_loss = list()
    # start training
    logging.info(f"Initiating training loop for {args.n_epochs} epochs...")
    total_start_time = time.time()

    if args.batch_size_update != -1:
        accumulation_steps = args.batch_size_update // args.batch_size

    if args.retrain:
        for epoch in range(args.n_epochs):
            model.train()
            pbar = tqdm(train_dataloader)

            # Reset the total loss for each epoch.
            total_train_loss = 0
            avg_train_loss = 0
            train_loss_trajectory = list()

            dev_log_frequency = 5
            n_steps = len(train_dataloader)
            dev_steps = int(n_steps / dev_log_frequency)

            start_time = time.time()
            for step, batch in enumerate(pbar):

                input_dict = {"input_ids": batch[0].to(device),
                              "entity_start_positions": batch[1].to(device),
                              "y": batch[2].to(device)}

                # Forward
                # x = x.to(device)
                # y = y.to(device)
                # subtask = subtask.to(device)
                # entity_position = entity_position.to(device)
                logits, loss = model(**input_dict)
                
                loss.backward()
                total_train_loss += loss.item()
                avg_train_loss = total_train_loss/(step+1)


                elapsed = format_time(time.time() - start_time)
                avg_train_loss = total_train_loss / (step + 1)

                # keep track of changing avg_train_loss
                train_loss_trajectory.append(avg_train_loss)
                pbar.set_description(
                    f"Epoch:{epoch+1}|Avg. Loss:{avg_train_loss:.4f}|Loss:{loss.item():.4f}")

                if args.batch_size_update == -1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                elif (step+1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                # Update the learning rate.
                pbar.update()

                # run validation
                if (step) % dev_steps == 0:
                    # model.eval()
                    precision, recall, f1, prediction, confusion_matrix, classification_report = evaluation(model, valid_dataloader, device=device)
                    print("\nValidation Result.")
                    output_precision_recall_f1(precision, recall, f1, model.subtasks)
                    mf1 = cal_micro_f1(confusion_matrix)
                    print(f"Micro F1 for each task: {mf1}")

                    f1 = np.mean(f1)
                    print(f"Macro F1: {f1}")
                    f1 = mf1  ## comment this if using macro-F1 for sub task
                    if f1 >= best_score:
                        best_model = copy.deepcopy(model.state_dict())
                        best_score = f1
                        best_epoch = epoch
                    model.train()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            training_time = format_time(time.time() - start_time)

            # Record all statistics from this epoch.
            training_stats.append({
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time})

            # Save the loss trajectory
            epoch_train_loss.append(train_loss_trajectory)

    # finish training
    print("Finished Training!")
    logging.info(f"Training complete with total Train time:{format_time(time.time() - total_start_time)}")
    log_list(training_stats)

    print(f"Best Validation Score = {best_score} at {best_epoch}")
    model.load_state_dict(best_model)
    # #model.save_pretrained(event_output_dir)
    torch.save(best_model, os.path.join(event_output_dir, "model.bin"))

    # Plot the train loss trajectory in a plot
    train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
    logging.info(f"Saving the Train loss trajectory at {train_loss_trajectory_plot_file}")
    plot_train_loss(epoch_train_loss, train_loss_trajectory_plot_file)

    # running testing

    # (1) probing threshold for each subtask
    # (2) save threshold 
    # precision, recall, f1, prediction = evaluation(model, test_dataloader, device=device)
    print("Testing Result without Post-processing")
    precision, recall, f1, prediction, confusion_matrix, classification_report = evaluation(model, test_dataloader, device=device)
    mf1 = cal_micro_f1(confusion_matrix)
    print(f"Micro F1 for each task: {mf1}")
    f1 = np.mean(f1)
    print(f"Macro F1: {f1}")

    print("post_processing!")
    results, model_config = post_processing(args, model, valid_dataloader, test_dataloader, event_output_dir, device=device)
    print("generating tsv files!")
    result_to_tsv(results, model_config, event, args.output_dir)
    total_preds, total_labels, total_batch_data = prepare_for_prediction(model,new_dataloader,device)
    threshold = results["best_dev_threshold"]
    prediction = np.vstack( ### (28898, 33)
        [(total_preds[:, i] > threshold[subtask]).astype(int)
         for i,subtask in enumerate(model.subtasks)]).T
    # total_batch_data = [] ##(28898
    start = 0
    for event in EVENT_LIST:
        total_batch_data = []
        with open(os.path.join(data_folder, f"{event}_new_data.json"), 'r', encoding='utf-8') as infile:
            total_batch_data_tmp = json.load(infile)
            total_batch_data.extend(total_batch_data_tmp)

        subtask_list_event = [i for i in model.subtasks if event in i]
        # print(subtask_list_event)
        preds_index = []
        for i,subtask in enumerate(model.subtasks):
            if subtask in subtask_list_event:
                preds_index.append(i)
            # print(i,j)
        preds_index = np.array(preds_index)
        event_prediction = prediction[start:start+len(total_batch_data), preds_index]
        subtask_list_event = [i[len(event)+1:] for i in subtask_list_event]
        prediction_to_submission(event_prediction,total_batch_data,event,subtask_list_event)
        start += len(total_batch_data)

    print("generating submission files!")
    # for event in EVENT_LIST:
    #     new_data_predict(event, logging, args, model)

def main():
    args = parse_arg()
    make_dir_if_not_exists(args.output_dir)

    results_tsv_save_file = os.path.join(args.output_dir, "result.tsv")
    if os.path.exists(results_tsv_save_file):
        os.remove(results_tsv_save_file)
    else:
        print("Can not delete the file as it doesn't exists")

    with open(results_tsv_save_file, "a") as tsv_out:
        writer = csv.writer(tsv_out, delimiter='\t')
        header = ["Event", "Sub-task",  "model name", "accuracy", "CM", "pos. F1", "dev_threshold", "dev_N",
                  "dev_F1", "dev_P", "dev_R", "dev_TP", "dev_FP", "dev_FN", "N", "F1", "P", "R", "TP", "FP", "FN"]
        writer.writerow(header)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logfile =  os.path.join(args.output_dir, "train_output.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

    train(logging, args)

if __name__ == "__main__":
    main()
