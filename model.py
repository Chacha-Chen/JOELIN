from transformers import BertTokenizer, BertTokenizerFast, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
        assert y_pred.ndim == 1
        assert y_true.ndim == 1
        y_true = y_true.type(torch.int64).cuda()
        y_true = F.one_hot(y_true, 2).to(torch.float32).cuda()
        y_pred = F.sigmoid(y_pred).cuda()

        y_pred_New = torch.zeros(y_true.shape,dtype=torch.float32).cuda()
        y_pred_New[:, 1] = y_pred
        y_pred_New[:, 0] = 1 - y_pred

        y_pred = y_pred_New.cuda()

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)

        return 1 - f1.mean()

class MultiTaskBertForCovidEntityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.subtasks = config.subtasks
        # We will create a dictionary of classifiers based on the number of subtasks
        self.classifiers = {subtask: nn.Linear(config.hidden_size, config.num_labels) for subtask in self.subtasks}
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            entity_start_positions,  ## TODO check what is entity_start_positions
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # DEBUG:
        # print("BERT model outputs shape", outputs[0].shape, outputs[1].shape)
        # print(entity_start_positions[:, 0], entity_start_positions[:, 1])

        # OLD CODE:
        # pooled_output = outputs[1]
        #	input      [8,68]
        # NOTE: outputs[0] has all the hidden dimensions for the entire sequence   	output[0]  [8,68,768]
        # We will extract the embeddings indexed with entity_start_positions	# TODO  why start position embedding	output[1]  [8,768]
        pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]

        pooled_output = self.dropout(pooled_output)  ## [batch_size, 768]
        # Get logits for each subtaskx
        # logits = self.classifier(pooled_output)  10 (#subtask batch_size 8 2 (0,1)]
        logits = {subtask: self.classifiers[subtask](pooled_output) for subtask in self.subtasks}

        outputs = outputs[2:] + (logits,) # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # DEBUG:
            # print(f"Logits:{logits.view(-1, self.num_labels)}, \t, Labels:{labels.view(-1)}")
            for i, subtask in enumerate(self.subtasks):
                # print(labels[subtask].is_cuda)
                if i == 0:
                    loss = loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1))
                else:
                    loss += loss_fct(logits[subtask].view(-1, self.num_labels), labels[subtask].view(-1))
            outputs = outputs + (loss,)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MultiTaskBertForCovidEntityClassificationNew(nn.Module):
    def __init__(self, auto_model_version, config):
        super(MultiTaskBertForCovidEntityClassificationNew, self).__init__()
        self.num_labels = config.num_labels
        self.f1_loss    = config.f1_loss
        self.device     = config.device
        self.f1loss     = F1_Loss().to(self.device)
        self.event      = config.event
        self.embedding_type = config.embedding_type


        self.subtasks = config.subtasks
        config.num_labels = len(config.subtasks) ##TODO
        self.weighting = config.weighting
        
        self.encoder = AutoModel.from_pretrained(auto_model_version, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.embedding_type == 2:  # Concat
            self.fc      = nn.Linear(config.hidden_size * 4, config.hidden_size)
        if self.embedding_type == 3:  # multi-head-concat
            self.fc1  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc2  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc3  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc4  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc_final = nn.Linear(config.hidden_size, config.hidden_size)
            
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        ## input [batch size, hidden size]
        ## output [batch size, #subtask]

    def resize_token_embeddings(self, length):
        self.encoder.resize_token_embeddings(length)

    @staticmethod
    def build_loss_weight(y, factor=10):
        weight = (y*(factor-1) + 1) / factor
        return weight

    def forward(self,
                input_ids,
                entity_start_positions,
                labels=None):
        outputs = self.encoder(input_ids)
       
        if self.embedding_type == 0:  # last layer
            pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]
        elif self.embedding_type == 1:  # sum
            hidden_states = torch.stack(
                [x[entity_start_positions[:, 0], entity_start_positions[:, 1], :]
                for x in outputs[2][-4:]], dim=0)
            pooled_output = torch.sum(hidden_states, dim=0)
        elif self.embedding_type == 2:  # Concat
            hidden_states = tuple([x[entity_start_positions[:, 0], entity_start_positions[:, 1], :]
                                for x in outputs[2][-4:]]) 
            pooled_output = self.fc(torch.cat(hidden_states, dim=1))
        elif self.embedding_type == 3:
            hidden_states = []
            hidden_states.append(
                self.fc1(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            hidden_states.append(
                self.fc2(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            hidden_states.append(
                self.fc3(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            hidden_states.append(
                self.fc4(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            pooled_output = self.fc_final(torch.cat(hidden_states, dim=1))

        pooled_output = self.dropout(pooled_output)  ## [batch_size, 768]

        # Get logits for each subtask
        all_logits = self.classifier(pooled_output) #[batch size, # subtask]

        if labels is not None:
            y = torch.stack([labels[subtask] for subtask in labels.keys()], dim =1).type(torch.float)
            if self.weighting:
                weight = self.build_loss_weight(y)
            else:
                weight = None
            
            # TODO: currently, weight is only applicable to BCE loss
            if self.f1_loss:
                loss = 0
                for i in range(len(self.subtasks)):
                    loss += self.f1loss(all_logits[:,i],y[:,i])
            else:
                loss = F.binary_cross_entropy(torch.sigmoid(all_logits), y, weight=weight) 

            output = (all_logits, loss)
        else:
            output = (all_logits, )

        return output  # logits, (loss)

class MultiTaskBertForCovidEntityClassificationShare(nn.Module):
    def __init__(self, auto_model_version, config):
        super(MultiTaskBertForCovidEntityClassificationShare, self).__init__()
        self.num_labels = config.num_labels
        self.f1_loss    = config.f1_loss
        self.device     = config.device
        self.f1loss     = F1_Loss().to(self.device)
        self.embedding_type = config.embedding_type

        self.subtasks = config.subtasks
        config.num_labels = len(config.subtasks) ##TODO
        self.weighting = config.weighting

        self.encoder = AutoModel.from_pretrained(auto_model_version, config=config)
        self.dropout = nn.Dropout(config.__dict__.get("hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if self.embedding_type == 2:  # Concat
            self.fc      = nn.Linear(config.hidden_size * 4, config.hidden_size)

        if self.embedding_type == 3:  # multi-head-concat
            self.fc1  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc2  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc3  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc4  = nn.Linear(config.hidden_size, config.hidden_size // 4)
            self.fc_final = nn.Linear(config.hidden_size, config.hidden_size)

    def resize_token_embeddings(self, length):
        self.encoder.resize_token_embeddings(length)

    def build_loss_weight(self, y, factor=10):
        weight = (y*(factor-1) + 1) / factor
        return weight

    def forward(self,
                input_ids,
                entity_start_positions,
                y=None):

        outputs = self.encoder(input_ids)

        if self.embedding_type == 0:  # last layer
            pooled_output = outputs[0][entity_start_positions[:, 0], entity_start_positions[:, 1], :]
        elif self.embedding_type == 1:  # sum
            hidden_states = torch.stack(
                [x[entity_start_positions[:, 0], entity_start_positions[:, 1], :]
                for x in outputs[2][-4:]], dim=0)
            pooled_output = torch.sum(hidden_states, dim=0)
        elif self.embedding_type == 2:  # Concat
            hidden_states = tuple([x[entity_start_positions[:, 0], entity_start_positions[:, 1], :]
                                for x in outputs[2][-4:]]) 
            pooled_output = self.fc(torch.cat(hidden_states, dim=1))
        elif self.embedding_type == 3:
            hidden_states = []
            hidden_states.append(
                self.fc1(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            hidden_states.append(
                self.fc2(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            hidden_states.append(
                self.fc3(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            hidden_states.append(
                self.fc4(outputs[2][-4][entity_start_positions[:, 0], entity_start_positions[:, 1], :]))
            pooled_output = self.fc_final(torch.cat(hidden_states, dim=1))

        pooled_output = self.dropout(pooled_output)  ## [batch_size, 768]
        
        # Get logits for each subtask
        all_logits = self.classifier(pooled_output) #[batch size, # subtask]

        if y is not None:
            if self.weighting:
                weight = self.build_loss_weight(y)
            else:
                weight = None
            
            # TODO: currently, weight is only applicable to BCE loss
            if self.f1_loss:
                loss = 0
                for i in range(len(self.subtasks)):
                    loss += self.f1loss(all_logits[:,i],y[:,i])
            else:
                loss = F.binary_cross_entropy(torch.sigmoid(all_logits), y, weight=weight) 

            output = (all_logits, loss)
        else:
            output = (all_logits, )

        return output  # logits, (loss)

