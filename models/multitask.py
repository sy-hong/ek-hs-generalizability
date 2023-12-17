
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from configs import defaults

"""
Multitask model
- config: a BERTConfig, as used by huggingface
- kwargs: kwargs as generated by utils/model_utils.py
"""
class BertMultitask(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        
        super().__init__(config)

        self.num_labels = kwargs["num_output_labels"]

        self.is_multilabel = kwargs["is_multilabel"]

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, i) for i in self.num_labels])

        self.init_weights()

    """
    Pretrained model initialization (e.g., BertForSequenceClassification)
    - pretrained_model_name_or_path: used by huggingface to identify models
    - model_args: huggingface model args
    - kwargs: huggingface model kwargs
    @return: an initialized model using some pretrained checkpoint
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        return model

    """
    Forward pass of the network
    - dataset_id: index of the task we want (0 for hate speech, 1 for emotion, etc.)
    - x: tensor, (batch_size, seq_length)
    - type_ids: tensor, (batch_size, seq_length)
    - attn_mask: tensor, (batch_size, seq_length)
    @return: logits, predictions for this single task
    """
    def forward(self, dataset_id, x, type_ids, attn_mask):
        
        outputs = self.bert(x, token_type_ids=type_ids, attention_mask=attn_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[dataset_id](pooled_output)

        return logits

    """
    Calculate the loss on one batch
    - dataset_id: index of the task we want (0 for hate speech, 1 for emotion, etc.)
    - x: tensor, (batch_size, seq_length)
    - type_ids: tensor, (batch_size, seq_length)
    - attn_mask: tensor, (batch_size, seq_length)
    - loss_calc: an instance of our loss calculator (utils/loss_utils.py)
    - golds: gold labels, tensor (batch_size, seq_length)
    @return: labels (predicted labels), loss (tensor, usually scalar)
    """
    def get_loss(self, dataset_id, x, type_ids, attn_mask, loss_calc, golds):
       
        logits = self(dataset_id, x, type_ids, attn_mask)

        if self.is_multilabel[dataset_id]:
            labels = (logits.sigmoid() > defaults.MULTILABEL_THRESHOLD) * 1
        else:
            labels = logits.argmax(dim=1)

        return labels, loss_calc.get_loss(logits, golds, dataset_id=dataset_id)

    """
    Predict labels for one batch
    - dataset_id: the ID of the task we want (0 for abusive language, 1 for emotion, etc.)
    - x: tensor, (batch_size, seq_length)
    - type_ids: tensor, (batch_size, seq_length)
    - attn_mask: tensor, (batch_size, seq_length)
    @return: preds (predicted labels)
    """
    def predict(self, dataset_id, x, type_ids, attn_mask):
       
        preds = self(dataset_id, x, type_ids, attn_mask)

        if self.is_multilabel[dataset_id]:
            preds = (preds.sigmoid() > defaults.MULTILABEL_THRESHOLD) * 1
        else:
            preds = preds.argmax(dim=1)

        return preds
