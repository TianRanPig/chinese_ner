import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF


class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size

        if need_birnn:
            self.need_birnn = need_birnn
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, inputs_ids, tags, token_type_ids=None, attention_mask=None):
        outputs = self.bert(inputs_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = output[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())
