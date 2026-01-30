from torch import nn
from transformers import AutoModel

from configuration import config


class ProductClassifier(nn.Module):
    def __init__(self, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = not freeze_bert

        self.linear = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        # last_hidden.shape: [batch_size, seq_len, hidden_size]
        cls_output = last_hidden[:, 0, :]
        # cls_output.shape: [batch_size, hidden_size]
        return self.linear(cls_output)
