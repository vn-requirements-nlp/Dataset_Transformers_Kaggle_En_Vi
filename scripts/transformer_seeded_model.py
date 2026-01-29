# scripts/transformer_seeded_model.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SeededSequenceClassifier(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))

        base_model = AutoModel.from_config(config)
        self.base_model_prefix = base_model.base_model_prefix
        setattr(self, self.base_model_prefix, base_model)

        self.seed_words_dim = int(getattr(config, "seed_words_dim", 0) or 0)
        self.config.seed_words_dim = self.seed_words_dim

        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + self.seed_words_dim, self.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return getattr(self, self.base_model_prefix).get_input_embeddings()

    def set_input_embeddings(self, value):
        return getattr(self, self.base_model_prefix).set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        seed_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        base = getattr(self, self.base_model_prefix)
        outputs = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        if self.seed_words_dim > 0:
            if seed_feats is None:
                seed_feats = torch.zeros(
                    pooled.size(0), self.seed_words_dim, device=pooled.device
                )
            else:
                seed_feats = seed_feats.to(pooled.device)
            pooled = torch.cat([pooled, seed_feats], dim=-1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
