from typing import Optional
from torch.functional import Tensor
from transformers import GPTNeoForSequenceClassification, GPT2ForSequenceClassification
from transformers import PretrainedConfig, PreTrainedModel
import torch
from typing import Type

from transformers.tokenization_utils_base import BatchEncoding

class ModulePipe(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        # self.args_pass = args_pass # args that pass on to next layer

    def forward(self, *args, **kwargs):
        # if type(self.module) == torch.nn.Dropout:
        #     return self.module.forward(args[0]), args[1], args[2]
        print(type(self.module), args)
        hidden_states, head_mask, attention_mask = args
        print('hidden_states', hidden_states)
        print('head_mask', head_mask)
        print('attention_mask', attention_mask)
        return self.module.forward(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask), head_mask, attention_mask # outputs, head_mask, attention_mask
        # if self.args_pass:
        #     return (self.module.forward(*args, **kwargs), *[kwargs[key] for key in self.args_pass])
        # return self.module.forward(*args, **kwargs)

class EmbeddingPipe(torch.nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        self.dtype = dtype
        self.embed_dim = config.hidden_size

        self.wte = torch.nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = torch.nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.config = config

        self.drop = torch.nn.Dropout(config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_dropout)

    def forward(self, args: BatchEncoding):
        input_ids = args.get('input_ids')
        past_key_values = args.get('past_key_values')
        attention_mask = args.get('attention_mask')
        token_type_ids = args.get('token_type_ids')
        position_ids = args.get('position_ids')
        head_mask = args.get('head_mask')
        inputs_embeds = args.get('inputs_embeds')

    # def forward(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    # ):
    #     print(input_ids)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_layers) # change len(self.h) to self.config.num_layers
        else:
            past_length = past_key_values[0][0].size(-2)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        # head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        # print(hidden_states)

        return hidden_states, head_mask, attention_mask
        # hidden_states = self.drop(hidden_states)

class GPTNeoForSequenceClassificationPipe(GPTNeoForSequenceClassification):
    def to_layers(self):
        layers = [EmbeddingPipe(self.config, self.dtype)] 
        layers += [ModulePipe(m) for m in self._modules['transformer'].h]
        layers += [self._modules['transformer'].ln_f, self._modules['score']]
        return layers

# class GPT2ForSequenceClassificationPipe(GPT2ForSequenceClassification):
#     def to_layers(self):
#         layers = [
#             self._modules['transformer'].wte, 
#             self._modules['transformer'].wpe,
#             self._modules['transformer'].drop,
#             *self._modules['transformer'].h,
#             self._modules['transformer'].ln_f,
#             self._modules['score'],
#         ]
#         return layers