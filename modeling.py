def replace_attention_forward(num_psg=100):
    import transformers
    import torch
    import torch.nn as nn

    def attn(self, query_states, key_states, value_states, position_bias):
        batch_size = query_states.size(0)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        # print(query_states.size(), key_states.size(), scores.size(), position_bias.size())
        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        return attn_output

    def forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=key_states.device, dtype=key_states.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if not self.is_decoder:  # Interleaving Attention
            batch_size, n_heads, seq_length, dim_per_head = query_states.size()
            batch_size, n_heads, key_length, dim_per_head = key_states.size()
            real_batch_size = batch_size // num_psg
            query_states = query_states.view(real_batch_size, num_psg, n_heads, seq_length, dim_per_head)
            key_states = key_states.view(real_batch_size, num_psg, n_heads, key_length, dim_per_head)
            value_states = value_states.view(real_batch_size, num_psg, n_heads, key_length, dim_per_head)
            position_bias = position_bias.view(real_batch_size, num_psg, n_heads, seq_length, key_length)

            prompt_q = query_states[:, 0, ...]
            prompt_k = key_states.transpose(1, 2).view(real_batch_size, n_heads, -1, dim_per_head)
            prompt_v = value_states.transpose(1, 2).view(real_batch_size, n_heads, -1, dim_per_head)
            prompt_p = position_bias.transpose(1, 2).view(real_batch_size, n_heads, -1, prompt_k.size(2))
            prompt_attn_output = attn(self, prompt_q, prompt_k, prompt_v, prompt_p)

            context_q = query_states[:, 1:, ...].view(-1, n_heads, seq_length, dim_per_head)
            context_k = torch.cat([key_states[:, :1, ...].repeat(1, num_psg - 1, 1, 1, 1), key_states[:, 1:, ...]],
                                  dim=3).view(-1, n_heads, key_length * 2, dim_per_head)
            context_v = torch.cat([value_states[:, :1, ...].repeat(1, num_psg - 1, 1, 1, 1), value_states[:, 1:, ...]],
                                  dim=3).view(-1, n_heads, key_length * 2, dim_per_head)
            context_p = torch.cat([position_bias[:, :1, ...].repeat(1, num_psg - 1, 1, 1, 1), position_bias[:, 1:, ...]],
                                  dim=3).view(-1, n_heads, seq_length, key_length * 2)
            context_attn_output = attn(self, context_q, context_k, context_v, context_p)

            perm = [i * num_psg for i in range(real_batch_size)]
            perm = perm + [i for i in range(batch_size) if i not in perm]
            perm = [perm.index(i) for i in range(batch_size)]
            attn_output = torch.cat([prompt_attn_output, context_attn_output], dim=0)[perm]
        else:
            position_bias = position_bias[:, :, -query_states.size(2):, :]
            attn_output = attn(self, query_states, key_states, value_states, position_bias)

        attn_weights = None
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    transformers.models.t5.modeling_t5.T5Attention.forward = forward
