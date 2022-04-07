#!/usr/bin/env python
# Modified code from gpt-neox/generate.py and megatron/text_generation_utils.py
# by David Turner (dmturner@princeton.edu). Original code copyright preserved below:
#
# Copyright (c) 2021 Josh Levy-Kramer <josh@levykramer.co.uk>. All rights reserved.
# This file is based on code by the authors denoted below and has been modified from its original version.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import os
import time
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

from megatron import print_rank_0
from megatron import mpu
from megatron.utils import is_mp_rank_0, setup_for_inference_or_eval
from megatron.text_generation_utils import (
    forward_model,
    pad_batch,
    broadcast_terminate_signal,
    get_batch,
)


def generate_embeddings(
    neox_args,
    model,
    context_tokens: List[List[int]],
):
    """
    Generate contextual embeddings from tokenzied texts.


    Args:
        neox_args: NeoXArgs.
        model: a Megatron model.
        context_tokens: the context tokens to generate the embeddings from; unpadded list of lists of tokens ids

    Returns:
        A tuple containting:
            - context_logits: a tensor of shape (batch_size, VOCAB_SIZE) representing the logits for each context
            - top_token_id: the id of the top predicted next token
            - top_token_text: the text of the next predicted top token
            - message: a status message, "Success" if things wen well.

    """

    model.eval()

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(
        context_tokens,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )
    torch.distributed.broadcast(
        token_generation_start_index,
        mpu.get_model_parallel_src_rank(),
        group=mpu.get_model_parallel_group(),
    )

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    # set variables
    batch_size = context_tokens.size(0)

    with torch.no_grad():

        model_inputs = (
            context_tokens,
            position_ids,
            attention_mask,
        )

        logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)

        if logits is not None:  # if pipe parallel, not all ranks return logits

            # Get the logits for each context.
            context_logits = logits[
                torch.arange(batch_size), token_generation_start_index - 1, :
            ]

            # Get top token
            top_token_id = torch.argmax(context_logits, dim=-1).view(-1)
            top_token_id = top_token_id.cpu().numpy().tolist()

            try:
                top_token_text = [
                    neox_args.tokenizer.detokenize([t]) for t in top_token_id
                ]
                message = "Success"
            except KeyError:
                top_word = None
                message = "WARNING: generated token which doesn't exist."

    return context_logits, top_token_id, top_token_text, message


def generate_embeddings_from_prompt(
    neox_args,
    model,
    text: Union[List[str], str],
) -> list[dict]:
    """
    Generates contextual embeddings from raw text and returns them in a dictionary.

    Args:
        neox_args: NeoXArgs.
        model: a Megatron model
        text: either a single prompt (str) or a list of prompts (List[str]).

    Returns:
        a list of dicts containing the following fields:
            - 'context' (the input)
            - 'logits' (the logits returned by the model for the context)
            - 'embedding' (the contextual embdedding)

    """
    eos_token_id = neox_args.tokenizer.eod

    # type check
    assert any(
        [isinstance(text, str), isinstance(text, list)]
    ), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]

    input_count = len(text)
    input_pos = 0

    inference_batch_size = 1
    print_rank_0(f"Using inference batch size of {inference_batch_size}")

    # generate completions
    generated_texts = []
    while True:
        model.module.clear_cache()  # clear kv cache between batches

        start_time = time.time()
        # Tokenize text, and check whether we should terminate process
        terminate_runs = 0
        if input_pos == input_count:
            terminate_runs = 1
        else:

            raw_texts = []
            context_tokens = []
            context_lengths = []
            for b_i in range(inference_batch_size):

                # If we reach the end of the texts, we have a partial batch
                if input_pos >= len(text):
                    break

                raw_text = text[input_pos]
                raw_texts.append(raw_text)
                input_pos += 1

                if raw_text == "":
                    context_tokens.append([eos_token_id])
                else:
                    context_tokens.append(neox_args.tokenizer.tokenize(raw_text))

                context_length = len(context_tokens)

                if context_length >= (neox_args.seq_length // 2):
                    print_rank_0(
                        "\nWarning! Context length",
                        context_length,
                        "\nPlease give smaller context (e.g. half of the "
                        "max sequence length)!",
                    )
                context_lengths.append(context_length)

        if not is_mp_rank_0():
            context_tokens = [neox_args.tokenizer.tokenize("EMPTY TEXT")]
            context_length = [len(context_tokens)]
            terminate_runs = 0

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts

        logits, top_tokens, top_tokens_text, message = generate_embeddings(
            neox_args=neox_args,
            model=model,
            context_tokens=context_tokens,
        )

        logits = logits.cpu().numpy()

        # Only generate output on rank 0
        if is_mp_rank_0():

            # Extract the hidden states for each layer
            for key, val in model.layer_outputs.items():
                # Get the batch hidden of hidden states from this layer. Remember we need to un-pad things
                hidden_states_batch = [
                    val[0][0: context_lengths[b], b, :].numpy()
                    for b in range(inference_batch_size)
                ]

            for raw_text, logit_vec, hidden_states, top_token, top_token_text in zip(
                raw_texts, logits, hidden_states_batch, top_tokens, top_tokens_text
            ):
                data = {
                    "context": raw_text,
                    "top_token_text": top_token_text,
                    "top_token_id": top_token,
                    "logits": logit_vec,
                    "hidden_states": hidden_states,
                    "message": message,
                    "duration_seconds": float(time.time() - start_time),
                }
                generated_texts.append(data)

                print_rank_0(f"Text {input_pos} of {len(text)}, inference_time = {data['duration_seconds']} seconds")

    return generated_texts


def generate_embeddings_input_from_file(
    neox_args,
    model,
    input_file,
    output_file=None,
):
    """
    Generates contextual emdeddings from an input file and writes them to an output file.

    Reads prompts from neox_args.sample_input_file and writes contextual embdeddings to neox_args.sample_output_file

    neox_args: NeoXArgs.
    model: a Megatron model

    input_file: path to input file. Each line in the input file will be treated as separate prompt. The line break at the end of the line is not included in the prompt.
    output_file: file where generation results are to be stored in jsonl format. defaults to input_file+'.output.jsonl' if not defined

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'embedding' (the completion)
        - 'length' (the length of the embedding in number of tokens)
        - 'finished':
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds
    """
    # Read the sample file
    print_rank_0(
        "generate_embeddings_input_from_file() loading input from {}".format(input_file)
    )
    with open(input_file, "r") as f:
        prompts = f.readlines()

    # If the prompts are stored as JSONL, extract the text.
    if input_file.endswith(".jsonl"):
        import json

        prompts = [json.loads(p)["text"] for p in prompts]
    else:
        prompts = [p.strip() for p in prompts]

    prompts = [p for p in prompts if len(p) > 0]
    print_rank_0(
        "generate_embeddings_input_from_file() prompts loaded: {}".format(len(prompts))
    )

    if is_mp_rank_0():
        if output_file is None:
            output_file = str(input_file) + ".output.pickle"
            print_rank_0(
                "generate_embeddings_input_from_file() setting default output file to {}".format(
                    output_file
                )
            )

    print_rank_0("generate_embeddings_input_from_file() generating...")
    embeddings = generate_embeddings_from_prompt(
        neox_args=neox_args,
        model=model,
        text=prompts,
    )

    if is_mp_rank_0():
        print(
            json.dumps(
                [{k: v for k, v in e.items() if k not in ['logits', 'hidden_states']} for e in embeddings],
                indent=4,
            )
        )

    print_rank_0("generate_samples_input_from_file() done")
    return embeddings


def main():
    """
    Extract contextual embeddings from text/sample model
    """
    model, neox_args = setup_for_inference_or_eval(use_cache=True)

    # Register hooks on all the layers to get the hidden states back
    model.register_forward_hook(
        layers_to_hook=list(range(2, 46)),
        layer_name_pattern="ParallelTransformerLayerPipe",
    )

    if neox_args.recompute:
        model.module.inference_mode(
            use_cache=False
        )  # don't use kv cache if recomputing

    print_rank_0(
        f"Generating contextual embeddings for samples from input file {neox_args.sample_input_file}"
    )
    assert neox_args.sample_input_file is not None
    results = generate_embeddings_input_from_file(
        neox_args=neox_args,
        model=model,
        input_file=neox_args.sample_input_file,
        output_file=neox_args.sample_output_file,
    )

    if is_mp_rank_0():
        print(f"Saving results to {neox_args.sample_output_file}")
        import pickle
        with open(neox_args.sample_output_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
