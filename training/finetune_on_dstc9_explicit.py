#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
import json

import datasets
import nltk
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from MultiWOZ_Evaluation.mwzeval.metrics import Evaluator as MultiwozEvaluator
from MultiWOZ_Evaluation.mwzeval.metrics_multiwoz_qa import Evaluator as MultiwozQAEvaluator

from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

import re
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    return s
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.strip()
    s = s.lower()
#    s = re_punc.sub(' ', s)
#    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default='taskbot',
        help="The configuration name of the dataset to use. `taskbot` or `taskbot_qa`",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    ) 
    parser.add_argument(
        "--max_history_length",
        type=int,
        default=256,
        help="The maximum total input history length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_knowledge_length",
        type=int,
        default=512,
        help="The maximum total input history length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=12,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=12,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5, #1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="max length"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=80,
        help="The maximum total sequence length for answer text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length", type=bool, default=True, help="do pading"
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss", type=bool, default=True, help="do pading"
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="do pading"
    )

    parser.add_argument(
        "--save_steps", type=int, default=100000, help="do pading"
    )

    parser.add_argument(
        "--save_every_checkpoint", action="store_true"
    )

    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="max_grad_norm"
    )

    parser.add_argument(
        "--no_kb", action="store_true"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        help="Description to the experiment",
        default='exp',
    )
    parser.add_argument(
        "--format_version",
        type=str, default='v1',
        help="format version"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="cache_dir for huggingface models and datasets",
        default=None,
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    
    if args.output_dir is not None:
        log_dir = args.output_dir
        os.makedirs(log_dir, exist_ok=True)
        log_name = 'train_' + str(args.seed) + '.log'
        log_file = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    t5_models = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b"]
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
#        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir = args.cache_dir, split="train")
#        raw_datasets = raw_datasets.train_test_split(test_size=0.01, seed=args.seed)
#        raw_datasets['validation'] = raw_datasets['test']
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir = args.cache_dir)
        #print(raw_datasets) 
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir = args.cache_dir
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir = args.cache_dir
            )

    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)
    
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
        except:
            model = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin'))
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, cache_dir=args.cache_dir)


    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens('<|knowledge|>')    

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    #max_length = args.max_length
    padding = "max_length" if args.pad_to_max_length else False
    max_source_length = min(args.max_history_length + args.max_knowledge_length, args.max_length)
    max_target_length = args.max_target_length
    
    def preprocess_function(examples):
        passage_ids = examples['Id']
#        turn_ids = examples['turn_id']
        contextes = examples['Context']
        responses = examples['Response']
        kbs = examples['Knowledge']
        selected_kbs = examples['Selected_knowledge']
        querys = examples['Query']
        tasks = examples['Task']
        responses_labels = []
        inputs = []
        
        for context, response, kb, s_kb, query, task in zip(contextes, responses, kbs, selected_kbs, querys, tasks):
            if args.format_version == 'v1':
                #inputs.append('Language Model: ' + ' EOS '.join(context.split(' EOS ')[-10:]).replace('  ',' '))
                #responses_labels.append(response)
                if task == 'TaskBot':
                    inputs.append('Tool Set : ' + truncate_string(' EOS '.join(context.split(' EOS ')[-5:]), args.max_history_length))
                    state = kb.split('|')[0]
                    responses_labels.append("DST: "+state)
                    k = kb.split('|')[0]
                    inputs.append('Reponse Generation: ' + truncate_string(' EOS '.join(context.split(' EOS ')[-5:]), args.max_history_length) +' <|knowledge|> ' + k)
                    responses_labels.append(response)
                elif task == 'QA':
                    inputs.append('Tool Set : ' + truncate_string(' EOS '.join(context.split(' EOS ')[-5:]), args.max_history_length))
                    responses_labels.append("Query: "+query)
                    k = s_kb
                    inputs.append('Reponse Generation: ' + truncate_string(' EOS '.join(context.split(' EOS ')[-5:]), args.max_history_length) +' <|knowledge|> ' + k)
                    responses_labels.append(response)
                else:
                    inputs.append('Reponse Generation: ' + truncate_string(' EOS '.join(context.split(' EOS ')[-5:]), args.max_history_length))
                    responses_labels.append(response)
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(responses_labels, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["labels"]
        
        return model_inputs


    def truncate_string(text, maxlength, backward=True):
        if backward:
            return ' '.join(text.split()[-maxlength:])
        else:
            return ' '.join(text.split()[:maxlength])


    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    
    # del raw_datasets['train']
    # del raw_datasets['test']
    column_names = list(raw_datasets['train'].features.keys())
    
    lm_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc=f"Processing dataset",
    )

    # import pdb
    # pdb.set_trace()

    train_dataset = lm_datasets["train"]
#    train_dataset = train_dataset.select([0, 10, 20, 30, 40, 50])
    eval_dataset = lm_datasets["validation"]
#    test_dataset = lm_datasets["test"]
#    eval_dataset = raw_datasets['validation'].map(
#        preprocess_function_for_eval,
#        batched=True,
#        remove_columns=column_names,
#        num_proc=args.preprocessing_num_workers,
#        load_from_cache_file=False,
#        desc=f"Processing eval dataset",
#    )


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )


    def postprocess_text(preds, labels):
        preds = [normalize_answer(pred.strip().split('|')[-1]) for pred in preds]
        labels = [normalize_answer(label.strip()) for label in labels]
        # rougeLSum expects newline after each sentence
        #preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        #labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
#    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Eval examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    #progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_eval_loss = 100000.0
    no_improvement = 0

#    if args.dataset_config_name == "taskbot":
#        evaluator = MultiwozEvaluator(True, True, False)
#    else:
#        evaluator = MultiwozQAEvaluator(True, True, False)
    metric_bleu = load_metric("bleu")

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            global_steps += 1
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        logger.info(f"  Train Loss:  {(tr_loss - logging_loss)/float(step+1)}")
        logging_loss = tr_loss
        progress_bar.update(step+1)

        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }
        
#        evaluator = Evaluator(True, True, False)
        eval_loss = 0.0
        decoded_preds_all = {}
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                eval_loss += model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        ).loss.item()
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                
                logger.info(decoded_preds)
                
                _decoded_preds = [i.split() for i in decoded_preds]
                _decoded_labels = [[i.split()] for i in decoded_labels]

                metric_bleu.add_batch(predictions=_decoded_preds, references=_decoded_labels)

#                id_list = tokenizer.batch_decode(batch['passage_ids'], skip_special_tokens=True)
#                turn_ids = batch['turn_ids'].tolist()
#                for i in range(len(decoded_preds)):
#                    if id_list[i] not in decoded_preds_all:
#                        decoded_preds_all[id_list[i]] = []
#                    turn_result = {}
#                    turn_result['id'] = turn_ids[i]
#                    turn_result['response'] = decoded_preds[i]
#                    decoded_preds_all[id_list[i]].append(turn_result)
#                    decoded_preds_all[id_list[i]] = sorted(decoded_preds_all[id_list[i]], key=lambda d: d['id'])
        logger.info(f"  Eval Loss:  {eval_loss/float(step+1)}")
#        results = evaluator.evaluate(decoded_preds_all)
#        results = metric_bleu.compute()
#        logger.info(results) 
        result = {}
        bleu = metric_bleu.compute()
        result['bleu'] = round(bleu['bleu']*100, 2)
        logger.info(result)
                
        if args.output_dir is not None and args.save_every_checkpoint:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                checkpoint_prefix = 'checkpoint'
                output_dir = os.path.join(args.output_dir, '{}-epoch-{}-seed{}'.format(checkpoint_prefix, epoch, args.seed if args.seed else 'none'))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                #unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                accelerator.save(unwrapped_model, os.path.join(output_dir, 'pytorch_model.bin'))
                tokenizer.save_pretrained(output_dir)
                config.save_pretrained(output_dir)

                #torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

        elif args.output_dir is not None and eval_loss/float(step+1) < best_eval_loss:
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                output_dir = os.path.join(args.output_dir,'best_eval')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                #unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                accelerator.save(unwrapped_model, os.path.join(output_dir, 'pytorch_model.bin'))
                tokenizer.save_pretrained(output_dir)
                config.save_pretrained(output_dir)
                logger.info("Saving best eval model to %s", output_dir)
                best_eval_loss = eval_loss/float(step+1)
                no_improvement = 0

        elif eval_loss/float(step+1) >= best_eval_loss:
            no_improvement += 1
            accelerator.wait_for_everyone()
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            #unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            accelerator.save(unwrapped_model, os.path.join(output_dir, 'pytorch_model.bin'))
            tokenizer.save_pretrained(output_dir)
            config.save_pretrained(output_dir)


#        if no_improvement > 5:
#            break


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        #unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        accelerator.save(unwrapped_model, os.path.join(output_dir, 'pytorch_model.bin'))
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
