import argparse
import logging
import math
import os
import random
import json
import torch.nn as nn
import datasets
import nltk
import numpy as np
import torch
from MultiWOZ_Evaluation.mwzeval.metrics import Evaluator as MultiwozEvaluator
from MultiWOZ_Evaluation.mwzeval.metrics_mix import Evaluator as MultiwozQAEvaluator
import src.search_engine as SearchEngine
from gpt_generator import run as gpt_generator
from gpt_generator_query import run as gpt_generator_query
from datasets import load_dataset, load_metric, DatasetDict, Dataset
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
    PreTrainedModel,
    T5PreTrainedModel,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
    BatchEncoding,
    T5ForConditionalGeneration,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

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
        default="taskbot",
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
        "--max_length", type=int, default=512, help="max length"
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
        help="The maximum total input sequence length of history + question after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=80, #128, #64,
        help="The maximum total sequence length for answer text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="pretrained tokenizer",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="The configuration name of model",
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
        "--per_device_train_batch_size",
        type=int,
        default=10,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=10,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=500, help="do pading"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="output file name"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--pad_to_max_length", type=bool, default=True, help="do pading"
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss", type=bool, default=True, help="do pading"
    )
    parser.add_argument(
        "--save_every_checkpoint", action="store_true"
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
        default=None,#5
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
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

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    import re
    #    re_art = re.compile(r'\b(a|an|the)\b')
    #    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    s = s.strip().lower()
    #    s = re_punc.sub(' ', s)
    #    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


def inference(
        model, 
        tokenizer, 
        input_ids,
        attention_mask=None,
        max_input_length=512,
        **kwargs
    ):
    def generate_list_repeatedly(elements, n):
        result = []
        while len(result) < n:
            result.append(elements[len(result) % len(elements)])

        return result

    def clean_knowledge(text):
        import re
        CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        text = text.split('query:')[0]
        text = text.split('knowledge:')[0]
        cleantext = re.sub(CLEANR, '', text)
        cleantext = re.sub(r'[^\x00-\x7f]',r'', cleantext)
        cleantext = cleantext.strip()
        return cleantext

    predict_bs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
            )
    decoded_bs = tokenizer.batch_decode(predict_bs, skip_special_tokens=True)
    
    padding = "max_length"
    max_source_length = max_input_length
    
    decoded_his = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    knowledges = []
    for i, state in enumerate(decoded_bs):
        if "query" in state.lower():
            decoded_query = state.lower().replace('query:','').strip()
            search_res = SearchEngine.retrieve([decoded_query], 5)
            for search_re in search_res:
                knowledge = ''
                if len(search_re) > 0:
                    knowledge = SearchEngine.clean_str(search_re[0]['content'][0]) # only use first snippet returned from Bing
        
        elif "unanswerable" in state.lower(): # or "unanserable" in state.lower():
            if "generation" in state.lower():
                data = {}
                history = decoded_his[i].split(' EOS ')
                roles = generate_list_repeatedly(['user: ', 'system: '], len(history))
                history = '\n'.join([roles[-(i+1)] + utterance for i, utterance in enumerate(history)])
                data['input_text'] = history + '\nsystem:'
                data['temperature'] = 0.9
                data['max_tokens']  = 64
                generated_text = gpt_generator(json.dumps(data))
                generated_text_list = generated_text[0].split('\n')
                for text in generated_text_list:
                    text = text.lower()
                    if text.startswith('user:'):
                        continue
                    elif text.startswith('system:'):
                        generated_text = text.replace('system:', '', 1).lstrip()
#                        generated_text = text.replace('system:','').replace('user:','')
                        generated_text = generated_text.split('user:')[0]
                        generated_text = generated_text.split('system:')[0]
                        break
                    else:
                        generated_text = text.lstrip()
                        generated_text = generated_text.split('user:')[0]
                        generated_text = generated_text.split('system:')[0]
                        break
                knowledge = generated_text
            else:
                data = {}
                data['input_text'] = state.lower().replace('unanswerable:','query:')
                data['temperature'] = 0.9
                data['max_tokens']  = 256
                generated_result = gpt_generator_query(json.dumps(data))
                generated_list = generated_result[0].split('\n\n')
                generated_text_list = []
                for text in generated_list:
                    generated_text_list += text.split('\n')
                generated_text = ''
                backup = '' 
                for text in generated_text_list:
                    if text.startswith('knowledge:'):
                        generated_text = clean_knowledge(text.replace('knowledge:', '', 1).lstrip())
                        if len(generated_text) > 0:
                            break
                        else:
                            continue
                    elif text.startswith('query:'):
                        continue
                    else:
                        backup = clean_knowledge(text)

                if len(generated_text) == 0:
                    generated_text = backup
                knowledge = generated_text

        else:
            knowledge = state.replace('DST:','')
    
        knowledges.append(knowledge)


    res_inputs = []
    for decoded_h, knowledge in zip(decoded_his, knowledges):
        res_inputs.append('Reponse Generation: ' + decoded_h.replace('Tool Set :', '') + ' <|knowledge|> ' + knowledge)
        
    encoded_inputs = tokenizer(res_inputs, max_length=max_source_length, padding=padding, truncation=True)
    encoded_inputs = {k: torch.tensor(v).cuda() for k,v in encoded_inputs.items()}        
    ans = model.generate(
            inputs=encoded_inputs['input_ids'],
            attention_mask=encoded_inputs['attention_mask'],
            **kwargs
            )
    return predict_bs, ans


def main():
    args = parse_args()
    
    if args.output_dir is not None:
        log_dir = os.path.join(args.output_dir,'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_name = 'eval_{}_seed{}.log'.format(args.output_name, args.seed)
        log_file = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    t5_models = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b"]
    

    accelerator = Accelerator()
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

    if args.seed is not None:
        set_seed(args.seed)

    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir = args.cache_dir, split = 'test')
#        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir = args.cache_dir, split = 'train')
    else:
        raise ValueError(
            "Test dataset is required."
        )
    
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
#        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        model = torch.load(
                    os.path.join(args.model_name_or_path, 'pytorch_model.bin')
                )
        print(model)
    else:
        raise ValueError(
            "Please provide model to evaluate"
        )

    if args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens('<|knowledge|>')

    model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args.pad_to_max_length else False
    max_source_length = args.max_history_length
    max_target_length = args.max_target_length

    def preprocess_function(examples):
        passage_ids = examples['Id']
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
                inputs.append('Tool Set : ' + truncate_string(' EOS '.join(context.split(' EOS ')[-5:]), args.max_history_length))                
                responses_labels.append(response)

        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        passage_ids = tokenizer(passage_ids, max_length=40, padding=padding, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(responses_labels, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["labels"]
        model_inputs["passage_ids"] = passage_ids["input_ids"]
        return model_inputs


    def truncate_string(text, maxlength, backward=True):
        if backward:
            return ' '.join(text.split()[-maxlength:])
        else:
            return ' '.join(text.split()[:maxlength])
    
    
    column_names = list(raw_datasets.features.keys())
    lm_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc=f"Processing dataset",
    )


    eval_dataset = lm_datasets


    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")


    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else query_tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        #model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    

    def postprocess_text(preds, labels):
        preds = [normalize_answer(pred.strip().split('|')[-1]) for pred in preds]
        labels = [normalize_answer(label.strip()) for label in labels]
        return preds, labels

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )


    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    
    if args.dataset_config_name == "taskbot":
        evaluator = MultiwozEvaluator(True, True, False)
    else:
        evaluator = MultiwozQAEvaluator(True, True, False)
    ###### Evaluation ######
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }

    decoded_preds_all = {}
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_bs, generated_tokens = inference(
                    model,
                    tokenizer=tokenizer,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_input_length = min(args.max_history_length + args.max_knowledge_length, args.max_length),
                    **gen_kwargs,
            )
            generated_bs = accelerator.pad_across_processes(
                generated_bs, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch['labels']
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=query_tokenizer.pad_token_id)
            generated_bs = accelerator.gather(generated_bs).cpu().numpy()
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_bs = tokenizer.batch_decode(generated_bs, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels) 
#            logger.info('label: %s', decoded_labels)
            logger.info('preds: %s', decoded_preds)
            id_list = tokenizer.batch_decode(batch['passage_ids'], skip_special_tokens=True)
            for i in range(len(decoded_preds)):
                if id_list[i] not in decoded_preds_all:
                    decoded_preds_all[id_list[i]] = []
                turn_result = {}
                turn_result['state_track'] = decoded_bs[i]
                turn_result['response'] = decoded_preds[i]
                turn_result['labels'] = decoded_labels[i]
                decoded_preds_all[id_list[i]].append(turn_result)

#    results = evaluator.evaluate(decoded_preds_all)
#    logger.info(results)

    import json
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            output_dir_file_name = os.path.join(args.output_dir, 'pred-{}-seed{}.json'.format(args.output_name, args.seed))
            json.dump(decoded_preds_all, open(output_dir_file_name,'w'), indent=4)
            logger.info("Saving model outputs to %s", output_dir_file_name)

#    results = evaluator.evaluate(decoded_preds_all)
#    logger.info(results)


if __name__ == "__main__":
    main()


