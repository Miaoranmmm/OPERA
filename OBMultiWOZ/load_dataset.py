import datasets
import jsonlines
import os

# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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

# Lint as: python3
"""Corpus for OBMultiWOZ"""


import csv

import datasets

# from datasets import dataclass

_DESCRIPTION = """\s
OBMultiWOZ
"""

_CITATION = """\
OBMultiWOZ
"""

_DOWNLOAD_URL = ""
_WEBPAGE = ""


class TaskConfig(datasets.BuilderConfig):
    """BuilderConfig for Task."""
    def __init__(self, name, description, task, **kwargs) -> None:
        super(TaskConfig).__init__()
        self.task = task
        self.name = name
        self.description = description


class obmultiwoz(datasets.GeneratorBasedBuilder):
    """obmultiwoz"""

    BUILDER_CONFIGS = [
                        TaskConfig(name='taskbot',
                                 description="original TOD turns",
                                 task='taskbot'),
                        TaskConfig(name='taskbot_qa',
                                 description="TOD + answerable QA",
                                 task='taskbot_qa'),
                        TaskConfig(name='taskbot_qa_unans_k',
                                 description="TOD + QA (GPT3 as a knowledge base)",
                                 task='taskbot_qa_unans_k'),
                        TaskConfig(name='taskbot_qa_unans_r',
                                 description="TOD + QA (GPT3 as a policy model)",
                                 task='taskbot_qa_unans_r')]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Context": datasets.Value("string"),
                    "Response": datasets.Value("string"),
                    "Knowledge": datasets.Value("string"),
                    "Selected_knowledge": datasets.Value("string"),
                    "Id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Query": datasets.Value("string"),
                    
                }
            ),
            homepage=_WEBPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):        
        current_dir = 'data/OBMultiWOZ' 
        if self.config.task == 'taskbot':
            train_path = [os.path.join(current_dir, 'multiwoz_train_from_dstc9.jsonl')]
            validation_path = [os.path.join(current_dir, 'multiwoz_val_from_dstc9.jsonl')]
            test_path = [os.path.join(current_dir, 'multiwoz_test_from_dstc9.jsonl')]
        elif self.config.task == 'taskbot_qa':
            train_path = [os.path.join(current_dir, 'multiwoz_train_from_dstc9_plus_ans_turns.jsonl'),
                    os.path.join(current_dir, 'multiwoz_train_from_dstc9_plus_unans_turns_gpt_knowledge.jsonl')]
            validation_path = [os.path.join(current_dir, 'multiwoz_val_from_dstc9_plus_ans_turns.jsonl'),
                    os.path.join(current_dir,'multiwoz_val_from_dstc9_plus_unans_turns_gpt_knowledge.jsonl')]
            test_path = [os.path.join(current_dir, 'multiwoz_test_from_dstc9_qa_turns.jsonl')]
        elif self.config.task == 'taskbot_qa_unans_k':
            train_path = [os.path.join(current_dir, 'multiwoz_train_from_dstc9_plus_ans_turns.jsonl'),
                    os.path.join(current_dir, 'multiwoz_train_from_dstc9_plus_unans_turns_gpt_knowledge.jsonl')]
            validation_path = [os.path.join(current_dir, 'multiwoz_val_from_dstc9_plus_ans_turns.jsonl'),
                    os.path.join(current_dir,'multiwoz_val_from_dstc9_plus_unans_turns_gpt_knowledge.jsonl')]
            test_path = [os.path.join(current_dir, 'multiwoz_test_from_dstc9_unans_turns.jsonl')]
        else:
            train_path = [os.path.join(current_dir, 'multiwoz_train_from_dstc9_plus_ans_turns.jsonl'),
                    os.path.join(current_dir, 'multiwoz_train_from_dstc9_plus_unans_turns_gpt_response.jsonl')]
            validation_path = [os.path.join(current_dir, 'multiwoz_val_from_dstc9_plus_ans_turns.jsonl'),
                    os.path.join(current_dir,'multiwoz_val_from_dstc9_plus_unans_turns_gpt_response.jsonl')]
            test_path = [os.path.join(current_dir, 'multiwoz_test_from_dstc9_unans_turns.jsonl')]
                
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": validation_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]
    def _generate_examples(self, filepath):
        print(filepath)
        key = 0
        for filename in filepath:
            with open(filename, "r", encoding="utf-8") as reader:
            
                for item in jsonlines.Reader(reader):
                    yield key, item
                    key += 1
