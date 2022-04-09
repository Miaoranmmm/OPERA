- datasets/: contains multiwoz_qa dataset

- query_generator/ is for training and evaluating query generator

- readers/: for training and evaluating reader model.
   subfolders:
	- t5/
	- fid/
	- rag/

- end2end_train/: training and evaluating end2end models with t5/rag reader

- end2end_eval/ is for:
	- end2end evaluation for end2end/modular models
	- calculating knowledge f1, parent score and entailment score
# MultiwozQA
