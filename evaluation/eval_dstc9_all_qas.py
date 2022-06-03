import copy
import json
import argparse
import jsonlines
from collections import Counter
from datasets import load_metric
from MultiWOZ_Evaluation.mwzeval.metrics import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--unansqa_file",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--ansqa_file",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--taskbot_file",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--gpt_knowledge",
        type=str,
        default='knowledge',
        help='response, knowledge'
    )
    parser.add_argument(
        "--indicator",
        type=str,
        default='qf1',
        help='kf1, qf1, bleu, bleu1'
    )
    args = parser.parse_args()
    assert args.indicator in ['kf1', 'qf1', 'bleu', 'bleu1'], "Choose among `kf1`, `qf1`, `bleu`, `bleu1`."

    return args

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    import re
    s = s.lower()
    s = ' '.join(s.split())
    return s

def normalize_str(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    import re
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

def normalize_str_with_delex(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    import re
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\\^`{|}~\']')
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

def compute_f1(gold, pred):
    gold = normalize_str(gold)
    pred = normalize_str(pred)
    gold_toks = gold.split() if gold else []
    pred_toks = pred.split() if pred else []
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    #print('precision:', precision)
    #print('recall:',recall)
    #print('F1:', f1)
    return f1

def compute_list_f1(gold_list, pred_list):
    results = []
    for gold, pred in zip(gold_list, pred_list):
        results.append(compute_f1(gold, pred))
    return results

def main():
    evaluator = Evaluator(False, True, False)
    metric_bleu_res = load_metric("bleu")
    print('eval loaded')
    args = parse_args()

    def clean_text(pred):
        pred = pred.strip().split('|')[-1]
        pred = normalize_str(pred)
        return pred

    def clean_text_qa(pred):
        pred = pred.strip().split('|')[-1]
        pred = normalize_str_with_delex(pred)
        return pred

    def postprocess_text(pred):
        pred = pred.strip().split('|')[-1]
        pred = normalize_answer(pred)
        return pred
  
    res_predictions = []
    res_labels = []
    qa_references = {}
    with open('/mnt/miaoran/multiwoz_multiwozqa/dataset/dstc9_multiwoz/multiwoz_test_from_dstc9_qa_turns.jsonl') as reader:
        for item in jsonlines.Reader(reader):
            if item['Id'] not in qa_references:
                qa_references[item['Id']] = [{
                        'query': item['Query'],
                        'response': item['Response'],
                        'knowledge': item['Selected_knowledge']
                        }]
            else:
                qa_references[item['Id']].append(
                        {
                        'query': item['Query'],
                        'response': item['Response'],
                        'knowledge': item['Selected_knowledge']
                        }
                    )
    unans_references = {}
    with open('/mnt/miaoran/multiwoz_multiwozqa/dataset/dstc9_multiwoz/dstc9qa_unanswerable/dstc9_qa_turns_unanswerable_processed_test.jsonl') as reader:
        for item in jsonlines.Reader(reader):
#            if args.gpt_knowledge == 'response':
#                query = 'gpt generation'
#            else:
#                query = item['Query']
            query = item['Query']
            if item['Id'] not in unans_references:
                unans_references[item['Id']] = [{
                        'query': query,
                        'response': item['Response'],
                        'knowledge': item['Selected_knowledge']
                        }]
            else:
                unans_references[item['Id']].append(
                        {
                        'query': query,
                        'response': item['Response'],
                        'knowledge': item['Selected_knowledge']
                        }
                    )

    qualified_passages = []
    bleu = load_metric("bleu")

    q_predictions = []
    q_labels = []
    qa_predictions = []
    qa_labels = []
    qa_count = 0
    qa_accuracy = 0
    with open(args.ansqa_file) as f:
        qa_data = json.load(f)
        print("Answerable QA: ", len(qa_data))
        for passage_name in qa_data:
            knowledges = []
            qa_res = []
            qa_res_labels = []
            query_preds = []
            query_labels = []
            qa_turns = qa_data[passage_name]
            for i, turn in enumerate(qa_turns):
                qa_count += 1
                res_predictions.append(clean_text(turn['response']))
                res_labels.append(clean_text(turn['labels']))
                qa_predictions.append(clean_text(turn['response']))
                qa_labels.append(clean_text(turn['labels']))
                qa_res.append(clean_text(turn['response']))
                qa_res_labels.append(clean_text(turn['labels']))
                knowledges.append(clean_text(qa_references[passage_name][i]['knowledge']))
                if 'query' in turn['state_track'].lower():
#                if 'dst' not in turn['state_track'].lower():
                    qa_accuracy += 1
                query_preds.append(clean_text(turn['state_track'].replace('Query:','')))
                query_labels.append(clean_text(qa_references[passage_name][i]['query']))
                q_predictions.append(clean_text(turn['state_track'].replace('Query:','')))
                q_labels.append(clean_text(qa_references[passage_name][i]['query']))

            if args.indicator == 'kf1':
                k_f1s = compute_list_f1(knowledges, qa_res)
                k_f1 = round(sum(k_f1s)/len(k_f1s)*100,2)
                if k_f1 >= 10:
                    qualified_passages.append(passage_name.split('.json')[0].lower())

            elif args.indicator == 'qf1':
                q_f1s = compute_list_f1(query_labels, query_preds)
                q_f1 = round(sum(q_f1s)/len(q_f1s)*100,2)
                if q_f1 >= 20:
                    qualified_passages.append(passage_name.split('.json')[0].lower())
            elif args.indicator == 'bleu':
                bleu_score = bleu.compute(references = [[labels.split()] for labels in qa_res_labels], predictions = [pred.split() for pred in qa_res])['bleu']*100
                if bleu_score >= 1:
                    qualified_passages.append(passage_name.split('.json')[0].lower())            

            else: # bleu-1
                bleu_score = bleu.compute(references = [[labels.split()] for labels in qa_res_labels], predictions = [pred.split() for pred in qa_res])['precisions'][0]*100
                if bleu_score >= 10:
                    qualified_passages.append(passage_name.split('.json')[0].lower())
    #qualified_passages = list(set(qualified_passages))
    print('Qualified Answerable QA: ', len(qualified_passages))
    print(qa_accuracy)
    unans_qualified_passages = []
    with open(args.unansqa_file) as f:
        unans_qa_data = json.load(f)
        print('Unanswerable QA: ',len(unans_qa_data))
        for passage_name in unans_qa_data:
            qa_res = []
            qa_res_labels = []
            query_preds = []
            query_labels = []
            qa_turns = unans_qa_data[passage_name]
            for i, turn in enumerate(qa_turns):
                qa_count += 1
                res_predictions.append(clean_text(turn['response']))
                res_labels.append(clean_text(turn['labels']))
                qa_predictions.append(clean_text(turn['response']))
                qa_labels.append(clean_text(turn['labels']))
                qa_res.append(clean_text(turn['response']))
                qa_res_labels.append(clean_text(turn['labels']))
                #if args.gpt_knowledge == 'response':
                #    if 'gpt generation' in turn['state_track'].lower():
                #        qa_accuracy += 1
                #else:
                #    if 'unanswerable' in turn['state_track'].lower():
                #        qa_accuracy += 1
                if 'unanswerable' in turn['state_track'].lower():# or 'unanserable' in turn['state_track'].lower():
                #if 'dst' not in turn['state_track'].lower():
                    qa_accuracy += 1
                query_preds.append(clean_text(turn['state_track'].replace('Unanswerable Response:','').replace('Unanswerable:','')))
                query_labels.append(clean_text(unans_references[passage_name][i]['query']))
                q_predictions.append(clean_text(turn['state_track'].replace('Unanswerable Response:','').replace('Unanswerable:','')))
                q_labels.append(clean_text(unans_references[passage_name][i]['query']))

            if args.indicator == 'qf1':
                q_f1s = compute_list_f1(query_labels, query_preds)
                q_f1 = round(sum(q_f1s)/len(q_f1s)*100,2)
                if q_f1 >= 20:
                    unans_qualified_passages.append(passage_name.split('.json')[0].lower())
            elif args.indicator == 'bleu':
                bleu_score = bleu.compute(references = [[labels.split()] for labels in qa_res_labels], predictions = [pred.split() for pred in qa_res])['bleu']*100
                if bleu_score >= 1:
                    unas_qualified_passages.append(passage_name.split('.json')[0].lower())

            else: # bleu-1
                bleu_score = bleu.compute(references = [[labels.split()] for labels in qa_res_labels], predictions = [pred.split() for pred in qa_res])['precisions'][0]*100
                if bleu_score >= 10:
                    unans_qualified_passages.append(passage_name.split('.json')[0].lower())
    #qualified_passages = list(set(qualified_passages))
    print('Qualified Unans QA: ', len(unans_qualified_passages)) 
    print(qa_accuracy)
    with open(args.taskbot_file) as f:
        data = json.load(f)
    qa_taskbot_dialogs = {}
    qualified_data = {}
   

    taskbot_res_pred = []
    taskbot_res_label = []
    for passage_name in data:
        if passage_name.upper()+'.json' in unans_qa_data or passage_name.upper()+'.json' in qa_data:#qualified_passages:
            if passage_name.upper()+'.json' in unans_qa_data and passage_name.upper()+'.json' in qa_data:
                print(passage_name.upper()+'.json')
            qa_taskbot_dialogs[passage_name] = []
            for i, turn in enumerate(data[passage_name]):
                res_predictions.append(clean_text(turn['response']))
                res_labels.append(clean_text(turn['labels']))
                taskbot_res_pred.append(clean_text(turn['response']))
                taskbot_res_label.append(clean_text(turn['labels']))
                turn_dict = {}
                turn_dict['response'] = postprocess_text(turn['response'])
                qa_taskbot_dialogs[passage_name].append(turn_dict)
            if (passage_name in qualified_passages and passage_name not in list(unans_references.keys())) or (passage_name in unans_qualified_passages and passage_name not in list(qa_references.keys())) or (passage_name in qualified_passages and passage_name in unans_qualified_passages):
                qualified_data[passage_name] = []
                for i, turn in enumerate(data[passage_name]):
                    turn_dict = {}
                    turn_dict['response'] = postprocess_text(turn['response'])
                    qualified_data[passage_name].append(turn_dict)

    print('Total passages: ',len(qa_taskbot_dialogs))
    print('Qualified: ',len(qualified_data))
#    result_inform = evaluator.evaluate(qa_taskbot_dialogs)['success']['inform']['total']
    taskbot_results = evaluator.evaluate(qa_taskbot_dialogs)
    result = evaluator.evaluate(qualified_data)
#    result_success = evaluator.evaluate(qualified_data)['success']['success']['total']
    result_inform = result['success']['inform'].get('total',0)
    result_success = result['success']['success'].get('total',0)
#    print(result_success)
    _res_preds = [i.split() for i in res_predictions]
    _res_labels = [[i.split()] for i in res_labels]
    metric_bleu_res.add_batch(predictions=_res_preds, references=_res_labels)
    bleu_res = metric_bleu_res.compute()

    _taskbot_res_preds = [i.split() for i in taskbot_res_pred]
    _taskbot_res_labels = [[i.split()] for i in taskbot_res_label]
    _qa_res_preds = [i.split() for i in qa_predictions]
    _qa_res_labels = [[i.split()] for i in qa_labels]
    metric_bleu_res.add_batch(predictions=_taskbot_res_preds, references=_taskbot_res_labels)
    taskbot_bleu_res = metric_bleu_res.compute()
    metric_bleu_res.add_batch(predictions=_qa_res_preds, references=_qa_res_labels)
    qa_bleu_res = metric_bleu_res.compute()
    
    
    results = {'full':{}, 'taskbot':{}, 'qa':{}}
    results['full']['bleu'] = round(bleu_res['bleu']*100, 2)
#    results['success'] = {key: round(value*len(qualified_data)/len(qa_taskbot_dialogs),2) for key, value in result_success.items()}
    results['full']['success'] = round(result_success*len(qualified_data)/len(qa_taskbot_dialogs),2)
    results['full']['inform'] = round(result_inform*len(qualified_data)/len(qa_taskbot_dialogs),2) 
    results['full']['combined'] = round(results['full']['bleu'] + 0.5*(results['full']['success'] + results['full']['inform']), 2)
    results['taskbot']['bleu'] = round(taskbot_bleu_res['bleu']*100, 2)
    results['taskbot']['success'] = round(taskbot_results['success']['success'].get('total',0), 2)
    results['taskbot']['inform'] = round(taskbot_results['success']['inform'].get('total',0), 2)
    results['taskbot']['combined'] = round(results['taskbot']['bleu'] + 0.5*(results['taskbot']['success'] + results['taskbot']['inform']), 2)
    results['qa']['accuracy'] = round(qa_accuracy/qa_count * 100, 2)
    results['qa']['success rate'] = round(len(qualified_data)/len(qa_taskbot_dialogs)*100, 2)
    query_f1s = compute_list_f1(q_labels, q_predictions)
    results['qa']['query f1'] = round(sum(query_f1s)/len(query_f1s) *100, 2)
    results['qa']['res bleu'] = round(qa_bleu_res['bleu']*100, 2)

    print(results)


if __name__ == "__main__":
    main()
