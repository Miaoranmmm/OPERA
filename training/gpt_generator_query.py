import openai
import os
import json
import argparse
openai.api_key = "OPENAI_API_KEY" # replace by your api key

# This is a wrapper for GPT3 for general text completion, i.e., given input text, what wo generate following that
# The purpose of this one is enable experimenting without using the gpt3 key, and standardize the intput
 
 
class TextCompletion:
    def __init__(self, example_text=""):
        self.example=example_text
    def __call__(self, question="", max_tokens=40, temperature = 0.9, n = 1):
        prompt = self.example + question
        #print(prompt)
        response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=max_tokens, temperature=temperature, logprobs= 1, n = n)
        #print(response)
        output = []
        for r in response.choices:
            output.append(r.text.strip())
        return output
 
tc_task = None
 
def init():
    global tc_task
    example_text = (
    "generate knowledge in English:\n\n"
    "query: Bike parking train station\n"
    "knowledge: Secure bike parking facility with 400 spaces opens at York railway station 24-hour access for those taking out weekly, monthly or annual options at facility managed by Cycle Heaven by Simon_MacMichael Tue, Jan 15, 2013 11:34 3\n\n"
    "query: Cat on train\n"
    "knowledge: Pets on a Train And by pets, we mean dogs, cats and small animals (livestock, even if hes your pet pig that lives indoors and wears a collar, is not permitted on any rail network) are allowed to travel by rail, with a few rules and restrictions to adhere to.\n\n"
    "query: season ticket use train london kings cross\n"
    "knowledge: for a weekly Season Ticket, the price of the ticket divided by 10 for a monthly Season Ticket, the price of the ticket divided by 40 for an annual Season Ticket, the price of the ticket divided by 464 If we have to introduce an emergency timetable, Delay Repay compensation will be re-calculated against this. If peak times are especially bad\n\n"
    "query: restaurant  Cambridge Chop House gluten free\n"
    "knowledge: Sunday lunch is a highlight at The Cambridge Chop House. Beef striploin, pork belly and chicken are served with all the trimmings. Please note that the Set Menu / Pre Theatre menu is available Monday to Thursday 11.30- 18.30. Main Menu Pudding Menu Gluten Free Menu Dairy Free Menu Click here to find out more about our wine and beers Drinks Menu\n\n"
    "query: Anatolia restaurant live music\n"
    "knowledge: Anatolia Up the Stairs Behind the Chapel | Close to the Marina, Greece +30 2424 022851 Website Improve this listing Ranked #76 of 132 Restaurants in Skopelos 271 Reviews Cuisines: Greek IlkkS Helsinki, Finland 56 64 Reviewed 8 July 2019 via mobile A great traditional music restaurant Live Rebetica music after 10.30 PM and good traditional food.\n\n"
    "query: De Luca Cucina gluten free\n"
    "knowledge: De Luca Cucina and Bar is a modern Italian restaurant in the heart of Cambridge City offering great quality food at affordable prices.Perfect for a quick lunch, a romantic evening or group night out, we cater for all tastes and wishes. PRICE RANGE 12 - 31 Special Diets Vegetarian Friendly, Vegan Options, Gluten Free Options Meals\n\n"
    "query: how can I cancel a taxi reservation in Cambridge UK\n"
    "knowledge: Here are the best bars for a night out in Cambridge; Panther Taxis. Panther Taxis have been operating in Cambridge for over 25 years, and currently have a whopping fleet of 500 cabs. Hiring a ...\n\n"
    )
    tc_task = TextCompletion(example_text)
 
# 
def run(data):
    #data = {"job_description":"", "example_text":xxx, "input_start" ="Input: ", "output_start":"Output: ", paras = {}, "new_input"}
    if not tc_task:
        init()
    if not isinstance(data, dict):
        data = json.loads(data)
    question = data['input_text']
    max_tokens = data['max_tokens'] if 'max_tokens' in data else 20
    temperature = data['temperature'] if 'temperature' in data else 0.1
    num = data['num_candidates'] if 'num_candidates' in data else 1
    
    return tc_task(question, max_tokens = max_tokens, temperature = temperature, n = num)
 

def parse_args():
    parser = argparse.ArgumentParser(description="Transform data format for FiD")
    parser.add_argument(
        "--dataset_train",
        type=str,
        default=None,
        help="dir of dataset train file",
    )
    parser.add_argument(
        "--dataset_test",
        type=str,
        default=None,
        help="dir of dataset test file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="path to save generated data",
    )
    args = parser.parse_args()
    if args.dataset_train is None and args.dataset_test is None:
        raise ValueError("Need training/test file.")
    if args.save_dir is None:
        raise ValueError("Need to specify save path.")

    return args

if __name__=='__main__':
    
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    import json
    import re
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    def clean_str(text):
        text = text.replace('\n','')
        cleantext = re.sub(CLEANR, '', text)
        return cleantext

    #examples = []
    #with open('gdp_train_h10000.jsonl', "r", encoding="utf-8") as reader:
    #    for item in jsonlines.Reader(reader):
    #        examples.append((item['context'], item['response']))

    #import random
    #random.seed(2021)
    #sample_list = random.sample(examples,4)
    #question = "Generate responses: \n"
    #for sample in sample_list:
    #    context, response = sample
    #    question += 'Q: '+context +'\n'
    #    question += 'A: '+response +'\n'
    
    for datafile in [args.dataset_train, args.dataset_test]:
        #test_examples = []
        with open(datafile) as reader:
            passages = json.load(reader)
            final_file = []
            for passage_name in list(passages.keys()):
                data = passages[passage_name]
                history = []
                question = ''
                dialogue = data['dialogue']
                question_pos = data['question_pos'][0]
                for turns in dialogue[:question_pos]:
                    history.append(clean_str(turns['utterance1']))
                    history.append(clean_str(turns['utterance2']))
                history = '\n'.join(history)
                question = data['questions'][0]
                response = data['answers'][0]
    
                context = history + "\nUser: " + question + '\n' + "System: "
    
                context = context.replace('START EOS TIL','')
                context = context.replace('EOS','\n')

                data = {}
                data['input_text'] = context
                data['temperature'] = 0.9
                data['max_tokens']  = 64
                generated_text = run(json.dumps(data))
                print(generated_text)
                generated_text_list = generated_text[0].split('\n')
                #print(generated_text_list)
                for text in generated_text_list:
                    if text.startswith('User:'):
                        continue
                    elif text.startswith('System:'):
                        generated_text = text.replace('System:', '', 1).lstrip()
                        #final_file.append(generated_text)
                        break
                    else:
                        generated_text = text.lstrip()
                        #final_file.append(generated_text)
                        break
                record = {}
                record['id'] = passage_name
                record['question'] = question
                record['answer'] = response
                record['pred'] = generated_text
                print(generated_text)
                final_file.append(record)
        
        with open(os.path.join(args.save_dir, os.path.basename(datafile)), 'w') as outfile:
            outfile.write(json.dumps(final_file, indent=4))
        #pickle.dump(gpt3_generated_knowledge, open('gpt3_generated_knowledge_no_prompts.pickle','wb'))
        # # break
 
    # print("\n---- after processing --- ")
    # output = []
    # for text in generated_text:
    #     index = text.index("\nQ:") if "\nQ:" in text else len(text)
    #     output.append(text[:index])
    # print(output)
#['It is London.', 'The capital of the UK is London.', 'London is the capital of the UK.']
