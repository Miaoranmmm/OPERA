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
# Q: What year was the first fax sent?\n A: The first fax was sent in 1843.
        prompt = self.example + question
        #print(prompt)
        response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=max_tokens, temperature=temperature, logprobs= 1, n = n)
        #print(response)
        output = []
        for r in response.choices:
            output.append(r.text.strip())
            #text = response.choices[0].text
        return output
 
tc_task = None
 
def init():
    global tc_task
    example_text = (
    "user: i 'm looking for an expensive restaurant in the center of town .\n"
    "system: would you be interested in the cambridge chop house ? it is a british cuisine restaurant located in the centre of town .\n"
    "user: thank you , i hope its in the expensive price range . and if you could give me the phone number that would be great\n"
    "system: yes , it 's expensive . phone number is 01223359506 .\n"
    "user: i need to book a table for 1 person at 16:45 on friday .\n"
    "system: that booking is unsuccessful . does another time/day work ?\n"
    "user: yes . how about 15:45 on friday instead ?\n"
    "system: booking was successful . the table will be reserved for 15 minutes . reference number is : 10p0levh .. anything else today ?\n"
    "user: i 'm also looking for a place to stay . ideally a hotel with free wifi and parking that is also expensive .\n"
    "system: if you 're looking to stay in the town centre area , there is the gonville hotel . it has free parking and internet and is rated 3 stars . would you like to book ?\n"
    "user: great can you book it for one person , for 4 nights starting friday ?\n"
    "system: your booking was successful starting friday for four nights . your reference number is drw9qm8s . can i help you with anything else ?"
    "user: i also need to book a taxi to commute between the two . i 'd like to arrive prior to the 15:45 time .\n"
    "system: your booking is complete , a grey bmw will be picking you up , the contact number is 07034136573. can i assist you with anything further today ?\n"
    "user: no , thanks for your help .\n\n"
    "user: i am looking for a train from stevenage for a trip to cambridge .\n"
    "system: there are several trains to your specifications . what day and time would you like to leave ?\n"
    "user: i need to leave on wednesday after 10:45\n"
    "system: the earliest train that meets your needs is tr6473 , leaving at 11:54 and arriving in cambridge at 12:43. would you like to book passage on that train ?\n"
    "user: yes i would . there are 3 of us traveling . could we get a private area on the train ?\n"
    "system: certainly , do you have a price range in mind ?\n"
    "user: for the train , no . and i made a mistake . i only need a booking for 1 person .\n"
    "system: not a problem at all . booking for one person was successful , the total fee is 12.8 gbp payable at the station . reference number is : 8wysleg0 . anything else today ?"
    "user: i also need a place to dine . i 'm looking for something cheap in the south .\n"
    "system: i have found 2 options for you : the lucky star , which serves chinese food , and nandos , which serves portuguese food . shall i make a reservation for one of them ?\n"
    "user: please book nandos .\n"
    "system: sure . what say and time are you thinking ? and for how many people ?\n"
    "user: actually , i 'll take care of the reservation . can i just get the phone number and address for nandos , please ?\n"
    "system: yes , the phone number is 01223327908 , and the address is cambridge leisure park clifton way . can i help with anything else ?\n"
    "user: no , that was all the information i needed . thanks so much . have a good day .\n\n"
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
 
    # print("\n---- after processing --- ")
    # output = []
    # for text in generated_text:
    #     index = text.index("\nQ:") if "\nQ:" in text else len(text)
    #     output.append(text[:index])
    # print(output)
#['It is London.', 'The capital of the UK is London.', 'London is the capital of the UK.']
