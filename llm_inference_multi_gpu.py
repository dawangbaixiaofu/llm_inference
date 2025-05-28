import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from time import time
from datetime import datetime
import csv
import sys

csv.field_size_limit(sys.maxsize) # in case of length of field is too long to handle


def load_csv(path, yield_size=10000):
    # 使用生成器，减少内存或显存占用
    f = open(path, mode='r')
    reader = csv.DictReader(f)
    contexts = []
    ccifs = []

    size = 0
    for row in reader:
        context = row['text']
        ccif = row['ccif_no']
        # todo: filter context which lenght is more than 8k
        size += 1
        contexts.append(context)
        ccifs.append(ccif)
    
        if size % yield_size == 0:
            yield ccifs, contexts
            contexts = []
            ccifs = []
    
    if len(ccifs) > 0:
        yield ccifs, contexts
    
    f.close()


######################################
q1 = """Q:请问客户是否对银行服务满意？"""
q2 = """Q:请问当前的额度对客户来说是否足够？"""


questions = [q1, q2]
from uuid import uuid5
import uuid 

def generate_question_info(questions:list):
    res = []
    namespace = uuid.NAMESPACE_DNS

    for q in questions:
        id_obj = uuid5(namespace, q) # generate question id
        id = id_obj.hex
        res.append({'question': q, 'question_id': id})
    return res 


question_info = generate_question_info(questions)


from itertools import product

def com_quesiton(ccifs:list, question_info:list = question_info):
    res = []
    for ccif, q_info in product(ccifs, question_info):
        temp = {'ccif_no':ccif, 'question': q_info['question'], 'question_id': q_info['question_id']}
        res.append(temp)
    return res 

############################


def template(context, quesitons:list) -> list:
    texts = []
    for q in quesitons:
        prompt = context + '\n' + q
        # few-shot prompting 
        messages = [
            {
                "role": "system",
                "content": "你是一个问答机器人，能够根据我提供的上下文信息进行回答，请使用简体中文进行回答，不要使用英文或者其他语言。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        texts = tokenizer.apply_chat(messages, tokenize=False, add_generation_prompt=True)
    return texts

def templated_contexts(contexts:list):
    text_prompts = []
    for context in contexts:
        text_prompts_per_user:list = template(context, questions)
        text_prompts.extend(text_prompts_per_user)
    return text_prompts


def generate(contexts, company_info, output_file):
    text_prompts = templated_contexts(contexts)
    print(f"contexts after templated count is: {len(text_prompts)}")

    outputs = model.generate(text_prompts, sampling_params)
    file = open(output_file, mode='a+')
    fieldnames = ['ccif_no', 'quesiton_id', 'question', 'answer']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    for output, user in zip(outputs, company_info):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        writer.writerow({'ccif_no': user['ccif_no'], 'question_id':user['question_id'], 'question':user['question'], 'answer':generated_text})
    file.close()

def main(batch_num, start_batch_file, end_batch_file):
    for i in range(start_batch_file, end_batch_file):
        path = f"./input_files_v9/data_batch_{i}_{batch_num}.csv"
        gen_obj = load_csv(path, yield_size=1000)
        print(f"check info batch {i}:")

        output_file = f"./output_files_v9/data_batch_{i}_{batch_num}.csv"
        file = open(output_file, mode='a+')
        fieldnames = ['ccif_no', 'quesiton_id', 'question', 'answer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        file.close()

        for ccifs, contexts in gen_obj:
            start = time()
            print(f"company count is: {len(ccifs)}")
            company_info = com_quesiton(ccifs)
            generate(contexts, company_info, output_file)

            end = time()
            duration = end - start
            print(f"sub batch time used is: {duration/60} min.")


if __name__ == "__main__":
    checkpoint = '/home/v_hooverhuang/model/modelName'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LLM(model=checkpoint, trust_remote_code=True, dtype='float16', tensor_parallel_size=8, max_model_len=20000)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048, skip_special_tokens=True)

    start = time()
    main(batch_num=76, start_batch_file=0, end_batch_file=25)
    end = time()
    duration = end-start
    print(f"time total used is {duration/60} min.")
