from IPython.display import clear_output
import os
import csv 
import pandas as pd 


def human_evaluation():
    # prompt 
    seq = input("select interval to evaluate.(example: 0-5000)")
    start, end = seq.split("-")
    file = f"./interval_{start}_{end}.csv" # samples from total users which content is company's context

    # record current progressing, if interupt exception, then can recover from this break
    record_seq_ccif = 0
    record_seq_question = 0

    record_file = f"./interval_{start}_{end}_process_record.csv"
    if not os.path.isfile(record_file):
        with open(record_file, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=['ccif_seq', 'question_seq'])
            writer.writeheader()
            writer.writerow({"ccif_seq": record_seq_ccif, "question_seq": record_seq_question})
    else:
        with open(record_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_seq_ccif = row['ccif_seq']
                record_seq_question = row['question_seq']
    
    # evaluation result file 
    eval_file = f"./human_evaluation_{start}_{end}.csv"
    if not os.path.isfile(eval_file):
        with open(eval_file, mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=['ccif_no', 'question_id', 'question', 'answer', 'human_eval', 'llm_ans'])
            writer.writeheader()
    
    # context and llm answer
    temp = pd.read_csv(file)
    cur_seq_ccif = 0

    for ccif in temp['ccif_no']:
        cur_seq_ccif += 1
        if cur_seq_ccif < record_seq_ccif:
            continue

        context = temp[temp['ccif_no'] == ccif]['text'].values[0]
        print(context)
        print('**********************************************************')

        table_llm_answer = "" # hive table contains llm's inference result 
        DF = spark.sql(f"select * from {table_llm_answer} where ccif_no = '{ccif}' ").toPandas()

        cur_seq_question = 0
        if cur_seq_ccif > record_seq_ccif:
            record_seq_question = 0
        
        for i in range(len(DF)):
            cur_seq_question += 1
            if cur_seq_question <= record_seq_question:
                continue

            e = DF.loc[i]
            print(e['question'])
            print('--------------------------------------------------------------')
            print(e['answer'])


            recv = input("""human eval and llm ans: 1-yes, 0-no, -1-unknown, -2-llm answer is wrong, s-stop. 
                         example1: 1 -1
                         example2: s
                         example3: -1 -1
                         """)
            if recv == 's':
                print('evaluation process has been saved.')
                return 
            else:
                human_eval, llm_ans = recv.split(" ")
                with open(file=eval_file, mode='a') as f:
                    writer = csv.DictWriter(f, fieldnames=['ccif_no', 'question_id', 'question', 'answer', 'human_eval', 'llm_ans'])
                    writer.writerow({'ccif_no': ccif, 'question_id': e['question_id'], 'question': e['question'], 
                                     'answer': e['answer'], 'human_eval': human_eval, 'llm_ans': llm_ans
                                     })
                
                # save progress state
                with open(record_file, mode='a') as f:
                    writer = csv.DictWriter(f, fieldnames=['ccif_seq', 'question_seq'])
                    writer.writerow({'ccif_seq': cur_seq_ccif, 'question_seq': cur_seq_question})
        clear_output()