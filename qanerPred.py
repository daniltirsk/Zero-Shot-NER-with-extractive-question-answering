from deeppavlov.core.commands.utils import parse_config
from deeppavlov import build_model
import json

model_config = parse_config('qa_squad2_bert')

model = build_model(model_config)

data = json.load(open('test-v2.0.json'))
cqas = []
if data:
    for article in data['data']:
        for par in article['paragraphs']:
            context = par['context']
            for qa in par['qas']:
                q = qa['question']
                ans_text = []
                ans_start = []
                if qa['answers']:
                    for answer in qa['answers']:
                        ans_text.append(answer['text'])
                        ans_start.append(answer['answer_start'])
                else:
                    ans_text = ['']
                    ans_start = [-1]
                cqas.append(((context, q), (ans_text, ans_start)))

contexts = [item[0][0] for item in cqas]
questions = [item[0][1] for item in cqas]

preds = model(contexts,questions)

answers_true = [item[1][0] for item in cqas]
answers_pred = preds[0]
start_ids = preds[1]
scores = preds[2]

with open('preds.txt','w') as out:
  for ans_p,ans_t,start,score in zip(answers_pred, answers_true,start_ids,scores):
    out.write("|".join([ans_p, ans_t[0], str(start),str(score) + '\n']))

