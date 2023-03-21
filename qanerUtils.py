from collections import Counter
import string
import numpy as np
import json

from typing import List, Tuple, Optional, Dict

def bioToSquad(tokens: List[str], mapper: Dict, labels: Optional[List[str]] = None) -> List:
    """
    converts bio labeled text into single span squad format

    Args:
        tokens: List of tokens
        mapper: Dict of entity:question pairs
        labels: List of entities, corresponding to tokens

    Returns:
      A list of ((context, question),(answer_text,answer_start),entity_type)
  """

    context = " ".join(tokens)
    possible_entities = list(mapper.keys())
    answers_dict = {}
    squad = []

    if labels:
        counts = Counter([label for label in labels if label.startswith('B')])
        banned_entities = [i[0][2:] for i in counts.items() if i[1] > 1]

        for i in range(len(labels)):
            token = tokens[i]
            label = labels[i]
            entity = label[2:]

            if label == "O" or entity in banned_entities:
                continue

            if entity not in answers_dict:
                answers_dict[entity] = [[], []]

            if label.startswith("B-"):
                answers_dict[entity][0].append(token)
                if i > 0:
                    answers_dict[entity][1].append(len(" ".join(tokens[:i])) + 1)
                else:
                    answers_dict[entity][1].append(len(" ".join(tokens[:i])))
            elif label.startswith("I-"):
                answers_dict[entity][0][-1] = f"{answers_dict[entity][0][-1]} {token}"

        for item in answers_dict.items():
            entity = item[0]
            if entity in possible_entities:
                possible_entities.remove(entity)

            if entity in mapper:
                question = mapper[entity]
                answer_text, answer_start = item[1]
                new_entry = ((context, question), (tuple(answer_text), tuple(answer_start)), entity)
                squad.append(new_entry)

    for neg_entity in possible_entities:
        question = mapper[neg_entity]
        answer = (("",), (-1,))
        new_entry = ((context, question), answer, neg_entity)
        squad.append(new_entry)

    return squad


def bioDatasetToSquad(dataset: List, mapper: Dict) -> List:
  """
    Converts a bio dataset into squad

    Args:
      dataset: a list of (tokens, entities) pairs
      mapper: mapper: Dict of entity:question pairs

    Returns:
      A list of ((context: str, question: str), (answer_text: List[str],answer_start: List[int]), entity_type: str)
  """
  data = []

  for pair in dataset:
    tokens = pair[0]
    labels = pair[1]
    squadConverted = bioToSquad(tokens,mapper,labels)
    data.extend(squadConverted)

  return data


def getBalancedData(data: List, positive_samples: int = 100, negative_samples: int = 0, seed: int = 42) -> List:
  """
    returns up to a specified number of positive and negative entries for each entity type

    Args:
      data: A list of ((context: str, question: str), (answer_text: List[str],answer_start: List[int]), entity_type: str)
      positive_samples: number of positive samples to return for each entity type
      negative_samples: number of negative samples to return for each entity type

    Returns:
      A list of ((context: str, question: str), (answer_text: List[str],answer_start: List[int]), entity_type: str)
  """

  np.random.seed(seed)

  balancedData = {}

  for entry in data:
    entity = entry[-1]

    if entity not in balancedData:
      balancedData[entity] = {"negative": [], "positive": []}

    answer_text = entry[1][0]

    if answer_text[0] == "":
      balancedData[entity]["negative"].append(entry)
    else:
      balancedData[entity]["positive"].append(entry)

  data = []

  for entry in balancedData.values():
    negative = entry['negative']
    size = negative_samples if negative_samples < len(negative) else len(negative)
    ids = np.random.choice(len(negative), size=size, replace=False)
    negativeSamples = [negative[id] for id in ids]

    positive = entry['positive']
    size = positive_samples if positive_samples < len(positive) else len(positive)
    ids = np.random.choice(len(positive), size=size, replace=False)
    positiveSamples = [positive[id] for id in ids]

    data.extend(negativeSamples + positiveSamples)

  return data


def toSquadJson(data: List) -> Dict:
  """
    Forms a dict in squad format

    Args:
      data: A list of ((context: str, question: str), (answer_text: List[str],answer_start: List[int]), entity_type: str)

    Returns:
      data_json: dataset in squad format
  """

  unique_contexts = {}

  for entry in data:
    context = entry[0][0]
    if context not in unique_contexts:
      unique_contexts[context] = set()
    unique_contexts[context].add(entry)

  data_json = {"data": [{"paragraphs": []}]}

  paragraphs = []
  for context, questions in unique_contexts.items():
    qas = []

    for entry in questions:
      (context, question), (ans_text, ans_start), neg_label = entry
      is_impossible = True if ans_text[0] == "" else False
      qas.append({"question": question, "is_impossible": is_impossible,
                  "answers": [{"text": ans_text[0], "answer_start": ans_start[0]}]})

    paragraphs.append({"context": context, "qas": qas})

  data_json['data'][0]['paragraphs'] = paragraphs

  return data_json


def squadToBio(contexts: List, questions: List, preds: List, mapper: Dict) -> List:
  """
    convert squad model predictions to bio markup

    Args:
      contexts: A list of contexts
      questions: A list of questions
      preds: A list of (predicted_answers, start_ids, scores)
      mapper: A dict mapping entities to questions

    Returns:
      instances: A list of bio markup for each
  """

  q_to_l = {v: k for k, v in mapper.items()}
  step = len(mapper)
  instances = []

  answers_pred = preds[0]
  start_ids = preds[1]
  scores = preds[2]

  for i in range(0, len(answers_pred), step):
    context_tokens = contexts[i].split()
    context_raw = contexts[i]

    pred_tags = ["O" for i in range(len(context_tokens))]

    inst_scores = np.array(scores[i:i + step])
    inds = np.argsort(-inst_scores)

    for ind in inds:
      score = inst_scores[ind]

      answer = answers_pred[i + ind].split()
      start_id = start_ids[i + ind]
      if start_id != -1:
        token_id = context_raw[:start_id].count(" ")
        entity = q_to_l[questions[i + ind]]

        skip = False
        for j in range(token_id, token_id + len(answer)):
          if pred_tags[j] != "O":
            skip = True
            break

        if not skip:
          pred_tags[token_id] = "B-" + entity

          if len(answer) > 1:
            for j in range(token_id + 1, token_id + len(answer)):
              pred_tags[j] = "I-" + entity

    instances.append(pred_tags)

  return instances