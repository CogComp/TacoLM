import math
import numpy as np

def evaluator(test_file, prediction_file):
    ref_lines = [x.strip() for x in open(test_file).readlines()]
    logit_lines = [x.strip() for x in open(prediction_file).readlines()]

    prediction_map = {}
    test_instances = []
    for i, line in enumerate(ref_lines):
        fname = line.split("\t")[-3]
        key_1 = int(line.split("\t")[-2])
        key_2 = int(line.split("\t")[-1])
        if key_1 == key_2:
            continue
        logits = [float(x) for x in logit_lines[i].split()]
        s = 0.0
        for num in logits:
            s += math.exp(num)
        logits = [math.exp(x) / s for x in logits]

        comb_key_1 = fname + " " + str(key_1) + " " + str(key_2)
        comb_key_2 = fname + " " + str(key_2) + " " + str(key_1)

        if comb_key_1 not in prediction_map:
            prediction_map[comb_key_1] = [0.0] * 4
        if comb_key_2 not in prediction_map:
            prediction_map[comb_key_2] = [0.0] * 4

        prediction_map[comb_key_1][0] += logits[0]
        prediction_map[comb_key_1][1] += logits[1]
        prediction_map[comb_key_1][2] += logits[2]
        prediction_map[comb_key_1][3] += logits[3]

        prediction_map[comb_key_2][0] += logits[0]
        prediction_map[comb_key_2][1] += logits[1]
        prediction_map[comb_key_2][2] += logits[3]
        prediction_map[comb_key_2][3] += logits[2]

        gold_label = int(line.split("\t")[4])
        if gold_label in [0, 1]:
            test_instances.append([comb_key_1, gold_label])
        if gold_label in [2]:
            test_instances.append([comb_key_1, 2])
        if gold_label in [3]:
            test_instances.append([comb_key_1, 3])

    correct_map = {}
    predicted_map = {}
    labeled_map = {}
    for key, cur_target in test_instances:
        target_label_id = np.argmax(np.array(prediction_map[key]))

        if cur_target == target_label_id:
            if cur_target not in correct_map:
                correct_map[cur_target] = 0.0
            correct_map[cur_target] += 1.0
        if target_label_id not in predicted_map:
            predicted_map[target_label_id] = 0.0
        predicted_map[target_label_id] += 1.0
        if cur_target not in labeled_map:
            labeled_map[cur_target] = 0.0
        labeled_map[cur_target] += 1.0

    for key in correct_map:
        if key == 0:
            print("NoRel")
        if key == 1:
            print("Coref")
        if key == 2:
            print("Child-Parent")
        if key == 3:
            print("Parent-Child")
        print("precision: " + str(correct_map[key] / predicted_map[key]))
        print("recall: " + str(correct_map[key] / labeled_map[key]))


for i in range(1, 4):
    print("TCS-BERT SEED " + str(i))
    evaluator("data/hieve/test.formatted.txt", "eval_results/hieve_{}/bert_outputs.txt".format(str(i)))

for i in range(1, 4):
    print("BERT SEED " + str(i))
    evaluator("data/hieve/test.formatted.txt", "eval_results/hieve_bert_{}/bert_outputs.txt".format(str(i)))
