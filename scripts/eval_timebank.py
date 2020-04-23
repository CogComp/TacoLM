import math
import numpy as np


def evaluator_timebank(test_file, prediction_file):
    ref_lines = [x.strip() for x in open(test_file).readlines()]
    logit_lines = [x.strip() for x in open(prediction_file).readlines()]

    s_predicted = 0
    l_predicted = 0
    s_labeled = 0
    l_labeled = 0
    s_correct = 0
    l_correct = 0
    total = 0
    correct = 0
    for i, line in enumerate(ref_lines):
        logits = [float(x) for x in logit_lines[i].split()]
        s = 0.0
        for num in logits:
            s += math.exp(num)
        logits = [math.exp(x) / s for x in logits]
        predicted_label = int(np.argmax(np.array(logits)))
        actual_label = int(line.split("\t")[4])
        total += 1

        if predicted_label == 0:
            s_predicted += 1
        else:
            l_predicted += 1

        if actual_label == 0:
            s_labeled += 1
        else:
            l_labeled += 1
        if actual_label == predicted_label:
            correct += 1
            if actual_label == 0:
                s_correct += 1
            else:
                l_correct += 1

    s_predicted = float(s_predicted)
    l_predicted = float(l_predicted)
    s_labeled = float(s_labeled)
    l_labeled = float(l_labeled)
    s_correct = float(s_correct)
    l_correct = float(l_correct)
    correct = float(correct)
    total = float(total)

    print("Acc.: " + str(float(correct) / float(total)))
    p = s_correct / s_predicted
    r = s_correct / s_labeled
    f = 2 * p * r / (p + r)
    print("Less than a day: " + str(f))
    p = l_correct / l_predicted
    r = l_correct / l_labeled
    f = 2 * p * r / (p + r)
    print("Longer than a day: " + str(f))


for i in range(1, 4):
    print("TacoLM SEED " + str(i))
    evaluator_timebank("data/timebank/test.formatted.txt", "eval_results/timebank_{}/bert_outputs.txt".format(str(i)))

for i in range(1, 4):
    print("BERT SEED " + str(i))
    evaluator_timebank("data/timebank/test.formatted.txt", "eval_results/timebank_bert_{}/bert_outputs.txt".format(str(i)))
