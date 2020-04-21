from pytorch_pretrained_bert.tokenization import BertTokenizer
from word2number import w2n
import random
import re

class TmpArgDimensionFilter:
    def __init__(self):

        self.file_path = "data/samples/tmparg_collection_all.txt"
        ## Required
        self.lines = list(set([x.strip() for x in open(self.file_path).readlines()]))
        self.value_map = {
            "second": 1.0,
            "seconds": 1.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 60.0 * 60.0,
            "hours": 60.0 * 60.0,
            "day": 24.0 * 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "week": 7.0 * 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "month": 30.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "year": 365.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decade": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        self.process_new_forms()

    def get_trivial_floats(self, s):
        try:
            n = float(s)
            return n
        except:
            return None

    def get_surface_floats(self, tokens):
        if tokens[-1] in ["a", "an"]:
            return 1.0
        if tokens[-1] == "several":
            return 4.0
        if tokens[-1] == "many":
            return 10.0
        if tokens[-1] == "some":
            return 3.0
        if tokens[-1] == "few":
            return 3.0
        if tokens[-1] == "tens" or " ".join(tokens[-2:]) == "tens of":
            return 10.0
        if tokens[-1] == "hundreds" or " ".join(tokens[-2:]) == "hundreds of":
            return 100.0
        if tokens[-1] == "thousands" or " ".join(tokens[-2:]) == "thousands of":
            return 1000.0
        if " ".join(tokens[-2:]) in ["a few", "a couple"]:
            return 3.0
        if " ".join(tokens[-3:]) == "a couple of":
            return 2.0
        return None

    def quantity(self, tokens):
        try:
            if self.get_trivial_floats(tokens[-1]) is not None:
                return self.get_trivial_floats(tokens[-1])
            if self.get_surface_floats(tokens) is not None:
                return self.get_surface_floats(tokens)
            string_comb = tokens[-1]
            cur = w2n.word_to_num(string_comb)
            for i in range(-2, max(-(len(tokens)) - 1, -6), -1):
                status = True
                try:
                    _ = w2n.word_to_num(tokens[i])
                except:
                    status = False
                if tokens[i] in ["-", "and"] or status:
                    if tokens[i] != "-":
                        string_comb = tokens[i] + " " + string_comb
                    update = w2n.word_to_num(string_comb)
                    if update is not None:
                        cur = update
                else:
                    break
            if cur is not None:
                return float(cur)
        except Exception as e:
            return None

    def order_number_convert(self, input):
        m = {"first": 1.0, "second": 2.0, "third": 3.0, "fourth": 4.0, "fifth": 5.0, "sixth": 6.0, "seventh": 7.0,
             "eighth": 8.0, "ninth": 9.0, "tenth": 10.0}
        if input in m:
            return m[input]
        return 0.0

    def transform_plural(self, unit):
        transform_map = {
            "second": "seconds",
            "seconds": "seconds",
            "minute": "minutes",
            "minutes": "minutes",
            "hour": "hours",
            "hours": "hours",
            "day": "days",
            "days": "days",
            "week": "weeks",
            "weeks": "weeks",
            "month": "months",
            "months": "months",
            "year": "years",
            "years": "years",
            "decade": "decades",
            "decades": "decades",
            "century": "centuries",
            "centuries": "centuries",
        }
        if unit in transform_map:
            return transform_map[unit]
        return unit

    """
    Requires tokens to be lower cased
    """
    def check_duration_sentences(self, tmparg_tokens):
        unit = ""
        num = -1.0
        for i, token in enumerate(tmparg_tokens):
            if token in self.value_map:
                num_args = []
                for t in tmparg_tokens[0:i]:
                    num_args.append(t)
                num = self.quantity(num_args)
                unit = self.transform_plural(token)
                if num is None:
                    if unit == token:
                        num = 4.0
                    else:
                        num = 1.0
                break
        if unit == "":
            return "NO_UNIT_FOUND"
        ret_str = str(num) + " " + unit
        if "for a second" in " ".join(tmparg_tokens):
            if tmparg_tokens[-1] != "second":
                return "NO_UNIT_FOUND"
        for t in tmparg_tokens:
            if self.order_number_convert(t) > 0.0:
                return "FOUND_UNIT_BUT_NOT_DURATION"
        if tmparg_tokens[0] in ["for"] and "second time" not in " ".join(tmparg_tokens) and "for the second" not in " ".join(tmparg_tokens):
            return ret_str
        return "FOUND_UNIT_BUT_NOT_DURATION"

    def check_frequency_sentences(self, tmparg_tokens):
        unit = ""
        num = -1.0
        quantity_stop = -1
        for i, token in enumerate(tmparg_tokens):
            if token in self.value_map:
                num_args = []
                for t in tmparg_tokens[0:i]:
                    num_args.append(t)
                num = self.quantity(num_args)
                unit = self.transform_plural(token)
                quantity_stop = i
                if num is None:
                    if unit == token:
                        num = 4.0
                    else:
                        num = 1.0
                break
        if unit == "":
            return "NO_UNIT_FOUND"
        valid = False
        start_anchor = max(0, quantity_stop - 5)
        for i, token in enumerate(tmparg_tokens[start_anchor:quantity_stop]):
            if token == "every" or token == "once" or token == "per" or token == "each":
                num /= 1.0
                valid = True
            if token == "twice":
                num /= 2.0
                valid = True
            if token == "times":
                div = self.quantity([tmparg_tokens[start_anchor:quantity_stop][i-1]])
                if div is not None and div > 0.0:
                    num /= div
                    valid = True
            if token == "time":
                num_key = tmparg_tokens[start_anchor:quantity_stop][i-1]
                converted_num_key = self.order_number_convert(num_key)
                if converted_num_key > 0.0:
                    num /= converted_num_key
                    valid = True
                else:
                    return "SKIP_DURATION"

        if tmparg_tokens[0] == "when":
            valid = False
        ret_str = "FOUND_UNIT_BUT_NOT_FREQUENCY"
        if valid:
            ret_str = str(num) + " " + unit
        return ret_str

    def check_typical_sentences(self, tmparg_tokens):
        keywords = {
            "dawns": [1, 0],
            "mornings": [1, 1],
            "noons": [1, 2],
            "afternoons": [1, 3],
            "evenings": [1, 4],
            "dusks": [1, 5],
            "nights": [1, 6],
            "midnights": [1, 7],
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
            "monday": [2, 0],
            "tuesday": [2, 1],
            "wednesday": [2, 2],
            "thursday": [2, 3],
            "friday": [2, 4],
            "saturday": [2, 5],
            "sunday": [2, 6],
            "mondays": [2, 0],
            "tuesdays": [2, 1],
            "wednesdays": [2, 2],
            "thursdays": [2, 3],
            "fridays": [2, 4],
            "saturdays": [2, 5],
            "sundays": [2, 6],
            "january": [3, 0],
            "february": [3, 1],
            "march": [3, 2],
            "april": [3, 3],
            "may": [3, 4],
            "june": [3, 5],
            "july": [3, 6],
            "august": [3, 7],
            "september": [3, 8],
            "october": [3, 9],
            "november": [3, 10],
            "december": [3, 11],
            "januarys": [3, 0],
            "januaries": [3, 0],
            "februarys": [3, 1],
            "februaries": [3, 1],
            "marches": [3, 2],
            "marchs": [3, 2],
            "aprils": [3, 3],
            "mays": [3, 4],
            "junes": [3, 5],
            "julys": [3, 6],
            "julies": [3, 6],
            "augusts": [3, 7],
            "septembers": [3, 8],
            "octobers": [3, 9],
            "novembers": [3, 10],
            "decembers": [3, 11],
            "springs": [4, 0],
            "summers": [4, 1],
            "autumns": [4, 2],
            "falls": [4, 2],
            "winters": [4, 3],
            "spring": [4, 0],
            "summer": [4, 1],
            "autumn": [4, 2],
            "fall": [4, 2],
            "winter": [4, 3],
        }
        if tmparg_tokens[0].lower() in ["until", "when", "while", "during", "as", "since", "following", "after", "before"]:
            return "NO_TYPICAL_FOUND", ""
        ret_pairs = []
        for t in tmparg_tokens:
            if t in keywords:
                ret_pairs.append([t, keywords[t][0]])
        min_val = 10
        selected_t = None
        for t, group_num in ret_pairs:
            if group_num < min_val:
                min_val = group_num
                selected_t = t
        if len(ret_pairs) > 0:
            return selected_t, min_val
        return "NO_TYPICAL_FOUND", ""

    def check_ordering_sentences(self, tokens, tmp_start, tmp_end):
        unit = ""
        num = -1.0
        tmparg_tokens = tokens[tmp_start:tmp_end]
        for i, token in enumerate(tmparg_tokens):
            if token in self.value_map:
                num_args = []
                for t in tmparg_tokens[0:i]:
                    num_args.append(t)
                num = self.quantity(num_args)
                unit = self.transform_plural(token)
                if num is None:
                    if unit == token:
                        num = 4.0
                    else:
                        num = 1.0
                break

        if tmp_start > 0 or unit == "":
            return "NO_ORDERING_FOUND"
        if tmparg_tokens[0] in ["after", "later", "while", "during"] and tmparg_tokens[-1] in self.value_map:
            return str(num) + " " + unit
        if tmparg_tokens[-1] == "later":
            return str(num) + " " + unit

        return "NO_ORDERING_FOUND"

    def check_ordering_phrases(self, tmp_tokens):
        if len(tmp_tokens) < 5:
            return "NO_ORDERING_FOUND"
        if tmp_tokens[0] in ["after", "before", "while", "during", "when"]:
            return " ".join(tmp_tokens)
        return "NO_ORDERING_FOUND"

    def check_boundary_durations(self, tmp_tokens):
        if tmp_tokens[0] in ["today", "yesterday", "tomorrow"]:
            return "1.0 days"
        find_unit = False
        for i, t in enumerate(tmp_tokens):
            if t in ["next", "last", "following", "previous", "recent", "this", "earlier"]:
                find_unit = True
        if tmp_tokens[0] == "in":
            for t in tmp_tokens:
                match = re.match(r'.*([1-2][0-9]{3})', t)
                if match is not None:
                    return "1.0 years"
            find_unit = True
        if find_unit:
            unit = ""
            num = 1.0
            for i, token in enumerate(tmp_tokens):
                if token in self.value_map:
                    num_args = []
                    for t in tmp_tokens[0:i]:
                        num_args.append(t)
                    num = self.quantity(num_args)
                    unit = self.transform_plural(token)
                    if num is None:
                        if unit == token:
                            num = 4.0
                        else:
                            num = 1.0
                    break
            if unit != "":
                return str(num) + " " + unit
        return "NO_BOUNDARY_FOUND"

    def normalize_timex(self, expression):

        u = expression.split()[1]
        v_input = float(expression.split()[0])

        if u in ["instantaneous", "forever"]:
            return u, str(1)

        convert_map = {
            "seconds": 1.0,
            "minutes": 60.0,
            "hours": 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        seconds = convert_map[u] * float(v_input)
        prev_unit = "seconds"
        for i, v in enumerate(convert_map):
            if seconds / convert_map[v] < 0.5:
                break
            prev_unit = v
        if prev_unit == "seconds" and seconds > 60.0:
            prev_unit = "centuries"

        return prev_unit

    def assign_round_soft_label(self, length, target):
        label_vector = [0.157, 0.001, 0.0]
        label_vector_rev = [0.0, 0.001, 0.157]
        ret_vec = [0.0] * length
        ret_vec[target] = 0.683
        for i in range(target + 1, target + 4):
            cur_target = i
            if i >= length:
                cur_target -= length
            ret_vec[cur_target] = label_vector[i - target - 1]
        for i in range(target - 1, target - 4, -1):
            cur_target = i
            ret_vec[cur_target] = max(ret_vec[cur_target], label_vector_rev[i - target + 3])
        return ret_vec

    def get_soft_labels(self, orig_label, tokenizer):
        markers = ["_dur", "_freq", "_bnd"]
        keywords = {
            "second" + markers[0]: [0, 0],
            "seconds" + markers[0]: [0, 0],
            "minute" + markers[0]: [0, 1],
            "minutes" + markers[0]: [0, 1],
            "hour" + markers[0]: [0, 2],
            "hours" + markers[0]: [0, 2],
            "day" + markers[0]: [0, 3],
            "days" + markers[0]: [0, 3],
            "week" + markers[0]: [0, 4],
            "weeks" + markers[0]: [0, 4],
            "month" + markers[0]: [0, 5],
            "months" + markers[0]: [0, 5],
            "year" + markers[0]: [0, 6],
            "years" + markers[0]: [0, 6],
            "decade" + markers[0]: [0, 7],
            "decades" + markers[0]: [0, 7],
            "century" + markers[0]: [0, 8],
            "centuries" + markers[0]: [0, 8],
            "dawns": [1, 0],
            "mornings": [1, 1],
            "noons": [1, 2],
            "afternoons": [1, 3],
            "evenings": [1, 4],
            "dusks": [1, 5],
            "nights": [1, 6],
            "midnights": [1, 7],
            "dawn": [1, 0],
            "morning": [1, 1],
            "noon": [1, 2],
            "afternoon": [1, 3],
            "evening": [1, 4],
            "dusk": [1, 5],
            "night": [1, 6],
            "midnight": [1, 7],
            "monday": [2, 0],
            "tuesday": [2, 1],
            "wednesday": [2, 2],
            "thursday": [2, 3],
            "friday": [2, 4],
            "saturday": [2, 5],
            "sunday": [2, 6],
            "mondays": [2, 0],
            "tuesdays": [2, 1],
            "wednesdays": [2, 2],
            "thursdays": [2, 3],
            "fridays": [2, 4],
            "saturdays": [2, 5],
            "sundays": [2, 6],
            "january": [3, 0],
            "february": [3, 1],
            "march": [3, 2],
            "april": [3, 3],
            "may": [3, 4],
            "june": [3, 5],
            "july": [3, 6],
            "august": [3, 7],
            "september": [3, 8],
            "october": [3, 9],
            "november": [3, 10],
            "december": [3, 11],
            "januarys": [3, 0],
            "januaries": [3, 0],
            "februarys": [3, 1],
            "februaries": [3, 1],
            "marches": [3, 2],
            "marchs": [3, 2],
            "aprils": [3, 3],
            "mays": [3, 4],
            "junes": [3, 5],
            "julys": [3, 6],
            "julies": [3, 6],
            "augusts": [3, 7],
            "septembers": [3, 8],
            "octobers": [3, 9],
            "novembers": [3, 10],
            "decembers": [3, 11],
            "springs": [4, 0],
            "summers": [4, 1],
            "falls": [4, 2],
            "autumns": [4, 2],
            "winters": [4, 3],
            "spring": [4, 0],
            "summer": [4, 1],
            "autumn": [4, 2],
            "fall": [4, 2],
            "winter": [4, 3],
            "after": [5, 0],
            "before": [5, 1],
            "while": [5, 2],
            "during": [5, 2],
            "when": [5, 3],
            "second" + markers[1]: [6, 0],
            "seconds" + markers[1]: [6, 0],
            "minute" + markers[1]: [6, 1],
            "minutes" + markers[1]: [6, 1],
            "hour" + markers[1]: [6, 2],
            "hours" + markers[1]: [6, 2],
            "day" + markers[1]: [6, 3],
            "days" + markers[1]: [6, 3],
            "week" + markers[1]: [6, 4],
            "weeks" + markers[1]: [6, 4],
            "month" + markers[1]: [6, 5],
            "months" + markers[1]: [6, 5],
            "year" + markers[1]: [6, 6],
            "years" + markers[1]: [6, 6],
            "decade" + markers[1]: [6, 7],
            "decades" + markers[1]: [6, 7],
            "century" + markers[1]: [6, 8],
            "centuries" + markers[1]: [6, 8],
            "second" + markers[2]: [7, 0],
            "seconds" + markers[2]: [7, 0],
            "minute" + markers[2]: [7, 1],
            "minutes" + markers[2]: [7, 1],
            "hour" + markers[2]: [7, 2],
            "hours" + markers[2]: [7, 2],
            "day" + markers[2]: [7, 3],
            "days" + markers[2]: [7, 3],
            "week" + markers[2]: [7, 4],
            "weeks" + markers[2]: [7, 4],
            "month" + markers[2]: [7, 5],
            "months" + markers[2]: [7, 5],
            "year" + markers[2]: [7, 6],
            "years" + markers[2]: [7, 6],
            "decade" + markers[2]: [7, 7],
            "decades" + markers[2]: [7, 7],
            "century" + markers[2]: [7, 8],
            "centuries" + markers[2]: [7, 8],
        }
        vocab_indices = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            1: [10, 11, 12, 13, 14, 15, 16, 17],
            2: [18, 19, 20, 21, 22, 23, 24],
            3: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
            4: [37, 38, 39, 40],
            # Note: not in order
            5: [41, 42, 61, 62],
            6: [43, 44, 45, 46, 47, 48, 49, 50, 51],
            7: [52, 53, 54, 55, 56, 57, 58, 59, 60],
        }
        if orig_label not in keywords:
            print("ERROR WHEN MAPPING SOFT LABELS")

        label_group = keywords[orig_label][0]
        label_id = keywords[orig_label][1]
        soft_labels = [0.0] * len(vocab_indices[label_group])
        if label_group in [0, 6, 7]:
            label_vector_map = [
                [0.7412392700826488, 0.24912843928090708, 0.009458406390501939, 0.00016606286478555588,
                 7.30646267660146e-06, 5.120999881870982e-07, 2.807187666675083e-09, 1.1281434169352893e-11,
                 2.2746998844715235e-14],
                [0.1966797321503831, 0.5851870686462161, 0.1966797321503831, 0.018764436502991182,
                 0.0023274596334087274, 0.0003541235975250493, 7.345990219746334e-06, 1.0063714849200163e-07,
                 6.917246071508814e-10],
                [0.006037846064281208, 0.15903304473396812, 0.47317575760473235, 0.2453125506936087,
                 0.08577848758081653, 0.028331933195186416, 0.002224080407310708, 0.00010386603946867244,
                 2.4336806272483765e-06],
                [7.670003601399247e-05, 0.010977975780067789, 0.17749198395637333, 0.3423587734906385,
                 0.26762063340149095, 0.1613272650199883, 0.03558053856215351, 0.004304815288057253,
                 0.00026131446521643803],
                [2.9988887071575075e-06, 0.0012100380728822476, 0.055152791268338504, 0.23782074256001023,
                 0.3042366976664677, 0.26508603808222186, 0.11004905411557955, 0.02384861332046269,
                 0.0025930260253300397],
                [2.0746396147545137e-07, 0.00018172156454292432, 0.017980428741889258, 0.14150527501625504,
                 0.2616505043598367, 0.3002937686386626, 0.20006984338208827, 0.06704560384872892,
                 0.011272646984034582],
                [1.2229783553502831e-09, 4.053791134884565e-06, 0.0015178670785093325, 0.03356114730820127,
                 0.11681011704848474, 0.21514985378380605, 0.32292803014499954, 0.22873846705624448,
                 0.08129046256564142],
                [5.572730351111348e-12, 6.296884648723474e-08, 8.037356297882278e-05, 0.004603998728773229,
                 0.02870209972984776, 0.08174969138345407, 0.25935558127072855, 0.3661526110790698,
                 0.25935558127072855],
                [1.5291249697362676e-14, 5.890006839530834e-10, 2.5628217033951704e-06, 0.0003803288616050351,
                 0.004246905247620703, 0.018704964177227536, 0.12543280829498968, 0.35294802591125274,
                 0.4982844040965849],
            ]
            soft_labels = label_vector_map[label_id]
        elif label_group == 5:
            label_vector_map = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            soft_labels = label_vector_map[label_id]
        else:
            soft_labels = self.assign_round_soft_label(len(soft_labels), label_id)

        label_vocab_id = vocab_indices[label_group][label_id]
        max_len = 12
        pad_size = max_len - len(vocab_indices[label_group])
        assert len(soft_labels) == len(vocab_indices[label_group])
        pad_vec = []
        for i in range(120, pad_size + 120):
            pad_vec.append(i)

        return vocab_indices[label_group] + pad_vec, soft_labels + [0.0] * pad_size, tokenizer.ids_to_tokens[label_vocab_id], tokenizer.convert_ids_to_tokens(vocab_indices[label_group])

    def word_piece_tokenize(self, tokens, verb_pos, tokenizer):
        if verb_pos < 0:
            return None, -1

        ret_tokens = []
        ret_verb_pos = -1
        for i, token in enumerate(tokens):
            if i == verb_pos:
                ret_verb_pos = len(ret_tokens)
                ret_tokens.append(token)
                continue
            sub_tokens = tokenizer.tokenize(token)
            ret_tokens.extend(sub_tokens)

        return ret_tokens, ret_verb_pos

    def process_new_forms(self):
        import time
        all_instances_map = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, do_basic_tokenize=False)
        total_lines = float(len(self.lines))

        start_time = time.time()
        cur_i = 0
        for i, line in enumerate(self.lines):
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(float(i) / total_lines)
                print("Avg Time: " + str(elapsed / float(i - cur_i)))
                print("Remaining: " + str(total_lines - i))
                start_time = time.time()
                cur_i = i
            group = line.split("\t")
            sent = group[0]
            tokens_lower = sent.lower().split()
            prev_sent = group[1].lower()
            next_sent = group[2].lower()
            verb_pos = int(group[3])
            tmp_start = int(group[4])
            tmp_end = int(group[5])

            duration_check = self.check_duration_sentences(tokens_lower[tmp_start:tmp_end])
            frequency_check = self.check_frequency_sentences(tokens_lower[tmp_start:tmp_end])
            typical_check, _ = self.check_typical_sentences(tokens_lower[tmp_start:tmp_end])
            ordering_check = self.check_ordering_phrases(tokens_lower[tmp_start:tmp_end])
            boundary_check = self.check_boundary_durations(tokens_lower[tmp_start:tmp_end])

            """IF FREQ, NO DUR"""
            if (frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY" or frequency_check == "SKIP_DURATION") and frequency_check != "NO_UNIT_FOUND":
                duration_check = "FOUND_UNIT_BUT_NOT_DURATION"

            no_tmp_token = []
            new_verb_pos = -1
            for j, t in enumerate(tokens_lower):
                if tmp_end > j >= tmp_start:
                    continue
                no_tmp_token.append(t)
                if j == verb_pos:
                    new_verb_pos = len(no_tmp_token) - 1

            no_tmp_token, new_verb_pos = self.word_piece_tokenize(no_tmp_token, new_verb_pos, tokenizer)

            if new_verb_pos < 0:
                continue

            verb_separator = "[unused500]"
            no_tmp_token.insert(new_verb_pos, verb_separator)

            prev_two_sents_tokens = ["[CLS]"] + prev_sent.split() + ["[SEP]"] + no_tmp_token
            if len(prev_two_sents_tokens) + len(next_sent.split()) + 4 > 128:
                continue

            mask_mode = False
            dimension_mode = False
            mlm_labels = [-1] * 128
            r = random.random()
            if r < 0.6:
                mask_mode = True
            elif r < 0.7:
                dimension_mode = True
            else:
                verb_no_lemma = "[MASK]"
                for it in range(2 + len(prev_sent.split()), 2 + len(prev_sent.split()) + len(no_tmp_token)):
                    if prev_two_sents_tokens[it] in ["[unused500]"]:
                        continue
                    prob = random.random()

                    if prob < 0.15:
                        prob /= 0.15

                        orig_token = prev_two_sents_tokens[it]
                        # 80% randomly change token to mask token
                        if prob < 0.8:
                            prev_two_sents_tokens[it] = "[MASK]"

                        # 10% randomly change token to random token
                        elif prob < 0.9:
                            prev_two_sents_tokens[it] = random.choice(list(tokenizer.vocab.items()))[0]

                        # -> rest 10% randomly keep current token

                        # append current token to output (we will predict these later)
                        try:
                            mlm_labels[it] = tokenizer.vocab[orig_token]
                        except KeyError:
                            """CHANGED: NEVER PREDICT [UNK]"""
                            mlm_labels[it] = -1
                    else:
                        # no masking token (will be ignored by loss function later)
                        mlm_labels[it] = -1
                """NOTE: NEW"""
                for it in range(0, 2 + len(prev_sent.split())):
                    if prev_two_sents_tokens[it] not in ["[SEP]", "[CLS]"]:
                        prev_two_sents_tokens[it] = "[MASK]"
                new_next_sent = []
                for iit in range(0, len(next_sent.split())):
                    new_next_sent.append("[MASK]")
                next_sent = " ".join(new_next_sent)

            if duration_check != "NO_UNIT_FOUND" and duration_check != "FOUND_UNIT_BUT_NOT_DURATION":
                unit_string = self.normalize_timex(duration_check) + "_dur"
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused501]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["DUR", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if frequency_check != "NO_UNIT_FOUND" and frequency_check != "FOUND_UNIT_BUT_NOT_FREQUENCY" and frequency_check != "SKIP_DURATION":
                unit_string = self.normalize_timex(frequency_check) + "_freq"
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused502]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["FREQ", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if typical_check != "NO_TYPICAL_FOUND":
                unit_string = typical_check
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused503]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["TYP", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if ordering_check != "NO_ORDERING_FOUND":
                unit_string = ordering_check.split()[0]
                content_string = " ".join(ordering_check.split()[1:])
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused504]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " " + content_string + " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["ORD", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

            if boundary_check != "NO_BOUNDARY_FOUND":
                unit_string = self.normalize_timex(boundary_check) + "_bnd"
                soft_label_indices, soft_label_values, rep_label, other_possible_labels = self.get_soft_labels(unit_string, tokenizer)
                target_index = -1
                dimension_marker = "[unused505]"
                if mask_mode or dimension_mode:
                    rr = random.random()
                    if rr < 0.8:
                        rep_label = "[MASK]"
                    elif rr < 0.9:
                        rep_label = random.choice(other_possible_labels)
                if dimension_mode:
                    mlm_labels[len(prev_two_sents_tokens) + 3 + len(next_sent.split())] = tokenizer.vocab[dimension_marker]
                    dimension_marker = "[MASK]"
                sent_string = " ".join(prev_two_sents_tokens) + " [SEP] " + next_sent + " [SEP] [unused500] " + dimension_marker + " "
                if mask_mode:
                    target_index = len(sent_string.split())
                sent_string += rep_label + " [SEP]"

                key = " ".join(tokens_lower) + " " + str(verb_pos)
                if key not in all_instances_map:
                    all_instances_map[key] = []
                all_instances_map[key].append(["BND", sent_string, target_index, soft_label_indices, soft_label_values, mlm_labels])

        care_order = ["DUR", "ORD", "FREQ", "BND", "TYP"]
        f_out = open("data/samples/formatted_for_training.txt", "w")
        count_map = {}
        for key in all_instances_map:
            select = None
            for care_target in care_order:
                for inst in all_instances_map[key]:
                    if inst[0] == care_target:
                        select = inst
                        break
                    if select is not None:
                        break
            if select is None:
                print("ERROR: SELECT CANNOT BE NONE")
            if select[0] not in count_map:
                count_map[select[0]] = 0
            count_map[select[0]] += 1
            f_out.write(select[1] + "\t" + str(select[2]) + "\t" + " ".join([str(x) for x in select[3]]) + "\t" + " ".join([str(x) for x in select[4]]) + "\t" + " ".join([str(x) for x in select[5]]) + "\n")
        print(count_map)


p = TmpArgDimensionFilter()
