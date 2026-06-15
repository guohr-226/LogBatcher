from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass
from typing import Optional, Union
import difflib
from hashlib import sha256
import re
import sys

sys.setrecursionlimit(1000000)
import multiprocessing as mp

import re
import signal


@dataclass
class MatchResult:
    template: str
    template_id: Union[int, str]
    relevant_templates: list
    trusted: bool = True
    match_type: str = "cache"
    best_similarity: float = 1.0
    matched_template: Optional[str] = None

    def __post_init__(self):
        if self.matched_template is None and self.template != "NoMatch":
            self.matched_template = self.template

    def __iter__(self):
        return iter((self.template, self.template_id, self.relevant_templates))

    def __getitem__(self, index):
        return (self.template, self.template_id, self.relevant_templates)[index]

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def safe_search(pattern, string, timeout=1):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = re.search(pattern, string)
    except TimeoutException:
        result = None
    finally:
        signal.alarm(0)
    return result

# _PATTERN = re.compile(r'(?:<\*>|\b\d+\b|[\s\/,:._-]+)')
# def old_standardize(log: str) -> str:
#     return _PATTERN.sub('', log)

# TODO: logb2 v3.1
_PATTERN1 = re.compile(r'/([^/]*)(?=/)')  # path
_PATTERN2 = re.compile(r'\d')               # digit
_PATTERN3 = re.compile(r'[\/:,._-]+')        # : , . _ -
_PATTERN4 = re.compile(r'\s')           # space

def standardize(input_string: str) -> str:
    result = _PATTERN1.sub('', input_string)
    result = _PATTERN2.sub('', result)
    result = _PATTERN3.sub('', result)
    result = _PATTERN4.sub('', result)
    return result

def print_tree(move_tree, indent=' '):
    for key, value in move_tree.items():
        if isinstance(value, dict):
            print(f'{indent}|- {key}')
            print_tree(value, indent + '|  ')
        elif isinstance(value, tuple):
            print(f'{indent}|- {key}: tuple')
        else:
            print(f'{indent}|- {key}: {value}')


def lcs_similarity(X, Y):
    m, n = len(X), len(Y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return 2 * c[m][n] / (m + n)


class ParsingCache(object):
    def __init__(self):
        self.template_tree = {}
        self.template_list = []
        self.hashing_cache = {}
        self.variable_candidates = []
        self.hit_num = 0
        self.template_records = {}

    def _ensure_template_record(self, template_id, template):
        if template_id is None or template_id == "NoMatch":
            return None
        try:
            if template_id < 0:
                return None
        except TypeError:
            return None
        if template_id not in self.template_records:
            self.template_records[template_id] = {
                "template_id": template_id,
                "template": template,
                "hit_count": 0,
                "stable_hit_count": 0,
                "llm_correction_count": 0,
                "conflict_count": 0,
                "router_trigger_count": 0,
                "routed_token_count": 0,
                "risk_score": 0.0,
                "status": "candidate",
            }
        else:
            self.template_records[template_id]["template"] = template
        return self.template_records[template_id]

    def _update_record_status(self, record):
        if record["conflict_count"] >= 5:
            record["status"] = "deprecated"
        elif record["conflict_count"] >= 3 and record["status"] == "risky":
            record["status"] = "split_candidate"
        elif record["risk_score"] >= 0.5 and record["status"] in ("candidate", "stable"):
            record["status"] = "risky"
        elif (
            record["status"] == "candidate"
            and record["stable_hit_count"] >= 3
            and record["conflict_count"] == 0
        ):
            record["status"] = "stable"

    def _is_trusted_template(self, template_id):
        record = self.template_records.get(template_id)
        if record is None:
            return True
        return record.get("status") not in ("risky", "split_candidate", "deprecated")

    def record_cache_hit(self, template_id, template, match_type="cache"):
        record = self._ensure_template_record(template_id, template)
        if record is None:
            return
        record["hit_count"] += 1
        record["stable_hit_count"] += 1
        record["risk_score"] = max(0.0, record["risk_score"] - 0.05)
        self._update_record_status(record)

    def update_by_signal(self, signal):
        if signal.matched_template_id is None:
            return

        template = None
        if isinstance(signal.matched_template_id, int) and signal.matched_template_id < len(self.template_list):
            template = self.template_list[signal.matched_template_id]
        template = template or signal.slm_template or signal.final_template
        record = self._ensure_template_record(signal.matched_template_id, template)
        if record is None:
            return

        record["hit_count"] += 1
        record["router_trigger_count"] += signal.router_trigger_count
        record["routed_token_count"] += signal.routed_token_count

        if signal.llm_used:
            record["llm_correction_count"] += 1

        if signal.template_changed or signal.conflict:
            record["conflict_count"] += 1
            record["risk_score"] += 0.2

        if signal.router_trigger_count > 0:
            record["risk_score"] += 0.1 * signal.router_trigger_count

        if (
            signal.cache_match_type in ("exact", "hash", "legacy_cache", "cache")
            and not signal.llm_used
            and not signal.template_changed
            and not signal.conflict
        ):
            record["stable_hit_count"] += 1
            record["risk_score"] = max(0.0, record["risk_score"] - 0.05)

        self._update_record_status(record)

    def add_templates(self, event_template, insert=True, relevant_templates=[], refer_log = ''):

            # if "<*>" not in event_template:
            #     self.template_tree["$CONSTANT_TEMPLATE$"][event_template] = event_template
            #     continue
            # original_template = event_template
            # event_template = self._preprocess_template(event_template)
            #print("event template after preprocess: ", event_template)
        template_tokens = message_split(event_template)
        if not template_tokens or event_template == "<*>":
            return -1,None,None
        if insert or len(relevant_templates) == 0:
            id = self.insert(event_template, template_tokens, len(self.template_list), refer_log)
            self.template_list.append(event_template)
            return id,None,None
        # print("relevant templates: ", relevant_templates)
        max_similarity = 0
        similar_template = None
        for rt in relevant_templates:
            splited_template1, splited_template2 = rt.split(), event_template.split()
            if len(splited_template1) != len(splited_template2):
                continue 
            similarity = lcs_similarity(splited_template1, splited_template2)
            if similarity > max_similarity:
                max_similarity = similarity
                similar_template = rt
        if max_similarity > 0.8:
            success, id = self.modify(similar_template, event_template, refer_log)
            if not success:
                id = self.insert(event_template, template_tokens, len(self.template_list), refer_log)
                self.template_list.append(event_template)
            return id, similar_template, success
        else:
            id = self.insert(event_template, template_tokens, len(self.template_list), refer_log)
            self.template_list.append(event_template)
            return id,None,None
            #print("template tokens: ", template_tokens)
            
    def insert(self, event_template, template_tokens, template_id, refer_log = ''):

        standardized = standardize(event_template)
        hash_key = sha256(standardized.encode()).hexdigest()
        self.hashing_cache[hash_key] = (standardized, event_template, template_id)

        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            self.template_tree[start_token] = {}
        move_tree = self.template_tree[start_token]

        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                move_tree[token] = {}
            move_tree = move_tree[token]
            tidx += 1

        move_tree["".join(template_tokens)] = (
            sum(1 for s in template_tokens if s != "<*>"),
            template_tokens.count("<*>"),
            event_template,
            template_id,
            refer_log
        )  # statistic length, count of <*>, original_log, template_id
        self._ensure_template_record(template_id, event_template)
        return template_id

    def modify(self, similar_template, event_template, refer_log):
        merged_template = []
        similar_tokens = similar_template.split()
        event_tokens = event_template.split()
        i = 0
        print(similar_template)
        print(event_template)
        for token in similar_tokens:
            print(token, event_tokens[i])
            if token == event_tokens[i]:
                merged_template.append(token)
            else:
                merged_template.append("<*>")
            i += 1
        merged_template = " ".join(merged_template)
        print("merged template: ", merged_template)
        success, old_ids = self.delete(similar_template)
        if not success:
            return False, -1
        self.insert(merged_template, message_split(merged_template), old_ids, refer_log)
        self.template_list[old_ids] = merged_template
        return True, old_ids
        
    
    def delete(self, event_template):
        template_tokens = message_split(event_template)
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            return False, []
        move_tree = self.template_tree[start_token]

        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                return False, []
            move_tree = move_tree[token]
            tidx += 1
        old_id = move_tree["".join(template_tokens)][3]
        del move_tree["".join(template_tokens)]
        return True, old_id


    def match_event(self, log):
        standardized = standardize(log)
        hash_key = sha256(standardized.encode()).hexdigest()
        if hash_key in self.hashing_cache:
            cached_str, template, id = self.hashing_cache[hash_key]
            if cached_str == standardized:
                self.hit_num += 1
                return MatchResult(
                    template=template,
                    template_id=id,
                    relevant_templates=[],
                    trusted=self._is_trusted_template(id),
                    match_type="hash",
                    best_similarity=1.0,
                )
        results = tree_match(self.template_tree, self.template_list, log)
        if results[0] != "NoMatch":
            standardized = standardize(log)
            hash_key = sha256(standardized.encode()).hexdigest()
            self.hashing_cache[hash_key] = (standardized, results[0], results[1])
            return MatchResult(
                template=results[0],
                template_id=results[1],
                relevant_templates=results[2],
                trusted=self._is_trusted_template(results[1]),
                match_type="tree",
                best_similarity=1.0,
            )
        return MatchResult(
            template="NoMatch",
            template_id="NoMatch",
            relevant_templates=results[2],
            trusted=False,
            match_type="nomatch",
            best_similarity=0.0,
        )


    def _preprocess_template(self, template):
        return template


def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')', ";"]
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens


def message_split(message):
    punc = "!\"#$%&'()+,-/;:=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    splitter_regex = re.compile("([{}])".format(splitters))
    tokens = re.split(splitter_regex, message)

    tokens = list(filter(lambda x: x != "", tokens))
    
    #print("tokens: ", tokens)
    tokens = post_process_tokens(tokens, punc)

    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' ' 
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    return tokens



def tree_match(match_tree,template_list, log_content):
    log_tokens = message_split(log_content)
    template, template_id, refer_log, relevant_templates = match_template(match_tree, log_tokens)
    # length matters
    if template:
        if abs(len(log_content.split()) - len(refer_log.split())) <= 1:
            return (template, template_id, relevant_templates)
    elif len(relevant_templates) > 0:
        if match_log(log_content, relevant_templates[0]):
            return (relevant_templates[0], template_list.index(relevant_templates[0]), relevant_templates)
    return ("NoMatch", "NoMatch", relevant_templates)

def match_log(log ,template):
    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = safe_search(regex, log)

    if matches == None:
        return False
    else:
        return True #all(len(var.split()) == 1 for var in matches.groups())

def match_template(match_tree, log_tokens):
    results = []
    find_results = find_template(match_tree, log_tokens, results, [], 1)
    relevant_templates = find_results[1]
    if len(results) > 1:
        new_results = []
        for result in results:
            if result[0] is not None and result[1] is not None and result[2] is not None:
                new_results.append(result)
    else:
        new_results = results
    if len(new_results) > 0:
        if len(new_results) > 1:
            new_results.sort(key=lambda x: (-x[1][0], x[1][1]))
        return new_results[0][1][2], new_results[0][1][3], new_results[0][1][4], relevant_templates
    return False, False, '', relevant_templates


def get_all_templates(move_tree):
    result = []
    for key, value in move_tree.items():
        if isinstance(value, tuple):
            result.append(value[2])
        else:
            result = result + get_all_templates(value)
    return result


def find_template(move_tree, log_tokens, result, parameter_list, depth):
    flag = 0 # no futher find
    if len(log_tokens) == 0:
        for key, value in move_tree.items():
            if isinstance(value, tuple):
                result.append((key, value, tuple(parameter_list)))
                flag = 2 # match
        if "<*>" in move_tree:
            parameter_list.append("")
            move_tree = move_tree["<*>"]
            if isinstance(move_tree, tuple):
                result.append(("<*>", None, None))
                flag = 2 # match
            else:
                for key, value in move_tree.items():
                    if isinstance(value, tuple):
                        result.append((key, value, tuple(parameter_list)))
                        flag = 2 # match
        # return (True, [])
    else:
        token = log_tokens[0]

        relevant_templates = []
        if token in move_tree:
            find_result = find_template(move_tree[token], log_tokens[1:], result, parameter_list,depth+1)
            if find_result[0]:
                flag = 2 # match
            elif flag != 2:
                flag = 1 # futher find but no match
                relevant_templates = relevant_templates + find_result[1]
        if "<*>" in move_tree:
            if isinstance(move_tree["<*>"], dict):
                next_keys = move_tree["<*>"].keys()
                next_continue_keys = []
                for nk in next_keys:
                    nv = move_tree["<*>"][nk]
                    if not isinstance(nv, tuple):
                        next_continue_keys.append(nk)
                idx = 0
                # print("len : ", len(log_tokens))
                while idx < len(log_tokens):
                    token = log_tokens[idx]
                    # print("try", token)
                    if token in next_continue_keys:
                        # print("add", "".join(log_tokens[0:idx]))
                        parameter_list.append("".join(log_tokens[0:idx]))
                        # print("End at", idx, parameter_list)
                        find_result = find_template(
                            move_tree["<*>"], log_tokens[idx:], result, parameter_list,depth+1
                        )
                        if find_result[0]:
                            flag = 2 # match
                        elif flag != 2:
                            flag = 1 # futher find but no match
                            relevant_templates = relevant_templates + find_result[1]
                        if parameter_list:
                            parameter_list.pop()
                        next_continue_keys.remove(token)
                    idx += 1
                if idx == len(log_tokens):
                    parameter_list.append("".join(log_tokens[0:idx]))
                    find_result = find_template(
                        move_tree["<*>"], log_tokens[idx + 1 :], result, parameter_list,depth+1
                    )
                    if find_result[0]:
                        flag = 2 # match
                    else:
                        if flag != 2:
                            flag = 1
                        # relevant_templates = relevant_templates + find_result[1]
                    if parameter_list:
                        parameter_list.pop()
    if flag == 2:
        return (True, [])
    if flag == 1:
        return (False, relevant_templates)
    if flag == 0:
        # print(log_tokens, flag)
        if depth >= 2:
            return (False, get_all_templates(move_tree))
        else:
            return (False, [])
