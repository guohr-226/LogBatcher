import re
from collections import Counter

def normalize_common_identifiers(template):
    rules = [
        (r'\b[A-Za-z][A-Za-z0-9+.-]*://\S+', '<*>'),
        (r'\b(?:\d{1,3}\.){3}\d{1,3}(?::(?:\d+|<\*>))?\b', '<*>'),
        (r'\b(?:[A-Za-z0-9-]+\.){2,}[a-z]{2,4}(?::(?:\d+|<\*>))?\b', '<*>'),
        (r'(\b[A-Za-z_][\w.-]*)=(?=(?:\s|$|[,;\]\)]))', r'\1=<*>'),
        (r'\b(?:\d+(?:\.\d+)?|<\*>) bytes \((?:\d+(?:\.\d+)?|<\*>)(?:\s*[KMGT]?B)?\)', '<*> bytes (<*>)'),
        (r'\b(?:\d+(?:\.\d+)?|<\*>)\s+([KMGT]?B)\b', r'<*> \1'),
        (r'\blifetime <\*> sec\b', 'lifetime <*>'),
        (r'\bblk_[-+]?(?:\d+|<\*>)\b', '<*>'),
        (r'\bblock_[-+]?(?:\d+|<\*>)\b', '<*>'),
        (r'\battempt_\d+_\d+_[mr]_\d+_\d+\b', '<*>'),
        (r'\battempt_<\*>_<\*>_[mr]_<\*>_<\*>\b', '<*>'),
        (r'\btask_\d+_\d+_[mr]_\d+\b', '<*>'),
        (r'\btask_<\*>_<\*>_[mr]_<\*>\b', '<*>'),
        (r'\bDFSClient_NONMAPREDUCE_[-+]?\d+_\d+\b', '<*>'),
        (r'\bDFSClient_NONMAPREDUCE_-<\*>\b', '<*>'),
        (r'\brdd_\d+_\d+\b', '<*>'),
        (r'\brdd_<\*>\b', '<*>'),
        (r'\bbroadcast_\d+(?:_piece\d+)?\b', '<*>'),
        (r'\bbroadcast_<\*>(?:_piece<\*>)?\b', '<*>'),
        (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<*>'),
        (r'\b0[xX][0-9a-fA-F]+\b', '<*>'),
        (r'<\*>(?:-[0-9a-fA-F<*>]+){2,}', '<*>'),
        (r'::ffff:<\*>', '<*>'),
        (r'\bmsra-sa-<\*>\b', '<*>'),
        (r'\bat (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) <\*> <\*> <\*>', 'at <*>'),
        (r'(connection from <\*> )\([^)]+\)( at )', r'\1(<*>)\2'),
        (r'tty=\S+', 'tty=<*>'),
        (r'\bOut of Memory: Killed process <\*> \([^)]+\)\.', 'Out of Memory: Killed process <*> (<*>).'),
        (r'\bFailed password for invalid user \S+ from ', 'Failed password for invalid user <*> from '),
        (r'\bInvalid user \S+ from ', 'Invalid user <*> from '),
        (r'\bcheck pass; user \S+\b', 'check pass; user <*>'),
    ]
    for pattern, replacement in rules:
        template = re.sub(pattern, replacement, template)
    return template

def normalize_template_text(template):
    if template is None:
        return ''

    template = str(template).replace('\n', ' ').strip()
    if len(template) >= 2 and template[0] == '`' and template[-1] == '`':
        template = template[1:-1].strip()

    template = re.sub(r'\{\{.*?\}\}', '<*>', template)
    template = re.sub(r'\$\{.*?\}', '<*>', template)
    template = normalize_common_identifiers(template)
    template = correct_single_template(template)
    template = normalize_common_identifiers(template)
    template = correct_single_template(template)
    if template.replace('<*>', '').replace(' ','') == '':
        template = ''

    return template

def post_process(response):

    response = response.replace('\n', '')
    first_backtick_index = response.find('`')
    last_backtick_index = response.rfind('`')
    if first_backtick_index == -1 or last_backtick_index == -1 or first_backtick_index == last_backtick_index:
        tmps = []
    else:
        tmps = response[first_backtick_index: last_backtick_index + 1].split('`')
    for tmp in tmps:
        if tmp.replace(' ','').replace('<*>','') == '':
            tmps.remove(tmp)
    tmp = ''
    if len(tmps) == 1:
        tmp = tmps[0]
    if len(tmps) > 1:
        tmp = max(tmps, key=len)

    return normalize_template_text(tmp)

def exclude_digits(string):
    '''
    exclude the digits-domain words from partial constant
    '''
    if not string:
        return False
    pattern = r'\d'
    digits = re.findall(pattern, string)
    if len(digits) == 0 or string[0].isalpha() or any(c.isupper() for c in string):
        return False
    elif len(digits) >= 4:
        return True
    else:
        return len(digits) / len(string) > 0.3

def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root'} # 'null', 'root', 'admin'
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'', r'\+',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\@', r'\#', r'\$', r'\%', r'\&', r'\/'
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)
    # default_strings = {}

    # apply DS
    # Note: this is not necessary while postprorcessing
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)
    template = normalize_common_identifiers(template)

    # apply PS
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:
        # print(p_token)
        # if re.match(r'^(\/[^\/]+)+$', p_token) or re.match(r'^([a-zA-Z0-9-]+\.){2,}[a-zA-Z]+$', p_token):
        if re.match(r'^(\/[^\/]+)+\/?$', p_token) or re.match(r'.*/.*\..*', p_token) or re.match(r'^([a-zA-Z0-9-]+\.){3,}[a-z]+$', p_token):
        # or re.match(r'^([a-z0-9-]+\.){2,}[a-z]+$', p_token)
            p_token = '<*>'
        
        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)
    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply BL, US
        for to_replace in boolean.union(default_strings):
            # if token.lower() == to_replace.lower():
            if token == to_replace:
                token = '<*>'

        # apply DG
        # Note: hexadecimal num also appears a lot in the logs
        # if re.match(r'^\d+$', token) or re.match(r'\b0[xX][0-9a-fA-F]+\b', token):
        #     token = '<*>'
        if exclude_digits(token):
            token = '<*>'

        # apply WV
        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token) or re.match(r'^<\*>.*<\*>$', token):
            token = '<*>'
        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')

    while " '<*>' " in template:
        template = template.replace(" '<*>' ", " <*> ")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    return normalize_common_identifiers(template)


def _template_words(template):
    return str(template or "").split()


def _has_placeholder(token):
    return "<*>" in token


def _is_variable_like_token(token):
    token = str(token or "")
    if _has_placeholder(token):
        return True
    if re.search(r"\d", token):
        return True
    if "/" in token or "\\" in token:
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", token):
        return True
    if re.match(r"^(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?$", token):
        return True
    if re.match(r"^(?:[A-Za-z0-9-]+\.){2,}[a-z]{2,4}(?::\d+)?$", token):
        return True
    return False


def _merge_template_token(left, right):
    if left == right:
        return left
    if _has_placeholder(left) or _has_placeholder(right):
        return "<*>"
    if _is_variable_like_token(left) and _is_variable_like_token(right):
        return "<*>"
    return None


def _generalize_template_pair(left, right):
    left_tokens = _template_words(left)
    right_tokens = _template_words(right)
    if len(left_tokens) != len(right_tokens) or not left_tokens:
        return None

    merged = []
    same_count = 0
    diff_count = 0
    placeholder_diff_count = 0
    for left_token, right_token in zip(left_tokens, right_tokens):
        merged_token = _merge_template_token(left_token, right_token)
        if merged_token is None:
            return None
        if left_token == right_token:
            same_count += 1
        else:
            diff_count += 1
            if _has_placeholder(left_token) or _has_placeholder(right_token):
                placeholder_diff_count += 1
        merged.append(merged_token)

    if diff_count == 0:
        return None
    if placeholder_diff_count == 0:
        return None
    if same_count < max(1, len(left_tokens) // 3):
        return None

    generalized = normalize_template_text(" ".join(merged))
    if generalized.replace("<*>", "").replace(" ", "") == "":
        return None
    return generalized


def merge_similar_templates(outputs, max_passes=2):
    """Merge near-duplicate templates produced by over-specific variables.

    The merge is intentionally conservative: templates must have the same
    whitespace-token shape, share a stable constant backbone, and differ at
    least once where one side already contains <*>.
    """
    current = [normalize_template_text(output) for output in outputs]
    templates_before = len(set(current))

    for _ in range(max_passes):
        counts = Counter(current)
        unique_templates = sorted(counts, key=lambda item: (-counts[item], item))
        mapping = {}

        for i, left in enumerate(unique_templates):
            if left in mapping:
                continue
            for right in unique_templates[i + 1:]:
                if right in mapping:
                    continue
                generalized = _generalize_template_pair(left, right)
                if generalized is None:
                    continue
                mapping[left] = generalized
                mapping[right] = generalized
                break

        if not mapping:
            break
        current = [mapping.get(output, output) for output in current]

    templates_after = len(set(current))
    return current, {
        "templates_before": templates_before,
        "templates_after": templates_after,
        "changed_rows": sum(1 for original, merged in zip(outputs, current) if original != merged),
    }
