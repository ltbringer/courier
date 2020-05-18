"""
NOTE: This is kept in __init__ to maintain compatibility.
TODO: Group functions according to meaning in different files
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Union

import dateutil.parser
import oyaml
import pydash as py_
import regex as re
import sentry_sdk
from num2words import num2words

from plute import IS_PROD, SENTRY_DSN
from plute.filters import is_int
from plute.types import Entity, Range

WORDS2NUM_CACHE = {
    re.compile(r"\b" + num2words(n, lang="en").replace("-", " ") + r"\b"): n for n in range(100)
}

WORDS2ORDINAL_CACHE = {
    re.compile(r"\b" + num2words(n, lang="en", ordinal=True).replace("-", " ") + r"\b"): num2words(n, lang="en", to="ordinal_num") for n in range(31)
}

REGEX_MEN = "guys?|mens?|man|boys?|males?|husbands?|gents?"
REGEX_WOMEN = "wife|wives|womens?|woman|girls?|females?|ladies|lady"
REGEX_ADULTS = "peoples?|persons?|adults?|elders?"
REGEX_KIDS = "(small )?(kids?|childrens?|childs?|baby|babies|infants?)"
REGEX_OTHERS = "members?|friends?|pax|guests?"


def first_non_empty(items):
    """
    Return first non empty item from the list. If nothing is found, we just
    pick the first item. If there is no item, we return the whole form
    (defaulting to []).
    """

    if items:
        for it in items:
            if it:
                return it

        return items[0]
    else:
        return items


def map_values(d: Dict, fn) -> Dict:
    """
    Run `fn` over values of `d` and return new dictionary.
    """

    return {k: fn(v) for k, v in d.items()}


def replace_date(iso_string: str, date) -> str:
    """
    Apply date in the iso_string and return new iso string.
    """

    parsed = dateutil.parser.parse(iso_string)
    return parsed.replace(year=date.year, month=date.month, day=date.day).isoformat()


def maximal(poset: List, comparable_pred, gt_pred) -> List:
    """
    Return maximal of a partially ordered set. `comparable_pred` tells whether
    we can compare to items of the set. `gt_pred` says whether argument 1 is
    strictly greater than the second.
    """

    indices_to_skip = set()

    # TODO: Number of comparisons can be reduced here
    for i in range(len(poset)):
        for j in range(i, len(poset)):
            if i != j:
                if comparable_pred(poset[i], poset[j]):
                    # one of =, > or < is true
                    if gt_pred(poset[i], poset[j]):
                        indices_to_skip.add(j)
                    elif gt_pred(poset[j], poset[i]):
                        indices_to_skip.add(i)

    output = []
    for i, it in enumerate(poset):
        if i not in indices_to_skip:
            output.append(it)

    return output


def read_oyaml(f, default=None):
    """
    Read ordered yaml with default.
    """

    if default is None:
        default = {}

    if os.path.exists(f):
        with open(f) as fp:
            return oyaml.load(fp) or default
    else:
        return default


def range_replace(text: str, ranges: List[Range], replacements: Optional[Union[List[str], str]] = None) -> str:
    """
    Replace the ranges given in text.

    TODO: This is NOT a complete implementation
          The correct implementation will check if the ranges are overlapping
          and will throw an error. In case you are using this function, make sure
          to only work with non-overlapping ranges.
    """

    if replacements is None:
        replacements = ""

    if isinstance(replacements, str):
        replacements = [replacements] * len(ranges)

    offset = 0
    for (start, end), rep in zip(sorted(ranges, key=lambda it: it[0]), replacements):
        text = text[:(start - offset)] + rep + text[(end - offset):]
        offset += (end - start) - len(rep)

    return text


def make_optional_pattern(items: List[str], word_break=True):
    if word_break:
        return re.compile(r"\b(" + r"|".join(items) + r")\b", re.I | re.UNICODE)
    else:
        return re.compile(r"((" + r")|(".join(items) + r"))", re.I | re.UNICODE)


def replace_subexpr(pattern: str, expansions: Dict[str, List[str]], prefix="EX") -> str:
    """
    Replace subexpr in pattern using the expansions and return the new pattern.
    """
    ranges = []
    replacements = []
    match_string = "\\{" + prefix + ":(?P<subexpr>.+?)}"
    for m in re.finditer(r"{}".format(match_string), pattern, flags=re.I | re.UNICODE):
        term = m.group("subexpr")
        span = m.span("subexpr")
        ranges.append((span[0] - 4, span[1] + 1))

        if term not in expansions:
            raise ValueError(f"Term {term} not found in {prefix}")
        else:
            replacements.append(make_optional_pattern(expansions[term], word_break=False).pattern)

    return range_replace(pattern, ranges, replacements)


@contextmanager
def temp_wd(wd: str):
    """
    Run the block in given working dir and restore on exit.
    """
    old_wd = os.getcwd()
    try:
        os.chdir(wd)
        yield
    finally:
        os.chdir(old_wd)


def is_list_empty(in_list):
    if isinstance(in_list, list):
        return all(map(is_list_empty, in_list))

    return False


def is_list_subset(subset_list, superset_list):
    return all(x in superset_list for x in subset_list)


def tag_entity(ent: Entity, tag: str) -> Entity:
    """
    Tag entity with a transformation tag. This might be useful for logging
    what transformations that entity went through.
    """

    return Entity({
        **ent,
        "transformers": ent.get("transformers", []) + [tag]
    })


def ent_to_range(ent: Entity) -> Range:
    """
    TODO: we should make our range functions work on the default entity
          type instead of this tuple
    """
    return ent["range"]["start"], ent["range"]["end"]


def all_repeated_digits(num: int) -> bool:
    """
    Tell if the number is made of all repeated digits.
    """

    s_num = str(num)

    if len(s_num) > 1:
        return all([ch == s_num[0] for ch in s_num[1:]])
    else:
        return False


def is_suppressing(a: Range, b: Range) -> bool:
    """
    Tells if a is suppressing b. Return false in all the other cases.
    """

    if a == b:
        return False
    else:
        return a[1] >= b[1] and a[0] <= b[0]


def range_overlap(a: Range, b: Range) -> bool:
    """
    Tells if a and b have any overlap.
    """

    if is_suppressing(a, b) or is_suppressing(b, a):
        # Pure suppression is a case of overlap
        return True
    else:
        return max(a[0], b[0]) < min(a[1], b[1])


def superior_ranges(ranges: List[Range]) -> List[Range]:
    """
    Return all the ranges which can't be suppressed by others.
    """

    output = []

    for r in ranges:
        if any([ot_r for ot_r in ranges if r != ot_r and is_suppressing(ot_r, r)]):
            continue
        else:
            output.append(r)

    return output


def non_overlapping_ranges(ranges: List[Range]) -> List[Range]:
    """
    Return ranges which are not overlapping any other range.
    """

    bad_ranges = set()

    for i in range(len(ranges)):
        for j in range(len(ranges)):
            if i != j and range_overlap(ranges[i], ranges[j]):
                bad_ranges.add(ranges[i])

    return [r for r in ranges if r not in bad_ranges]


def digtoto(text: str) -> str:
    """
    Convert `x-y` to `x to y`.

    NOTE: Should filter only when x and y are digits.
    """

    rule = (r"\b(?P<fromd>\d+)( )?-( )?(?P<tod>\d+)\b", "{fromd} to {tod}")
    match = re.search(rule[0], text, flags=re.I | re.U)
    if match:
        replacement = rule[1].format(fromd=match.group("fromd"), tod=match.group("tod"))
        x, y = match.span()
        return text[:x] + replacement + text[y:]

    return text


def for_at_fix(text: str) -> str:
    r"""
    Convert patterns:
    - `for <num> at <num>` to `for <num> people at <num>`
    - `at <num> for <num>` to `at <num> for <num> people`

    Since this is run after running words_to_num, we only need to consider cases
    where <num> takes \d+ values.
    """

    patterns = [
        (r"for (?P<people>\d+) at (?P<num>\d+)", "for {people} people at {num}"),
        (r"at (?P<num>\d+) for (?P<people>\d+)", "at {num} for {people} people")
    ]

    for pattern, template in patterns:
        match = re.search(pattern, text, flags=re.I | re.U)
        if match:
            replacement = template.format(people=match.group("people"), num=match.group("num"))
            x, y = match.span()
            return text[:x] + replacement + text[y:]

    return text


def words_to_ordinal(text: str) -> str:
    """
    Convert words representing ordinal to ordinal. Example: twenty fourth to 24th
    """

    for pattern, num in sorted(WORDS2ORDINAL_CACHE.items(), key=lambda it: int(it[1][:-2]), reverse=True):
        text = pattern.sub(str(num), text)

    return text


def words_to_num(text: str) -> str:
    """
    Convert words representing number to number. Example: thirty four to 34
    """

    # NOTE: Since a few words have numbers in them, we canonicalize them first
    rules = [(r"\bevery one\b", "everyone"), (r"\bno one\b", "none")]

    for rule in rules:
        text = re.sub(rule[0], rule[1], text, flags=re.I | re.U)

    for pattern, num in sorted(WORDS2NUM_CACHE.items(), key=lambda it: it[1], reverse=True):
        text = pattern.sub(str(num), text)

    return text


def plus_fix(text: str) -> str:
    """
    Convert "a+b" or "a plus b" to the numerical value (a+b).
    """
    rule = r"(\b)(?P<intA>\d+)(\s*)([\w]*)(\s*)(\+|\bplus\b)(\s*)(?P<intB>\d+)(\s*)(\4)(\b)"
    text = re.sub(rule, lambda match: "{0} {1}".format(int(match.group("intA")) + int(match.group("intB")), match.group(4)), text)
    return text


def percent_to_person(text: str) -> str:
    """
    Phonetically person and percent are similar. Therefore this is a bbq specific regex matcher.
    """
    rule = r"\b(?P<person>\d+)((\s)?\%|\s\bpercent\b)"
    text = re.sub(rule, lambda match: "{0} person".format(int(match.group("person"))), text)
    return text


def for_to_fix(text: str) -> str:
    """
    Specific for BBQN:
    For adults -> 4 adults
    to adults -> 2 adults
    Same with kids
    """
    rule = r"\b(?P<num>for|to) (?P<set>" + REGEX_ADULTS + r"|" + REGEX_OTHERS + r"|" + REGEX_KIDS + r"|" + \
           REGEX_WOMEN + r"|" + REGEX_MEN + r")\b"

    text = re.sub(rule, lambda match: "{0} {1}".format(4 if str(match.group("num")) == "for" else 2,
                                                       str(match.group("set"))), text)
    return text


def num_adults_and_num_kids_fix(text: str) -> str:
    """
    Specific for BBQN
    ASR doesn't give proper results but through heuristics we can say the following transformation will help us:
    1. 5( adds)? and 1 kid -> 5 adults and 1 kid :: 5 and 1 kid -> 5 adults and 1 kid
    2. 4 (adds|~to) 1 kid -> 4 adults 1 kid
    3. 3 adults and 1 (xyz|~years?) -> 3 adults and 1 kid

    ~ means negative regex (shouldn't be present)
    """

    ignore_regex_to = "to"
    ignore_regex_years = "years?"

    rules = [(r"\b(?P<start>\d+)( )(\w+ )?(?P<end>and \d+ (" + REGEX_KIDS + r"))\b", "{0} adults {1}"),
             (r"\b(?P<start>\d+)( )((?!" + ignore_regex_to + r"\b)\w+ )(?P<end>\d+ (" + REGEX_KIDS + r"))\b", "{0} adults {1}"),
             (r"\b(?P<start>\d+ (" + REGEX_ADULTS + r"))( )(?P<end>and \d+)( )((?!" + ignore_regex_years + r"\b)\w+)\b", "{0} {1} kids")]
    for rule, txt_format in rules:
        text = re.sub(rule, lambda match: txt_format.format(str(match.group("start")), str(match.group("end"))), text)

    return text


def repeating_same_number_fix(text: str) -> str:
    """
    People sometimes tend to repeat the numbers. Therefore we are removing the number the same number is repeated more
    than certain number of times
    """
    upper_limit_repetation = 3

    def _pop_from_list(list, count):
        for _ in range(count):
            list.pop()
        return list

    count_continuous_digits = 0
    int_value = None
    output_list: List[str] = []
    for word in text.split():
        if is_int(word):
            if int_value is None:
                """
                Previously there was no interger
                """
                int_value = int(word)
                count_continuous_digits = 1
            elif int_value == int(word):
                """
                Same integer found
                """
                count_continuous_digits += 1
            else:
                """
                Different integer found
                """
                if count_continuous_digits >= upper_limit_repetation:
                    _pop_from_list(output_list, count_continuous_digits - 1)

                int_value = int(word)
                count_continuous_digits = 1
        else:
            if count_continuous_digits >= upper_limit_repetation:
                _pop_from_list(output_list, count_continuous_digits - 1)

            count_continuous_digits = 0
            int_value = None

        output_list.append(word)

    if count_continuous_digits >= upper_limit_repetation:
        _pop_from_list(output_list, count_continuous_digits - 1)

    return " ".join(output_list)


def consecutive_number_fix(text: str) -> str:
    """
    People sometimes tend to count the numbers. Therefore we are removing the number the number is incremented more than
    a certain number of times
    """
    upper_limit_repetation = 3

    def _pop_from_list_leaving_last(list, count):
        last_element = list.pop()
        for _ in range(count):
            list.pop()
        list.append(last_element)
        return list

    count_continuous_digits = 0
    int_value = None
    output_list: List[str] = []
    for word in text.split():
        if is_int(word):
            if int_value is None:
                """
                Previously there was no interger
                """
                int_value = int(word)
                count_continuous_digits = 1
            elif int(word) == (int_value + 1):
                """
                Next integer found
                """
                int_value += 1
                count_continuous_digits += 1
            else:
                """
                Different integer found
                """
                if count_continuous_digits >= upper_limit_repetation:
                    _pop_from_list_leaving_last(output_list, count_continuous_digits - 1)

                int_value = int(word)
                count_continuous_digits = 1
        else:
            if count_continuous_digits >= upper_limit_repetation:
                _pop_from_list_leaving_last(output_list, count_continuous_digits - 1)

            count_continuous_digits = 0
            int_value = None

        output_list.append(word)

    if count_continuous_digits >= upper_limit_repetation:
        _pop_from_list_leaving_last(output_list, count_continuous_digits - 1)

    return " ".join(output_list)


def convert_to_12hour_clock(value):
    """
    Change 24 hour clock to 12 hour clock.
    """

    twelve_hour_value = value.strftime("%I:%M")
    datetime_value = datetime.strptime(twelve_hour_value, "%I:%M")
    return datetime_value


def remove_address_specifics(text: str) -> str:
    """
    Remove address specifics like, 12th cross road.

    This is to ensure no entity is returned by Rhea.
    """

    patterns = [
        re.compile(pattern, re.I) for pattern in [
            r"(sector|block|floor|main|stage|phase) (\d+)\b", r"(number|no\.?) (\d+)",
            r"(\d+)(?:st|nd|rd|th)? (sector|floor|block|feet road|cross road|road|main|stage|phase)",
            r"(\d+) mall", r"unity (\d+)", r"city cent(er|re|ral) (\d+)", r"(\d+)( )?mg"
        ]
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            text = text.replace(match.group(), "#" * len(match.group()))
    # remove whitespaces
    text = " ".join(text.split())
    return text


def capture_exception(e, raise_exception=True):
    """
    Capture the exception on Sentry explicitly.

    :param e: Exception to be caught
    :param raise_exception: Set to true if the exception is to be raised
    """

    if IS_PROD and SENTRY_DSN:
        sentry_sdk.capture_exception(e)
    if raise_exception:
        raise e


def string_range(list_of_alternatives: List[str], delimiter="$$$") -> List[Range]:
    """
    Return ranges of string separated via the `delimiter`.

    Example: bangalore$$$bangalore at 6pm for 7 people$$$bangalore at 7pm
    returns [(0, 9), (12, 41), (44, 60)]
    """

    str_ranges = [(0, len(list_of_alternatives[0]))]

    for alt in list_of_alternatives[1:]:
        offset = len(delimiter) + str_ranges[-1][1]
        str_ranges.append((offset, offset + len(alt)))

    return str_ranges


def filter_entities_in_range(string_ranges: List[Range], ents: List[Entity]) -> List[List[Entity]]:
    """
    Take ranges of strings (from the concatenated one) and entities, and return
    entities corresponding to each string.
    """

    ranges = []
    for st, en in string_ranges:
        string_ent = []

        for ent in ents:
            if ent["range"]["start"] >= st and ent["range"]["end"] <= en:
                string_ent.append(ent)

        ranges.append(string_ent)
    return ranges


def group_delimited_entities(input_text: str, ents: List[Entity], delimiter="$$$") -> List[List[Entity]]:
    """
    Group entities according to the range offsets and the delimiter.
    """

    list_of_alternatives = input_text.split(delimiter)
    string_ranges = string_range(list_of_alternatives)
    filtered_entities = filter_entities_in_range(string_ranges, ents)

    grouped = []
    for idx, (str_range, entities) in enumerate(zip(string_ranges, filtered_entities)):
        grouped.append([
            Entity({
                **ent,
                "range": {
                    "start": ent["range"]["start"] - str_range[0],
                    "end": ent["range"]["end"] - str_range[0]
                },
                "alternative_index": idx
            })
            for ent in entities
        ])

    return grouped


def get_expected_ack_slot_types(intents_info, context=None) -> List:
    """
    :return: all the slot types (i.e. ["location", "time", "people"] etc.) based on expected and ack slots
    """

    return py_.uniq(get_slot_types(intents_info, context, key="expected_slots") + get_slot_types(intents_info, context, key="ack_slots"))


def get_slot_types(intents_info, context=None, key="expected_slots") -> List:
    """
    :return: all the slot types (i.e. ["location", "time", "people"] etc.) based on key
    Valid keys: expected_slots, ack_slots
    """
    slot_types: List = []
    if context.get(key):
        for exp_slot in context.get(key):
            for intent in intents_info:
                for slot in intent["slots"]:
                    if slot.get("name") == exp_slot:
                        slot_types.extend(slot.get("type", []))
                        break
    return py_.uniq(slot_types)


def nine_to_no_fix(alts: List[str]) -> List[str]:
    """
    Google ASR specific
    GASR gives 9 when the person says no or not. Therefore if both of them
    are present at the start of the sentence, convert all 9 to no
    """
    count_9 = 0
    count_no = 0

    if len(alts) < 2:
        return alts

    for txt in alts:
        if txt:
            if txt.split()[0] == "9":
                count_9 += 1
            elif txt.split()[0] in ["no", "not"]:
                count_no += 1

    if count_no and count_9:
        output_texts = []
        for txt in alts:
            if txt and txt.split()[0] == "9":
                output_texts.append("no " + " ".join(txt.split()[1:]))
            else:
                output_texts.append(txt)
        return output_texts
    else:
        return alts


def word_mapping_fix(text: str):
    """
    Replace certain patterns in text.

    This is google ASR specifc. GASR gives certain words that alone doesn't
    mean anything but with certain context means a time.

    TODO: This is bad way to implement and structure pattern rules. This idea
          is inherent to the system and so should be generalized properly.
    """
    pattern_rules = [(r"\bstate p(.| )?m(.)?\b", "8 pm")]
    for rule in pattern_rules:
        text = re.sub(rule[0], rule[1], text, flags=re.I | re.U)

    string_rules = [("atm", "8 pm")]
    for rule in string_rules:
        text = rule[1] if rule[0] == text else text

    return text


def ek_8_fix(alts: List[str]) -> List[str]:
    """
    Replace ek, 8 patterns in text.

    This is google ASR specifc. Google gets confused between 1 and 8.

    Therefore if alternatives only contain 8 and 1, we change everything to 8
    pm.

    TODO: Another really structurally bad piece of logic.
    """
    count_ek = 0
    count_8 = 0
    count_other = 0

    rule = r"\b(?P<num>\d+) p(.| )?m(.)?"

    for text in alts:
        match = re.search(rule, text, flags=re.I | re.U)
        if match:
            if match.group("num") == "1":
                count_ek += 1
            elif match.group("num") == "8":
                count_8 += 1
            else:
                count_other += 1

    if count_8 and count_ek and not count_other:
        output_alts = []
        substitute = "8 pm"
        for text in alts:
            text = re.sub(rule, substitute, text, flags=re.I | re.U)
            output_alts.append(text)
        return output_alts
    return alts


def make_ngrams(n: int, tokens: List[str]):
    """
    Create N-grams for given set of tokens.

    example:
    --------
    > make_ngrams(2, ["have", "a", "good", "day"])
    < [("have", "a), ("a", "good"), ("good", "day")]

    """
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def has_more_than(limit):
    def f(items):
        return len(items) > limit
    return f
