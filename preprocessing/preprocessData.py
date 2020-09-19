
import os
import sys
import json
import copy

from itertools import compress as itertools_compress

from preprocessing.getQuestionTagAndKey import getQuestionTagAndKeyList
from preprocessing.utils import (saveToPickleFile, loadFromPickleFile)
from preprocessing import const

# NOTE 1) Separate statistics and preprocessing for readability

# QUESTION 1) Can we use a different tokenizer instead of tweeter_nlp to get more information?

# Copied from their code
def readJSONLDatafile(data_file):
    data_instances = []
    with open(data_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                data_instances.append(json.loads(line))
    return data_instances

# Direct Copy from their code
def getTokenCharMapping(text, tweet_token_list):
    # NOTE: tweet_token_list is a list of strings where each element is a token
    current_tok = 0
    current_tok_c_pos = 0
    n_toks = len(tweet_token_list)
    tweet_toks_c_mapping = [list()]
    for c_pos, c in enumerate(text):
        if c.isspace():
            # Just ignore
            continue
        if current_tok_c_pos == len(tweet_token_list[current_tok]):
            # Change current tok and reset c_pos
            current_tok += 1
            current_tok_c_pos = 0
            tweet_toks_c_mapping.append(list())

        if c == tweet_token_list[current_tok][current_tok_c_pos]:
            # Add mapping
            tweet_toks_c_mapping[current_tok].append(c_pos)
            current_tok_c_pos += 1
        else:
            # Something wrong. This shouldn't happen
            print("Wrong mapping:")
            print(text)
            print(tweet_token_list)
            print(c_pos, f"{text[c_pos-1]};{c};{text[c_pos+1]}")
            print(current_tok, current_tok_c_pos, f";{tweet_token_list[current_tok][current_tok_c_pos]};")
            sys.exit()

    # Check if reached end
    assert len(tweet_token_list)-1 == current_tok and len(tweet_token_list[current_tok]) == current_tok_c_pos
    return tweet_toks_c_mapping


def getTokenOffsetFromCharOffset(
        candidate_chunk_char_offset_list, token_char_mapping_list):
    # NOTE Have to go through each char to properly count the multiple spaces

    candidate_chunk_token_offset_list = []
    valid_flag_list = []
    
    for char_sidx, char_eidx in candidate_chunk_char_offset_list:

        token_sidx, token_eidx = None, None

        # NOTE I think they are doing something incomplete as the char_eidx could be after
        #      a space, in which case the inclusive char_eidx will never be in the mapping
        char_eidx_inclusive = char_eidx - 1

        valid_flag = False
        for token_idx, token_char_mapping in enumerate(token_char_mapping_list):
            if char_sidx in token_char_mapping:
                token_sidx = token_idx

            # open right end
            if char_eidx_inclusive in token_char_mapping:
                token_eidx = token_idx + 1 

            # TODO Need better processing here
            # if char_eidx in token_char_mapping:
            #         token_

            if char_eidx in token_char_mapping:
                token_eidx = token_idx + 1 

            if token_sidx != None and token_eidx != None:
                valid_flag = True
                break

        if valid_flag:
            # TODO Port the logging functions here
            pass
        
        candidate_chunk_token_offset_list.append((token_sidx, token_eidx))
        valid_flag_list.append(valid_flag)

    return candidate_chunk_token_offset_list, valid_flag_list


def getAnnotationInToken(
        annotation, tweet_token_list, chunk_char_offset_to_token_offset_mapping):
    """Return processed annotation values and show in terms of tokens
    Input:
        annotation [dict]: from data_item['annoation'], whose value can be a list of offsets sub-list,
                           a str "NO_CONSENSUS" or a list of ['Not Specified']
    """
    annotation_in_token_for_q_key_dict = {}
    for k, v in annotation.items():
        if v == "NO_CONSENSUS":
            new_v = [const.NOT_SPECIFIED]
        else:
            new_v = []
            for item in v:
                if type(item) == list:
                # if v is a list of offest sub-list
                    token_sidx, token_eidx = chunk_char_offset_to_token_offset_mapping[tuple(item)]
                    new_v.append(' '.join(tweet_token_list[token_sidx:token_eidx]))
                else:
                # other case like ['yes']
                    new_v.append(item)
            
        annotation_in_token_for_q_key_dict[k] = new_v
    
    return annotation_in_token_for_q_key_dict

def replaceURLToken(tweet_token_list):
    return [
        const.URL_TOKEN if e.startswith("http") or 'twitter.com' in e or e.startswith('www.')
        else e for e in tweet_token_list]


def replaceUSERToken(tweet_token_list):
    return ["@USER" if word.startswith("@") else word for word in tweet_token_list]


def extractInstanceListForQuestion(
        q_tag, q_key,
        candidate_chunk_and_token_list, tweet_token_list,
        annotation_in_token_for_q_key_dict):
    """Convert candidate_chunk_token_offset 
    """

    candidate_chunk_set = set()
    instance_list = []
    for candidate_chunk_and_token in candidate_chunk_and_token_list:
        (candidate_chunk_id, candidate_chunk, (token_sidx, token_eidx)) = candidate_chunk_and_token
        
        # QUESTION why?
        if candidate_chunk.lower() == 'coronavirus':
            continue
        
        # NOTE Leave out `chunk_start_text_id` and `chunk_end_text_id`
        # if candidate_chunk == const.AUTHOR_OF_THE_TWEET:
        if candidate_chunk in [const.AUTHOR_OF_THE_TWEET, const.NEAR_AUTHOR_OF_THE_TWEET]:
            # QUESTION Okay but why?
            pass
        else:
            if token_eidx > len(tweet_token_list):
                # incorrect token end index
                continue
            candidate_chunk = ' '.join(tweet_token_list[token_sidx:token_eidx])
    
        if candidate_chunk in candidate_chunk_set:
            # Skip if the same candidate_chunk 
            # TODO Leave out the skip_count stats
            continue
        else:
            candidate_chunk_set.add(candidate_chunk)


        # DEBUG
        # QUESTION what are those tags?
        assert q_tag not in ['believe', "binary-relation", "binary-symptoms"]
            
        if q_tag in ["relation", "gender_male","gender_female",
                     "symptoms", "opinion"]:
            # NOTE
            # relation - event 1 to 4
            # gender   - event 1, 2
            # symptoms - event 3 and 4
            # opinion  - event 5

            # yes/no question tag
            special_answer_list = annotation_in_token_for_q_key_dict[q_key]
            assert len(special_answer_list) == 1,\
                f"Yes/No Question {q_tag} has answer {special_answer_list}"
            answer = special_answer_list[0]

            # Unify ansewr
            if answer == "No":
                answer = const.NOT_SPECIFIED 

            # get SPECIAL Q_Label 
            if q_tag in ["gender_male", "gender_female"]:
                gender = "Male" if q_tag == "gender_male" else "Female"
                if gender == answer:
                    q_label_special = getQLabelFromAnswer(answer)
                else:
                    q_label_special = 0
            else:
                q_label_special = getQLabelFromAnswer(answer)

            # get Q_Label (for opinion or name)
            # QUESTION shouldn't this one be called SPECIAL???
            if q_tag == "opinion":
                # NOTE opinion is for event 5, the only one w/o q_tag `name`
                gold_chunk_list = []
                if candidate_chunk == const.AUTHOR_OF_THE_TWEET:
                    q_label = 1
                    gold_chunk_list.append(const.AUTHOR_OF_THE_TWEET)
                else:
                    q_label = 0
            else:
                q_label, gold_chunk_list = getQLabelForKeyFromAnnotation(
                    "part2-name.Response", annotation_in_token_for_q_key_dict, candidate_chunk)

            q_label = q_label & q_label_special
            if q_label == 0:
                gold_chunk_list = []
        else:
            q_label, gold_chunk_list = getQLabelForKeyFromAnnotation(
                q_key, annotation_in_token_for_q_key_dict, candidate_chunk)
        # end of q_label, gold_chunk_list extraction

        # DEBUG
        if q_label:
            assert gold_chunk_list

        # NOTE leave out chunk_start_text_id and chunk_end_text_id
        instance = (candidate_chunk, candidate_chunk_id,
                    ' '.join(
                        tweet_token_list[:token_sidx]\
                        + [const.Q_TOKEN]\
                        + tweet_token_list[token_eidx:]),
                    gold_chunk_list,
                    q_label)

        instance_list.append(instance)

    return instance_list
        

def getQLabelFromAnswer(answer):
    if answer == const.NOT_SPECIFIED:
        return 0
    elif answer == "Yes":
        return 1
    elif answer == "Male":
        return 1
    elif answer == "Female":
        return 1
    elif answer.startswith("no_cure"):
        return 0
    elif answer.startswith("not_effective"):
        return 0
    elif answer.startswith("no_opinion"):
        return 0
    elif answer.startswith("effective"):
        return 1
    else:
        print(f"Unknown answer {answer}")
        exit()


def getQLabelForKeyFromAnnotation(q_key, annotation_in_token_for_q_key_dict, candidate_chunk):
    gold_chunk_list = annotation_in_token_for_q_key_dict[q_key]
    q_label = 0
    if gold_chunk_list:
        if (q_key in ["name", "who_cure", "close_contact", "opinion"]
            and ("I" in gold_chunk_list or "i" in gold_chunk_list)):
	    # if key is "name", "who_cure", and "I" is a gold chunk
            # then add "AUTHOR OF THE TWEET" as a gold chunk
            gold_chunk_list.append(const.AUTHOR_OF_THE_TWEET)

        if any(map(lambda x: x == candidate_chunk, gold_chunk_list)):
            q_label = 1
        # for answer in gold_chunk_list:
        #     if answer == candidate_chunk:
        #         q_label = 1
            
    return q_label, gold_chunk_list

def getValidSubtaskList(
        raw_subtask_list,gold_label_stats_for_q_tag_dict,MIN_POS_SAMPLES_THRESHOLD):
    subtask_list = []
    for subtask in raw_subtask_list:
        # 1 here for positive
        if gold_label_stats_for_q_tag_dict[subtask][1]  >= MIN_POS_SAMPLES_THRESHOLD:
            subtask_list.append(subtask)
    return subtask_list

    
def preprocessDataAndSave(event,whether_new = False):

    # DEBUG
    if whether_new:
        data_file = os.path.join(const.NEW_DATA_FOLDER, f'shared_task-test-{event}-annotated.jsonl')
    else:
        data_file = os.path.join(const.DATA_FOLDER, f'{event}-add_text.jsonl')
    dataset = readJSONLDatafile(data_file)

    # NOTE In their code, things are processed on the fly, which
    #      can actually be processed once, copied from their results actually 
    question_tag_and_key_list = getQuestionTagAndKeyList(event)


    # Init return variables
    instance_list_for_subtask_dict = {q_tag: [] for q_tag, _ in question_tag_and_key_list}


    for data_item_idx, data_item in enumerate(dataset):

        tweet_id = data_item['id']
        annotation = data_item['annotation']
        tweet_text = data_item['text']

        candidate_chunk_char_offset_list = data_item['candidate_chunks_offsets']
        # c is a two-item list of the corresponding indices in text (left close, right open)
        candidate_chunk_in_text_list = [tweet_text[c[0]:c[1]] for c in candidate_chunk_char_offset_list]

        tweet_token_list = [item.rsplit('/', 3)[0] for item in data_item['tags'].split()]

        # QUESTION Skip the tokenized tweet check here, is it really necessary?

        # STEP 1 | Clean Broken Chunk (from wrong tokenization)
        token_char_mapping_list = getTokenCharMapping(tweet_text, tweet_token_list)

        (candidate_chunk_token_offset_list,
        valid_flag_list) = getTokenOffsetFromCharOffset(
            candidate_chunk_char_offset_list, token_char_mapping_list)

        candidate_chunk_in_token_list = [
            ' '.join(tweet_token_list[c[0]:c[1]]) for c in candidate_chunk_token_offset_list]

        # print('tweet_id:', tweet_id)
        # print('text_list:', candidate_chunk_in_text_list)
        # print('token_list:', candidate_chunk_in_token_list)

        # STEP 2 | Reproduce annotation from tokens
        # Filter broken chunks
        if not all(valid_flag_list):
            candidate_chunk_char_offset_list = list(itertools_compress(
                candidate_chunk_char_offset_list, valid_flag_list))
            candidate_chunk_token_offset_list = list(itertools_compress(
                candidate_chunk_token_offset_list, valid_flag_list))

            candidate_chunk_in_text_list = list(itertools_compress(
                candidate_chunk_in_text_list, valid_flag_list))
            candidate_chunk_in_token_list = list(itertools_compress(
                candidate_chunk_in_token_list, valid_flag_list))

            # print("     !!! Reduction occured !!!     ")
            # print('text_list:', candidate_chunk_in_text_list)
            # print('token_list:', candidate_chunk_in_token_list)

        # print("")

        # DEBUG in their code, why don't they use tuple in creating mapping?
        assert all(map(lambda x: len(x) == 2, candidate_chunk_char_offset_list))
        assert all(map(lambda x: len(x) == 2, candidate_chunk_token_offset_list))

        # Convert annotation in terms of token offset
        chunk_char_offset_to_token_offset_mapping = {
            tuple(char_offset): token_offset for char_offset, token_offset in zip(
                    candidate_chunk_char_offset_list, candidate_chunk_token_offset_list)}
        annotation_in_token_for_q_key_dict = getAnnotationInToken(
            annotation, tweet_token_list, chunk_char_offset_to_token_offset_mapping)


        # DEBUG
        # tweet_token_list_ori = copy.copy(tweet_token_list)

        # QUESTION Why use such form?  Name, content, indices?
        # QUESTION Why use original text?
        # NOTE: used in step 4 but the content need to be built from the original text
        candidate_chunk_and_token_list = [
            (f"{c[0]}_{c[1]}",
            ' '.join(tweet_token_list[c[0]:c[1]]),
            c) for c in candidate_chunk_token_offset_list]

        # STEP 3 | Token Replacement
        # URL Replacement
        tweet_token_list = replaceURLToken(tweet_token_list)

        # USER Repalcement
        # QUESTION Should we do it here? Is there a reason why they put it to the Collator?
        tweet_token_list = replaceUSERToken(tweet_token_list)
        # ' '.join(["@USER" if word.startswith("@") else word for word in tokenized_tweet.split()])

        # TODO Placeholder for further preprocessing

        # STEP 4 | Prepare Golden Tokens

        for q_tag, q_key in question_tag_and_key_list:

            # Special Question
            if q_tag in ["name", "close_contact", "who_cure", "opinion"]:
                candidate_chunk_and_token_list.append(
                    ("author_chunk", const.AUTHOR_OF_THE_TWEET, [0,0]))

            if q_tag in ["where", "recent_travel"]:
                candidate_chunk_and_token_list.append(
                    # QUESTION suspect the original copy is wrong, should be NEAR
                    ("near_author_chunk", const.NEAR_AUTHOR_OF_THE_TWEET, [0,0]))
                    # ["near_author_chunk", const.AUTHOR_OF_THE_TWEET, [0,0]])

            raw_instance_list = extractInstanceListForQuestion(
                q_tag, q_key, candidate_chunk_and_token_list, tweet_token_list,
                annotation_in_token_for_q_key_dict)

            tweet_text_tokenized = ' '.join(tweet_token_list)
            instance_list_for_subtask_dict[q_tag] += [
                (tweet_id, tweet_text, tweet_text_tokenized, *raw_instance)
                for raw_instance in raw_instance_list]

    # End of extracting instances

    # Section II
    # Analyze all instances, separated from section I for readability

    gold_label_stats_for_q_tag_dict = {}
    for q_tag, _ in question_tag_and_key_list:
        gold_label_stats_for_q_tag_dict[q_tag] = [0, 0]

        for instance in instance_list_for_subtask_dict[q_tag]:
            q_label = instance[-1]

            # DEBUG
            assert q_label in [0, 1]

            gold_label_stats_for_q_tag_dict[q_tag][q_label] += 1


    # After Preprocessing
    if whether_new:
        MIN_POS_SAMPLES_THRESHOLD = 0
    else:
        MIN_POS_SAMPLES_THRESHOLD = 10

    raw_subtask_list = list(instance_list_for_subtask_dict.keys())
    subtask_list = getValidSubtaskList(
        raw_subtask_list, gold_label_stats_for_q_tag_dict,MIN_POS_SAMPLES_THRESHOLD)

    # Collect instances according to the same text
    # TODO Think about rewrtting this part, why use a triple nested dict?
    instance_to_subtask_label_dict_for_text = {}
    for subtask in subtask_list:
        for instance in instance_list_for_subtask_dict[subtask]:
            instance_new = instance[:-2]
            gold_chunk, gold_label = instance[-2:]

            tweet_text = instance_new[0]

            instance_to_subtask_label_dict_for_text.setdefault(tweet_text, {})
            instance_to_subtask_label_dict_for_text[tweet_text].setdefault(instance_new, {})
            instance_to_subtask_label_dict_for_text[tweet_text][instance_new][subtask]\
                = (gold_chunk, gold_label)

    # Check if there is missing subtask label
    # TODO add more documentation on instance here
    for tweet_text, instance_to_subtask_label_dict in instance_to_subtask_label_dict_for_text.items():
        for instance, subtask_label_dict in instance_to_subtask_label_dict.items():

            for subtask in subtask_list:
                if subtask not in subtask_label_dict:
                    subtask_label_dict[subtask] = ([], 0)
            # some candiadate such as const.AUTHOR_OF_THE_TWEET are not avaliable to every subtask
            # if len(subtask_label_dict) != len(subtask_list):
            #     print(f"{instance}: {subtask_label_dict}")
            #     raise ValueError("Wrong length of instance_to_subtask_label_dict")
            assert len(subtask_label_dict) == len(subtask_list)
            instance_to_subtask_label_dict_for_text[tweet_text][instance] = subtask_label_dict

    # Merging all instances
    all_instance_list = []
    for instance_to_subtask_label_dict in instance_to_subtask_label_dict_for_text.values():
        for instance, subtask_label_dict in instance_to_subtask_label_dict.items():
            all_instance_list.append((*instance, subtask_label_dict))

    # NOTE
    # END of getting all instances, beyond this point the behaviour will depend on
    # hyperparameters like train_ratio, batch size
    # --------------------------------------------------------
    # Split into train_dev_test set

    # TODO This could be merged with previous steps, keep it in this form to keep consistent
    #      with their code, just for now (maybe).
    # Replace with BERT Tokens
    raw_input_text_and_label_list = []
    #print(all_instance_list[0])
    for instance in all_instance_list:
        # DEBUG
        assert len(instance) == 7

        tweet_text = instance[1]
        token_text = instance[2]
        tweet_id = instance[0]
        chunk = instance[3]
        tweet_text_tokenized_with_masked_chunk = instance[5]

        if chunk in [const.AUTHOR_OF_THE_TWEET, const.NEAR_AUTHOR_OF_THE_TWEET]:
            input_text = tweet_text_tokenized_with_masked_chunk.replace(const.Q_TOKEN, "<E> </E>")
        else:
            input_text = tweet_text_tokenized_with_masked_chunk.replace(
                const.Q_TOKEN, f"<E> {chunk} </E>")

        subtask_label_dict = instance[-1]

        # Tweet_text is used in spliting all list into train_dev_test sets

        raw_input_text_and_label_list.append((tweet_text, input_text, subtask_label_dict, tweet_id, token_text))
    # filename = os.path.join(const.DATA_FOLDER, f'{event}-preprocessed-data.pkl')

    if whether_new:
        filename = os.path.join(const.NEW_DATA_FOLDER, f'{event}-preprocessed-data.pkl')
    else:
        filename = os.path.join(const.DATA_FOLDER, f'{event}-preprocessed-data.pkl')
    saveToPickleFile(
        (subtask_list, raw_input_text_and_label_list), filename)

    return filename


# Split into train_dev_test datasets
def splitDatasetIntoTrainDevTest(raw_input_text_and_label_list, train_ratio=0.6, dev_ratio=0.15):

    # NOTE Doing so will result in different split each time, as set is unordered 
    # uniq_tweet_text_list = list(
    #     set(tweet_text for tweet_text, _, _ in raw_input_text_and_label_list))

    # NOTE Following their method, which will return a fixed split
    uniq_tweet_text_list = []
    uniq_tweet_text_set = set()
    for tweet_text, input_text, subtask_label_dict, tweet_id, token_text in raw_input_text_and_label_list:
        if tweet_text not in uniq_tweet_text_set:
            uniq_tweet_text_set.add(tweet_text)
            uniq_tweet_text_list.append(tweet_text)

    train_size = int(len(uniq_tweet_text_list) * train_ratio)
    dev_size   = int(len(uniq_tweet_text_list) * dev_ratio)

    train_tweet_text_set = set(uniq_tweet_text_list[:train_size])
    dev_tweet_text_set = set(uniq_tweet_text_list[train_size:train_size+dev_size])
    test_tweet_text_set = set(uniq_tweet_text_list[train_size+dev_size:])

    train_input_text_and_label_list = []
    dev_input_text_and_label_list   = []
    test_input_text_and_label_list  = []
    for tweet_text, input_text, subtask_label_dict, tweet_id, token_text in raw_input_text_and_label_list:
        item = (input_text, subtask_label_dict, tweet_id, token_text)
        if tweet_text in train_tweet_text_set:
            train_input_text_and_label_list.append(item)
        elif tweet_text in dev_tweet_text_set: 
            dev_input_text_and_label_list.append(item)
        elif tweet_text in test_tweet_text_set:
            test_input_text_and_label_list.append(item)
        else:
            raise ValueError(f"Tweet text: {tweet_text} is not correctly categorized")

            
    return (train_input_text_and_label_list,
            dev_input_text_and_label_list,
            test_input_text_and_label_list)
