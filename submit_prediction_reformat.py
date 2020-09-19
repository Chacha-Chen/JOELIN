#
import os
import jsonlines
import sys

def parseBinaryValue(value):
    if len(value) == 1 and value[0] == 'Not Specified':
        return False
    else:
        return True

def formatData(data):

    pred_anno = data['predicted_annotation']

    data['predicted_annotation'] = {
        f'part2-{key}.Response': val for key, val in pred_anno.items()
        }

    return data

event_list = ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']

def run_reformat(folder):
    print(f"running for {folder}")
    print(f"output to reformatted_submission/{folder}")
    os.makedirs(f'reformatted_submission/{folder}', exist_ok=True)

    for event in event_list:
        filename = f"{folder}/OURTEAM-global-{event}.jsonl"
    
        with jsonlines.open(filename, mode='r') as f:
    
            formatted_data_list = []
            for data in f:
    
                pred_anno = data['predicted_annotation']
    
                if 'name' in pred_anno:
                    tmp_list = []
                    for x in pred_anno['name']:
                        if x in ['i', 'I']:
                            tmp_list.append('AUTHOR_OF_THE_TWEET')
                        else:
                            tmp_list.append(x)
                    pred_anno['name'] = tmp_list
    
                for key in ['where', 'recent_travel']:
                    if key in pred_anno:
                        tmp_list = []
                        for x in pred_anno['name']:
                            if x in ['i', 'I', 'AUTHOR_OF_THE_TWEET']:
                                tmp_list.append('NEAR_AUTHOR_OF_THE_TWEET')
                            else:
                                tmp_list.append(x)
                        pred_anno[key] = tmp_list
    
    
                if 'gender_male' in pred_anno:
    
                    assert 'gender_female' in pred_anno
    
                    gender_male = parseBinaryValue(pred_anno.pop('gender_male'))
                    gender_female = parseBinaryValue(pred_anno.pop('gender_female'))
    
                    if (gender_male, gender_female) == (True, False):
                        gender = 'male'
                    elif (gender_male, gender_female) == (False, True):
                        gender = 'female'
                    else:
                        gender = 'Not Specified'
    
                    pred_anno['gender'] = [gender]
    
                for key in ['relation', 'symptoms', 'opinion']:
                    if key in pred_anno:
                        binary_value = parseBinaryValue(pred_anno[key])
    
                        if key == 'opinion':
                            pred_anno[key] = ['effective'] if binary_value else ['not_effective']
                        else:
                            pred_anno[key] = ['yes'] if binary_value else ['Not Specified']
    
                data['predicted_annotation'] = pred_anno
    
                formatted_data_list.append(data)
    
        filename_new = f"reformatted_submission/{folder}/OURTEAM-global-{event}.jsonl"
        with jsonlines.open(filename_new, mode='w') as f:
            for data in formatted_data_list:
                f.write(formatData(data))

run_reformat(sys.argv[1])
