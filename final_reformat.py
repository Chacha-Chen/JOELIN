
import os
import jsonlines

import spacy
nlp = spacy.load("en_core_web_sm")

event_list = ['positive', 'negative', 'can_not_test', 'death', 'cure']

for event in event_list:
    filename = f"./Test_Positive/Test_Positive-{event}.jsonl"

    count = 0
    person_count = 0
    loc_count = 0
    with jsonlines.open(filename, mode='r') as f:

        print(filename)
        
        processed_data_list = []
        for data in f:
            pred_anno = data['predicted_annotation']

            for key in ['part2-where.Response', 'part2-recent_travel.Response']:
                if key in pred_anno:
                    tmp_list = []
                    for x in pred_anno[key]:
                        if x in ['i', 'I', 'AUTHOR_OF_THE_TWEET', 'AUTHOR OF THE TWEET']:
                            tmp_list.append('NEAR AUTHOR OF THE TWEET')
                        else:
                            tmp_list.append(x)
                    pred_anno[key] = tmp_list

            if event == 'negative':
                _ = pred_anno.pop('part2-how_long.Response')

            if event == 'death':
                _ = pred_anno.pop('part2-symptoms.Response')
                

            if event == 'positive':
                if pred_anno["part2-where.Response"] == pred_anno["part2-name.Response"]:
                    if pred_anno["part2-where.Response"] == pred_anno["part2-recent_travel.Response"]:
                        if 'Not Specified' not in pred_anno["part2-where.Response"]:
                            # print(pred_anno["part2-where.Response"],
                            #       len(pred_anno["part2-where.Response"]))
                            doc = nlp(pred_anno["part2-where.Response"][0])
                            label_list = [token.label_ for token in doc.ents]
                            if 'PERSON' in label_list or "ORG" in label_list or 'NORP' in label_list:
                                pred_anno["part2-where.Response"] = ['Not Specified']
                                pred_anno["part2-recent_travel.Response"] = ['Not Specified']
                            #     person_count += 1
                            # if 'GPE' in label_list or 'LOC' in label_list or 'FAC' in label_list:
                            #     loc_count += 1
                            #     print('Label:', label_list)
                            # print("")
                            
                            count += 1
            if event == 'negative':
                if pred_anno["part2-where.Response"] == pred_anno["part2-name.Response"]:
                        if 'Not Specified' not in pred_anno["part2-where.Response"]:
                            # print(pred_anno["part2-where.Response"])
                            # count += 1
                            doc = nlp(pred_anno["part2-where.Response"][0])
                            label_list = [token.label_ for token in doc.ents]
                            if 'PERSON' in label_list or "ORG" in label_list or 'NORP' in label_list:
                                pred_anno["part2-where.Response"] = ['Not Specified']


            data['predicted_annotation'] = pred_anno
            processed_data_list.append(data)

    save_folder = 'Test_Positive_formatted_hard_coded_rule'
    os.makedirs(f'./{save_folder}', exist_ok=True)
    filename_new = f"./{save_folder}/Test_Positive-{event}.jsonl"
    with jsonlines.open(filename_new, mode='w') as f:
        for data in processed_data_list:
            f.write(data)
            

