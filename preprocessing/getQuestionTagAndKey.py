

def getQuestionTagAndKeyList(subtask):

    if subtask == 'positive':
        question_tag_and_key_list = [
            ("age"           , "part2-age.Response"           ),
            ("close_contact" , "part2-close_contact.Response" ),
            ("employer"      , "part2-employer.Response"      ),
            ("gender_male"   , "part2-gender.Response"        ),
            ("gender_female" , "part2-gender.Response"        ),
            ("name"          , "part2-name.Response"          ),
            ("recent_travel" , "part2-recent_travel.Response" ),
            ("relation"      , "part2-relation.Response"      ),
            ("when"          , "part2-when.Response"          ),
            ("where"         , "part2-where.Response"         )]
    elif subtask == 'negative':
        question_tag_and_key_list = [
            ("age"           , "part2-age.Response"           ),
            ("close_contact" , "part2-close_contact.Response" ),
            ("gender_male"   , "part2-gender.Response"        ),
            ("gender_female" , "part2-gender.Response"        ),
            ("how_long"      , "part2-how_long.Response"      ),
            ("name"          , "part2-name.Response"          ),
            ("relation"      , "part2-relation.Response"      ),
            ("when"          , "part2-when.Response"          ),
            ("where"         , "part2-where.Response"         )]
    elif subtask == 'can_not_test':
        question_tag_and_key_list = [
            ("relation" , "part2-relation.Response"),
            ("symptoms" , "part2-symptoms.Response"),
            ("name"     , "part2-name.Response"    ),
            ("when"     , "part2-when.Response"    ),
            ("where"    , "part2-where.Response"   )]
    elif subtask == 'death':
        question_tag_and_key_list = [
            ("age"      , "part2-age.Response"      ),
            ("name"     , "part2-name.Response"     ),
            ("relation" , "part2-relation.Response" ),
            ("symptoms" , "part2-symptoms.Response" ),
            ("when"     , "part2-when.Response"     ),
            ("where"    , "part2-where.Response"    )]
    elif subtask == 'cure_and_prevention':
        question_tag_and_key_list = [
            ("opinion", "part2-opinion.Response"  ),
            ("what_cure", "part2-what_cure.Response"),
            ("who_cure", "part2-who_cure.Response" )]
    else:
        raise ValueError(f"Wrong subtask name, input is {subtask}")


    return question_tag_and_key_list
