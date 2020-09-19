
import re
import emoji
import unidecode
import unicodedata
from html import unescape

import logging

def asciifyEmojis(text):
    """
    Converts emojis into text aliases. E.g. 👍 becomes :thumbs_up:
    For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
    """
    text = emoji.demojize(text)
    return text


def lowerCase(text):
    return text.lower()


def standardizePunctuation(text):
    return ''.join(
        [unidecode.unidecode(t)
         if unicodedata.category(t)[0] == 'P' else t for t in text])


def removeUnicodeSymbols(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
    return text


def removeAccentedCharacters(text):
    text = unidecode.unidecode(text)
    return text


def replaceTags(text: str):

    covid_tag_set = { '#COVID19', '#coronavirus', '#Coronavirus',
                      '#COVID', '#Covid', '#Covid19', '#Covid_19',
                      '#covid19', '#CoronaVirus',
                      '#CoronavirusOutbreak', '#CoronavirusPandemic',
                      '#CoronaVirusUpdate', '#COVID2019', '#Corona',
                      '#CoronaVirusUpdates', '#corona' }
    covid_tag_exist = False

    new_text = []
    for word in text.split():
        if word.startswith('#'):
            if word in covid_tag_set:
                if not covid_tag_exist:
                    new_text.append("<COVID_TAG>")
                    covid_tag_exist = True
            # Remove the other tags
        else:
            new_text.append(word)

    new_text = " ".join(new_text)
    return new_text


def getTextProcessingFuncList(args):

    stats = {
        'lowerCase': 'Off',
        'removeAccentedCharacters': 'Off',
        'asciifyEmojis': 'Off',
        'standardizePunctuation': 'Off',
        'removeUnicodeSymbols': 'Off',
        'replaceTags': 'Off'}

    func_list = []
    if args.force_lower_case:
        func_list.append(lowerCase)
        stats['lowerCamel'] = 'On'

    if args.remove_accented_characters:
        func_list.append(removeAccentedCharacters)
        stats['removeAccentedCharacters'] = 'On'


    if args.clean_all or args.asciify_emojis:
        func_list.append(asciifyEmojis)
        stats['asciifyEmojis'] = 'On'

    if args.clean_all or args.standardize_punctuation:
        func_list.append(standardizePunctuation)
        stats["standardizePunctuation"] = 'On'

    if args.clean_all or args.remove_unicode_symbols:
        func_list.append(removeUnicodeSymbols)
        stats['removeUnicodeSymbols'] = 'On'

    if args.clean_all or args.replace_tags:
        func_list.append(replaceTags)
        stats['replaceTags'] = 'On'

    log_process_func(stats)
        
    return func_list

def log_process_func(stats):
    for key, val in stats.items():
        logging.info(f'[TextProcess] {key} option is {val}')
