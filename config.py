import logging
from logging import handlers
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_filename = './log/random_seed_training.txt'
logging.basicConfig(level=logging.INFO,
                    filename=log_filename,
                    filemode='a',
                    format=fmt)
logger = logging.getLogger(__name__)

format_str = logging.Formatter(fmt)
# self.logger.setLevel(self.level_relations.get(level))#设置日志级别
sh = logging.StreamHandler() # Cast to terminal
sh.setFormatter(format_str) # Format displayed at terminal
 
th = handlers.TimedRotatingFileHandler(filename=log_filename,when='D',backupCount=3,encoding='utf-8')
logger = logging.getLogger(log_filename)
logger.addHandler(sh)
logger.addHandler(th)

data_dict = {
    'ru_btl': 'Russian',
    'ru_wiki_rule': 'Russian',
    'trk': 'Turkish',
    'trk_wiki_rule': 'Turkish',
    'ita': 'Italian',
    'ita_wiki_rule': 'Italian',
    'spa': 'Spanish',
    'spa_wiki_rule': 'Spanish',
    'fin': 'Finnish',
    'fin_wiki_rule': 'Finnish',
    'en': 'English',
    'ainu': 'Ainu',
    'griko': 'Griko',
    'Greek': 'el'
}

category = ['language', 'city', 'landmark', 'politician', 'nation',
            'company', 'football club', 'untiversity', 'history', 'event', 'currency', 'other']

langs = ['Russian', 'Finnish', 'Spanish', 'Italian', 'Turkish', 'English']
lang2code = {'Russian': 'ru', 'Finnish': 'fi', 'Spanish': 'es', 'Italian': 'it', 'Turkish': 'tr', 'English': 'en',
             'Ainu': 'ainu', 'Griko': 'griko', 'Greek': 'el'}
data2langcode = {"ru": "ru",
                 'en': "en",
                 'fin': "fi", 'fin_wiki': "fi", 'fin_wiki_rule': "fi",
                 'ru_btl': "ru", 'ru_wiki': "ru", 'ru_wiki_rule': "ru",
                 'ita': "it", 'ita_wiki': "it", 'ita_wiki_rule': "it",
                 'spa': "es", 'spa_wiki': "es", 'spa_wiki_rule': "es",
                 'trk': "tr", 'trk_wiki': "tr", 'trk_wiki_rule': "tr",
                 'ainu': 'ainu',
                 'griko': 'griko',
                 'el': 'el'}

glagolitsa = "А,Б,В,Г,Д,Е,Ё,Ж,З,И,Й,К,Л,М,Н,О,П,Р,С,Т,У,Ф,Х,Ц,Ч,Ш,Щ,Ъ,Ы,Ь,Э,Ю,Я"
ascii_lowercase = 'abcdefghijklmnopqrstuvwxyzäñáéóòúüùè̩àâíìïîñçğöş¿¡' 
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÑÁÉÓÒÚÜÙÈ̩ÀÂÌÍÏÎÑÇĞÖŞ¿¡'
turkish_lowercase = 'abcçdefgğhiıjklmnoöpqrsştuüvyzáéóòúüùè̩àâäíìïîñ'
turkish_uppercase = 'ABCÇDEFGĞHİIJKLMNOÖPQRSŞTUÜVYZÁÉÓÒÚÜÙÈ̩ÀÂÄÌÍÏÎÑ'
greek_lowercase = 'αβγδϵζηθικλμνξοπρστυϕχψωεϑϰϖϱςφϒϝϟϛϡϜϞϚϠ'
digits = '0123456789'

Glagolitsa = glagolitsa.split(',') + ' '.join(ascii_lowercase).split() \
             + ' '.join(ascii_uppercase).split() + ' '.join(digits).split()
glagolitsa_pure = [x.lower() for x in glagolitsa.split(',')]
glagolitsa = glagolitsa.lower().split(',') + ' '.join(ascii_lowercase).split() + ' '.join(digits).split() + ['.']
letters = ' '.join(ascii_lowercase).split() + ' '.join(digits).split() + ['.', "'", '='] # need to add =?
letters_pure = ' '.join(ascii_lowercase + ".'=").split() # need to add .
en_letters_pure = ' '.join('abcdefghijklmnopqrstuvwxyz'+'.').split()
tr_letters = list(set(' '.join(turkish_lowercase).split() + ' '.join(turkish_uppercase.lower()).split())) + ' '.join(digits).split() + ['.', "'"]
tr_letters_pure = list(set(' '.join(turkish_lowercase).split() + ' '.join(turkish_uppercase.lower()).split())) + ['.', "'"]
greek_letters = ' '.join('αβγδϵζηθικλμνξοπρστυϕχψωεϑϰϖϱςφϒϝϟϛϡϜϞϚϠ'+"'.").split() + ['.', "'"]
greek_letters_pure = ' '.join('αβγδϵζηθικλμνξοπρστυϕχψωεϑϰϖϱςφϒϝϟϛϡϜϞϚϠ').split()

langCode2alphabet = {'ru': glagolitsa, 'fi': letters, 'es': letters, 'it': letters, 'tr': tr_letters,
                     'en': letters, 'el':greek_letters}
langCode2alphabet_pure = {'ru': glagolitsa_pure, 'fi': letters_pure, 'es': letters_pure,
                          'it': letters_pure, 'tr': tr_letters_pure, 'en': en_letters_pure,
                          'el':greek_letters_pure}

choice = ['l2ll', 'll2l', 'r2rr', 'rr2r', 'l2r', 'n2nn', 'f2ph', 'p2b', 's2c', 'e2a', 'b2v', 'r2d', 'v2b',
          'double_vowel', 'replace_vowel', 'delete_vowel',
          'c/s/z', 'j/g', 'remove_slient_cons',
          'partialdrop', 'insert j', 'drop_h',
          'replace', 'transpose', 'delete', 'insert',
          'origin']

rules = {'en': ['l2ll', 'll2l', 'r2rr', 'rr2r', 'l2r', 'n2nn', 'f2ph', 'p2b', 's2c', 'e2a', 'b2v', 'r2d', 'v2b',
                'double_vowel', 'replace_vowel', 'delete_vowel',
                'c/s/z', 'j/g', 'drop_h'],
         'fi': ['insert_j', 'r2d', 'partialdrop', 'double_vowel', 'replace_vowel', 'delete_vowel'],
         'es': ['v2b', 'll2l', 'e2a', 'drop_h', 'c/s/z', 'j/g', 'l2r', 'replace_vowel', 'delete_vowel'],
         'it': ['v2b', 'll2l', 'e2a', 'drop_h', 'c/s/z', 'j/g', 'l2r', 'replace_vowel', 'delete_vowel'],
         # 'it': ['f2ph', 's2c', 'r2rr', 'replace_vowel', 'delete_vowel'],
         'ru': ['l2ll', 'll2l', 'p2b', 'b2v', 'r2d', 'v2b', 'remove_slient_cons', 'replace_vowel', 'delete_vowel'],
         'tr': ['l2ll', 'n2nn', 'b2v', 'replace_vowel', 'delete_vowel'],
         'griko': ['n/i/u/ei/oi/ui', 'o/ou', 'ch/g', 't/th', 'd/nt', 'p/mp',
                   's/k', 'ps/ms', 'gg/gk', 'mm2m', 'll2l',
                   'double_vowel', 'replace_vowel', 'delete_vowel'],
         'el': ['η/ι/υ/ει/οι/υι', 'ο/ω', 'ε/αι', 'μ/ν', 'χ/γ', 'τ/θ',
                'δ/ντ', 'π/μπ', 'σ/ζ', 'ψ/πσ', 'ξ/κσ', 'γγ/γκ', 
                'λλ/λ', 'μμ/μ', 'σσ/σ', 'χχ/χ', 'ρρ/ρ', 'ππ/π', 'ξξ/ξ', 'νν/ν', 'κκ/κ', 'δδ/δ', 'ββ/β',
                'double_vowel', 'replace_vowel', 'delete_vowel']}

vowels = {'en': ['a', 'e', 'i', 'o', 'u'],
          'fi': ['i', 'y', 'u', 'e', 'o', 'ö', 'a', 'ä'],
          'es': ['a', 'e', 'i', 'y', 'o', 'u'],
          'it': ['a', 'e', 'i', 'o', 'u'],
          'ru': ['а', 'о', 'у', 'ы', 'э', 'и'],
          'tr': ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü'],
          'griko': ['a', 'e', 'i', 'o', 'u'],
          'el':['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']}

# rules see https://en.wikipedia.org/wiki/Romanization_of_Greek
Latin2Kiril = {'l': 'л', 'r': 'р', 'n': 'н', 's': 'с', 'p': 'п',
               'b': 'б', 'c': 'с', 'e': 'е', 'a': "а", 'ph': 'ф',
               'v': 'в', 'd': 'д', 'z': 'з', 'g': 'г',
               'll': 'лл', 'rr': 'рр', 'nn': 'нн'}

Latin2Greek = {'α' : 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ϵ': 'e',
               'z':'ζ', 'h':'η', 'th':'θ', 'i':'ι', 'k':'κ', 
               'l':'λ', 'm':'μ', 'n':'ν', 'x':'ξ', 'o':'ο',
               'pi':'π', 'r':'ρ', 's':'σ', 't':'τ', 'y':'υ',
               'f':'ϕ', 'ch':'χ', "ps":'ψ', 'ωεϑϰϖϱςφϒϝϟϛϡϜϞϚϠ':''}

Greek2Latin = {value:key for key, value in Latin2Greek.items()}

choice2id = {i:c for i,c in enumerate(choice)}
id2choice = {v:k for k, v in choice2id.items()}

summary_columns = ['accuracy', 'batch_size', 'dev_acc', 'dev_acc_pos', 'dev_f1', 'dropout',
       'early_stop', 'embedding_size', 'epoch_stop', 'hidden_size',
       'iteration_stop', 'lang', 'langcode', 'loss', 'lr', 'lr_decay',
       'max_token_length', 'minion_group_neg_ratio', 'minion_group_size',
       'mode', 'n_layers', 'protect_epoch', 'random_seed', 'trigger',
       'tune_epoch', 'volume', 'weight', 'correct_acc', 'correct_f1',
       'correct_precision', 'correct_probs', 'correct_recall', 'test2_acc',
       'test2_f1', 'test2_precision', 'test2_probs', 'test2_recall',
       'test_acc', 'test_f1', 'test_precision', 'test_probs', 'test_recall',
       'stop_mode', 'last_dev_acc', 'last_dev_f1', 'misspell_mode', 'corpus',
       'skip']


# Language Model Parameters
length_avg = True
save_result = True
save_model = True
smoothing = True
k = 1