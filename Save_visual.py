import random
import string
import pickle
import pdb

ascii_lowercase = 'abcdefghijklmnopqrstuvwxyzäöñáéóúüè̩àìçğöş'
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÑÁÉÓÚÜÈ̩ÀÌÇĞÖŞ'
cryllic = 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'

digits = '0123456789'
letters = ' '.join(ascii_lowercase).split() + ['-']


color = {
    'white':    "\033[1;37m",
    'yellow':   "\033[1;33m",
    'green':    "\033[1;32m",
    'blue':     "\033[1;34m",
    'cyan':     "\033[1;36m",
    'red':      "\033[1;31m",
    'magenta':  "\033[1;35m",
    'black':      "\033[1;30m",
    'darkwhite':  "\033[0;37m",
    'darkyellow': "\033[0;33m",
    'darkgreen':  "\033[0;32m",
    'darkblue':   "\033[0;34m",
    'darkcyan':   "\033[0;36m",
    'darkred':    "\033[0;31m",
    'darkmagenta':"\033[0;35m",
    'darkblack':  "\033[0;30m",
    'off':        "\033[0;0m"
}
print("Hello "+"\033[43;{}m{}\033[m".format(40, "world") + "!")
print("Hello "+"\033[{};100m{}\033[m".format(46, "world") + "!")
print("Hello "+"\033[{};100m{}\033[m".format(41, "world") + "!")


def text_visual(corpus, error_words, decisions_dict, error_indexes, check_list=False):
    stat_summary = {'FN': 0, 'TN': 0, 'TP': 0, 'FP': 0, 'check_list': 0}
    error_words_list = error_words.split()
    error_words_lower = [x.lower() for x in error_words_list]

    erroneous_text = []
    error_pointer = 0
    for i in range(len(corpus)):
        if i in error_indexes:
            try:
                erroneous_text.append(error_words_list[error_pointer])
                error_pointer += 1
            except:
                erroneous_text.append(corpus[i])
        else:
            erroneous_text.append(corpus[i])

    text_highlight = []

    for index, word in enumerate(erroneous_text):
        if word.lower() in decisions_dict.keys():
            decision = decisions_dict[word.lower()]['decision']

            # correct + decision 1
            if word.lower() == decisions_dict[word.lower()]['golden'] and decision == 1:
                text_highlight.append(color['darkwhite'] + word + color['off'])
                stat_summary['FN'] += 1

            # misspelled + decision 1
            elif word.lower() != decisions_dict[word.lower()]['golden'] and decision == 1:
                text_highlight.append(
                    color['cyan'] + decisions_dict[word.lower()]['golden'] + '({})'.format(word) + color['off'])
                stat_summary['TN'] += 1

                # misspelled + decision 0
            elif word.lower() != decisions_dict[word.lower()]['golden'] and decision == 0:
                text_highlight.append(
                    color['yellow'] + decisions_dict[word.lower()]['golden'] + '({})'.format(word) + color['off'])
                stat_summary['TP'] += 1

            # correct + decision 0
            elif word.lower() == decisions_dict[word.lower()]['golden'] and decision == 0:
                text_highlight.append(color['green'] + word + color['off'])
                stat_summary['FP'] += 1

            # checkout by list in the training data
            elif check_list and word.lower() != decisions_dict[word.lower()]['golden'] and decisions_dict[word.lower()][
                'dict']:
                text_highlight.append(
                    color['darkmagenta'] + decisions_dict[word.lower()]['golden'] + '({})'.format(word) + color['off'])
                stat_summary['check_list'] += 1

            # true misspelling
            elif word.lower() in error_words_lower:
                text_highlight.append(color['red'] + word + color['off'])

        else:
            if index in error_indexes:
                text_highlight.append(color['red'] + word + color['off'])
            else:
                text_highlight.append(color['black'] + word + color['off'])
    return text_highlight, stat_summary


def misspelling_generator_wrapper(word, letters, times):
    options = ['delete', 'transpose', 'replace', 'insert']
    for i in range(times):
        action = random.choice(options)
        word = misspelling_generator(word, letters, action)
    return word


def misspelling_generator(word, letters, action):
    length = len(word)
    if action == 'delete':
        position = random.choice(range(length))
        word = word[:position] + word[position + 1:]
    if action == 'transpose':
        if length > 1:
            position = random.choice(range(length - 1))
            word = word[:position] + word[position + 1] + word[position] + word[position + 1:]
        else:
            word = misspelling_generator(word, letters, 'replace')

    if action == 'insert':
        position = random.choice(range(length))
        insertion = random.choice(letters)
        word = word[:position] + insertion + word[position:]
    if action == 'replace':
        position = random.choice(range(length))
        insertion = random.choice(letters)
        word = word[:position] + insertion + word[position + 1:]
    return word

def generate_misspelling_seq(test_indomain, edit_prob=0.1, random_seed=0):
    random.seed(random_seed)
    edit_prob = 0.1
    data_clean_indomain = []
    test_indomain = [x.strip('\n') for x in test_indomain]
    test_indomain = ''.join([x for x in test_indomain if len(x) > 0]).split()
    data_clean_indomain = [x.strip(string.punctuation).strip(string.digits) for x in test_indomain]
    indexes = list(range(len(data_clean_indomain)))
    random.shuffle(indexes)
    threshold = int(edit_prob*len(indexes))
    error_indexes = indexes[:threshold]
    other_indexes = indexes[threshold:]
    error_words = [data_clean_indomain[x] for x in error_indexes]
    return data_clean_indomain, error_indexes, other_indexes, error_words

### Finish
test_indomain = """
The Beatles aloitti Beatles for Salen äänitykset vain kaksi kuukautta A Hard Day’s Nightin ilmestymisen jälkeen. Albumi nauhoitettiin kahden ja puolen kuukauden aikana, elokuun 1964 puolivälistä lokakuun loppuun Samaan aikaan yhtye esiintyi kiertueilla Britanniassa, Kanadassa ja Yhdysvalloissa ja lisäksi vielä radiossa ja televisiossa. Äänityksiin käytettiinkin aikaa yhteensä vain seitsemän päivää.[8] Kiireessä yhtye ei saanut aikaan tarpeeksi omaa materiaalia, joten albumin 14 kappaleesta vain kahdeksan on John Lennonin ja Paul McCartneyn kirjoittamia. Muut ovat lainakappaleita, joita yhtye oli aikoinaan esittänyt Liverpoolin The Cavern Clubilla sekä Hampurissa
Paul McCartney on todennut, ettei albumin äänitys vienyt paljon aikaa, koska siinä on vain muutama uusi kappale ja loput olivat samoja kuin yhtyeen konserttiohjelmistossa. George Harrison on kertonut nauhoituksista
Harjoittelimme tätä albumia varten vain niitä kappaleita, jotka olivat uusia. Sellaisia lauluja kuin ”Honey Don’t” ja ”Everybody’s Trying to Be My Baby” olimme esittäneet niin usein, että meidän piti vain saada niiden soundi kuntoon ja nauhoittaa ne. Mutta esimerkiksi ”Baby’s in Black” meidän piti opetella ja harjoitella George Harrison, soolokitaristi
Beatles for Sale oli yhtyeen toinen neliraitatekniikalla äänitetty albumi. Jo edellisellä albumilla A Hard Day’s Night esimerkiksi päällekkäisäänitykset oli voitu tehdä huomattavasti kätevämmin kuin kahdella ensimmäisellä albumilla Please Please Me ja With The Beatles. Beatles for Salen nauhoituksissa Beatles ja George Martin ottivat käyttöön uuden menettelytavan, jossa kappaleita toistettiin ja sovituksia hiottiin äänittämisen lomassa. Kappaleen nauhoituksia kuunneltiin ja sitä muutettiin, kunnes lopputulokseen oltiin tyytyväisiä. Uudesta toimintatavasta käy esimerkkinä 6. lokakuuta äänitetty ”Eight Days a Week”. Yhtye äänitti kappaleen iltapäivällä monta kertaa ja kokeili erilaisia ideoita, ja lopulta kuudetta nauhoitusta pidettiin parhaana. Äänitykset jatkuivat illalla, jolloin pyrittiin tekemään mahdollisimman hyvä soitintausta, ja lisäksi lauluraidat nauhoitettiin ensimmäiselle ja toiselle raidalle. Parhaaksi valittiin 13:s otto. Sen kolmannelle raidalle lisättiin vielä kättentaputukset, patarummut ja kitara, ja lisäksi viimeiselle raidalle äänitettiin tuplattu lauluraita. Lopun kitarakooda lisättiin eräänä toisena päivänä, ja alun häivytysefekti luotiin kappaletta miksatessa
Suurin osa levyn kappaleista äänitettiin toiseksi viimeisenä studiopäivänä, 18. lokakuuta 1964, jolloin yhtyeellä oli vapaapäivä Britannian-kiertueelta. Beatles äänitti tuolloin yhdeksässä tunnissa seitsemän kappaletta. Viittä niistä yhtye oli soittanut konserteissa, kuten lainakappaletta ”Mr. Moonlight”. Lisäksi nauhoitettiin itse tehdyt kappaleet ”I’ll Follow the Sun” ja ”I Feel Fine”, joista jälkimmäinen jäi albumin ulkopuolelle ja julkaistiin singlenä
"""

# princple
# reference: https://www.quora.com/In-languages-with-a-clear-pronunciation-e-g-Finnish-how-common-are-spelling-mistakes
# 1. insert/missing j for agent name
# 2. double/misduplicate the vowel
# 3. r -> d
# 4. lose part of the word
# 5. usage of '-' when multiple words get composed

error_text = "lokakuun albumia Me  joita lainakappaletta muutettiin The joten oltiin tuolloin albumin Clubilla Viittä George Harrison kahdella toiselle äänitettiin George Moonlight Please Sen Days Follow Martin raidalle jolloin Week Trying Yhdysvalloissa ja Nightin valittiin aloitti"
error_text1 = "lokakjuun albumjia Me  joijta lainajkappaletta muutettjiin Thje ojten oltjiin tuollojin albjumin Clubijlla Viijttä Georjge Harrjison kahdjella toisjelle äänijtettiin Georgje Moonlijght Pleajse jSen Djays Fjollow Martjin raijdalle olloin Weejk Tryjing Yhdysvalloijssa a Nighjtin valittijin aloitjti"
error_text_all = "lokakun alhbumia Me  jointa lainakkappaletta muutetiin Thea joden oltin tuollooin albjumin Cluubilla Viittää Geodge Haddison kaahdella toijselle ääniteettiin Geoorge Moonllight Pleas Seen Daays Folloow Madtin daidalle joolloin Wek Tdying Yhdysvaalloissa jaa Nightiin vallittiin alloitti"

### Spanish
test_domain = '''
The Beatles fue una banda de rock9 inglesa activa durante la década de 1960, y reconocida como la más exitosa comercialmente y la más alabada por la crítica en la historia de la música popular y de la música rock.10​11​12​13​14​ Formada en Liverpool, estuvo constituida desde 1962 por John Lennon (guitarra rítmica, vocalista), Paul McCartney (bajo, vocalista), George Harrison (guitarra solista, vocalista) y Ringo Starr (batería, vocalista). Enraizada en el skiffle y el rock and roll de los años 1950, la banda trabajó más tarde con distintos géneros musicales, tales como rock and roll y pop de los 50's, rock pop, música beat, folk rock, rock psicodélico hasta hard rock, incorporando a menudo elementos clásicos, entre otros, de forma innovadora en sus canciones. La naturaleza de su enorme popularidad, que había emergido primeramente con la moda de la «beatlemanía», se transformó al tiempo que sus composiciones se volvieron más sofisticadas. Llegaron a ser percibidos como la encarnación de los ideales progresistas, extendiendo su influencia en las revoluciones sociales y culturales de la década de 1960.
Inicialmente se trató de un grupo de estilo skiffle, integrado por adolescentes en etapa escolar, fundado en 1956 por Lennon durante la llamada "locura del skiffle" (skiffle craze) en Gran Bretaña, con el nombre de The Quarry Men, al que luego se sumaron McCartney y Harrison. Luego de varios cambios de integrantes y nombres, decantó en The Beatles, con una formación de cinco miembros que incluía a Lennon, McCartney, Harrison, Stuart Sutcliffe (bajo) y Pete Best (batería). Construyó su reputación en los clubes de Liverpool y Hamburgo sobre un período de tres años a partir de 1960. Sutcliffe abandonó la formación en 1961, y Best fue reemplazado por Starr al año siguiente. Establecidos como grupo profesional después de que Brian Epstein les ofreciera ser su representante, y con su potencial musical mejorado por la creatividad del productor George Martin, lograron éxito comercial en el Reino Unido a finales de 1962 con su primer sencillo, «Love Me Do». A partir de ahí, fueron adquiriendo popularidad internacional a lo largo de los siguientes años, en los cuales hicieron un extenso número de giras hasta 1966, año en que cesaron la actividad en vivo para dedicarse únicamente a la grabación en el estudio hasta su disolución en 1970. Después, todos sus integrantes se embarcaron en exitosas carreras independientes de diversa duración. Lennon sería asesinado a las afueras de su casa de Nueva York en 1980, y Harrison fallecería de cáncer en 2001. McCartney y Starr, los dos miembros sobrevivientes, aún permanecen musicalmente activos.
Durante sus años de estudio crearon algunos de sus mejores materiales, incluyendo el álbum Sgt. Pepper's Lonely Hearts Club Band (1967), considerado por muchos como una obra maestra. Cuatro décadas después de su separación, la música que crearon continúa siendo popular. Se mantienen como el grupo con más números uno en las listas británicas, situando más álbumes en esta posición que cualquier otra agrupación musical.15​ De acuerdo con las certificaciones de la RIAA, han vendido más discos en los Estados Unidos que cualquier otro artista.13​ Fueron galardonados con siete premios Grammy,16​ y recibieron un total de quince premios Ivor Novello de parte de la British Academy of Songwriters, Composers and Authors.17​ En 2004, la revista Rolling Stone los clasificó en el número uno en su lista de los «100 artistas más grandes de todos los tiempos».10​ De acuerdo con la misma publicación, la música innovadora de The Beatles y su impacto cultural ayudaron a definir los años 1960.18​ En 2010, el canal de televisión especializado en música VH1 los clasificó en el número uno en su lista de los «100 artistas más grandes de todos los tiempos».19​ También fueron colocados en el puesto número 1 por el sitio de Internet Acclaimed Music en su lista «The Top 1000 Artists of All Time».11​ Figuran, asimismo, en la primera posición como los más grandes artistas de todos los tiempos de las listas Hot 100 y Billboard 200 en la clasificación de Billboard de 2015
'''

data_clean, error_indexes, other_indexes, errs = generate_misspelling_seq(test_domain, edit_prob=0.1, random_seed=0)

# princple
# reference: https://www.grittyspanish.com/2018/07/26/spanish-spelling-mistakes/
# 1. b and v pronouns the same
# 2. ll -> y, e->a
# 3. missing h (since h is silent)
# 4. misspelled c with s and z(when combined with a and o and becomes soften) (doesn't happen to hard c)
# 5. j and g (softened ones sounds like h)
# 6. l - > r
# 7. missing masculine and feminiline (ending o and a)

# Schem 1-5
err_text_spn_orig = 'especializado a con en colocados en Liverpool se la al televisión de miembros representante Men popular  los estudio ofreciera los de John progresistas las otros y hasta estilo formación miembros otro la el sociales afueras McCartney a puesto de Band Time primera de Composers The extenso llamada en la artista menudo en al de una más clasificación las por un sumaron Hearts el como of menudo'
err_text_spn = 'especialicado e cong en colocadas en Livepool ce las all televición te mllembros representande Man porular loos esdudio ofresiera alos der Jojn progrecistas les odros ll jasta estiro formasión membros atro ala ell sosiales avueras McCartnell at puesdo del Bend Time premera te Compasers Te exdenso yamada en sla articta menudo en al de una más clasificación las por un sumeron Heearts el somo af menuto'

err_text_spn_auto = [misspelling_generator_wrapper(x, letters, 1) for x in err_text_spn_orig.split()]

with open("./online_test/Spanish/artif_Spanish_test_Beatles.pkl", 'rb') as fp:
    pall_decisions = pickle.load(fp)

origin_text = []
for index, word in enumerate(data_clean):
    if index in error_indexes:
        origin_text.append(color['red'] + word + color['off'])
    else:
        origin_text.append(color['black'] + word + color['off'])

misspl_rate = len(error_indexes) / len(data_clean)
text_highlight_all, stat_summary = text_visual(data_clean, err_text_spn, pall_decisions, error_indexes, check_list=True)

print(color['darkwhite'] + "misclassified correct spelling (FN): {}".format(round(stat_summary['FN']/(len(data_clean) - len(error_indexes)), 3)) + color['off'])
print(color['cyan'] + "correctly classified misspelling (TN): {}".format(round(stat_summary['TN']/len(error_indexes), 3)) + color['off'])
print(color['green'] + "correctly classified correct spelling (TP): {}".format(round(stat_summary['FP']/(len(data_clean) - len(error_indexes)), 3)) + color['off'])
print(color['yellow'] + "misclassified misspelling (FP): {}".format(round(stat_summary['TP']/len(error_indexes), 3)) + color['off'])
print(color['darkmagenta'] + "check by the list: {}".format(round(stat_summary['check_list']/len(error_indexes), 3)) + color['off'])
print(color['red'] + "missed misspelling/golden misspelling (Negative samples)" + color['off'])
print(color['black'] + "other words" + color['off'])
print(color['red'] + 'misspelling in the bracket: {}'.format(round(misspl_rate, 3)) + color['off'])
# print("\n Origin Text \n")
# print(' '.join(origin_text))
print("\n" + color['black'] + " All Schemes" +  color['off'] + "\n")
print(' '.join(text_highlight_all))

lang = 'Russian'
with open('./files/{}/auto_{}_test_Beatles.txt'.format(lang, lang), 'w') as fp:
    count = 0
    err_words = err_text_ru_auto
    err_words_orig = err_text_ru_orig.split()
    for i, word in enumerate(data_clean):
        if i in error_indexes:
            try:
                fp.write(err_words[count] + ' 1 ' + err_words_orig[count])
            except:
                pdb.set_trace()
            count += 1
            fp.write('\n')
        else:
            fp.write(data_clean[i] + ' 0 ' + data_clean[i])
            fp.write('\n')
    fp.close()

### Italian
test_domain = '''
The Beatles è stato un gruppo musicale inglese fondato a Liverpool nel 1960, composto da John Lennon, Paul McCartney, George Harrison e Ringo Starr (quest’ultimo a partire dal 1962, chiamato a sostituire Pete Best; della prima formazione faceva parte anche Stuart Sutcliffe), e attivo fino al 1970.
Ritenuti un fenomeno di comunicazione di massa di proporzioni mondiali i Beatles hanno segnato un'epoca nella musica, nel costume, nella moda e nella pop art.[11] A distanza di vari decenni dal loro scioglimento ufficiale – e dopo la morte di due dei quattro componenti – i Beatles contano ancora un enorme seguito e numerosi sono i loro fan club esistenti in ogni parte del mondo
Stando alle stime dichiarate hanno venduto a livello mondiale un totale di circa 600 milioni di copie fra album, singoli e musicassette, di cui oltre 170 milioni nei soli Stati Uniti d'America, risultando fra gli artisti di maggior impatto e successo e, negli Stati Uniti, quelli con il maggior numero di vendite. Sono inoltre al primo posto della lista dei 100 migliori artisti secondo Rolling Stone
L'aura che circonda lo sviluppo del loro successo mediatico e che ha favorito la nascita della cosiddetta Beatlemania e lo straordinario esito artistico raggiunto come musicisti rock sono inoltre oggetto di studio di università, psicologi e addetti del settore
Durante la loro carriera decennale sono stati ufficialmente autori di 186 canzoni la maggior parte delle quali portano la firma Lennon-McCartney. 
'''

data_clean, error_indexes, other_indexes, errs = generate_misspelling_seq(test_domain, edit_prob=0.1, random_seed=0)

# principle
# 1. replace f with ph
# 2. S -> C
# 3. random delete vowel
# 4. double r
err_text_ita_orig = 'fino primo Stuart Lennon-McCartney della dei stime e gli oltre di partire mondiale quattro nella università che al la fra hanno nei venduto forza'
err_text_ita = 'phino premo Ctruart Leinnon-McCarrtney dela dai stim ei glio olter de partre mondiare quattro nella univerrsità cher eal lar frra hannor ne benduto froza'

err_text_ita_auto = [misspelling_generator_wrapper(x, letters, 1) for x in err_text_ita_orig.split()]
with open("./online_test/Italian/artif_Italian_test_Beatles2.pkl", 'rb') as fp:
    pall_decisions = pickle.load(fp)

origin_text = []
for index, word in enumerate(data_clean):
    if index in error_indexes:
        origin_text.append(color['red'] + word + color['off'])
    else:
        origin_text.append(color['black'] + word + color['off'])

misspl_rate = len(error_indexes) / len(data_clean)
text_highlight_all, stat_summary = text_visual(data_clean, err_text_ita, pall_decisions, error_indexes, check_list=True)
print(color['darkwhite'] + "misclassified correct spelling (FN): {}".format(
    round(stat_summary['FN'] / (len(data_clean) - len(error_indexes)), 3)) + color['off'])
print(color['cyan'] + "correctly classified misspelling (TN): {}".format(
    round(stat_summary['TN'] / len(error_indexes), 3)) + color['off'])
print(color['green'] + "correctly classified correct spelling (TP): {}".format(
    round(stat_summary['FP'] / (len(data_clean) - len(error_indexes)), 3)) + color['off'])
print(color['yellow'] + "misclassified misspelling (FP): {}".format(round(stat_summary['TP'] / len(error_indexes), 3)) +
      color['off'])
print(color['darkmagenta'] + "check by the list: {}".format(round(stat_summary['check_list'] / len(error_indexes), 3)) +
      color['off'])
print(color['red'] + "missed misspelling/golden misspelling (Negative samples)" + color['off'])
print(color['black'] + "other words" + color['off'])
print(color['red'] + 'misspelling in the bracket: {}'.format(round(misspl_rate, 3)) + color['off'])
# print("\n Origin Text \n")
# print(' '.join(origin_text))
print("\n" + color['black'] + " All Schemes" + color['off'] + "\n")
print(' '.join(text_highlight_all))

