import os


if __name__ == "__main__":
    for lang in ["English", "eng_TOEFL", "Finnish", "Italian", "Griko", "Ainu", "Russian"]:
        for lengthAvg in [True, False]:
            for add_k in [1, 0.1]:
                for corpus in ['table', 'confusion_set']:
                    os.system("python language_model.py --lang={} --corpus={} --lengthAvg={} --smoothing={} --add_k={} --save_model={} --save_result={}".format(
                        lang, corpus, lengthAvg, True, add_k, True, True 
                    ))
