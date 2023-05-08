
# This will be an implementation of the IBM1 word alignment model from Czech to English
# Aligned sentences are in the format "English sentence . Czech sentence ." and stored in the file czenali (provided by the lecturer)
# Each aligned sentence is followed up by already decided word alignments - these are just for reference and shall not be used in the script
# Example of an aligned sentence: Arab rhetoric often takes the colonial experiences of Algeria and South Africa as models for their conflict with Zionism . Arabská politika často konflikt se sionismem přirovnává ke koloniálním zkušenostem Alžírska a Jižní Afriky . 1-1 10-12 11-13 12-14 14-7 17-4 18-5 19-6 20-15 3-3 4-7 6-9 7-10 9-11    13-7 15-8 2-2 5-10 8-11
# Results will be written to the file translation_dictionary.txt
# Usage of the script: python3 ibm1.py <number of sentence pairs to process> <number of training iterations>
# If no command line arguments are provided, default values of 1000 sentence pairs and 10 iterations will be used


# More details: https://ufal.mff.cuni.cz/courses/npfl124#assignment03
# Git repository: #ADD GIT LINK

import simplemma
import sys


class ibm1:
    special_chars = [';', ':', '"', "'", ',', '.', '-', '(', ')', '^', '&','#', '$', '%','~', '`', "?", "!"]
    sentence_pairs = set()
    english_words = set()
    czech_words = set()


    def preprocess_word(word: str, language: str):
        word_stripped = word.strip()
        for char in ibm1.special_chars:
            word_stripped = word_stripped.replace(char, "")
        word_lower_cased = word_stripped.lower()
        if len(word_lower_cased) == 0:
            return None
        if language == "en":
            word_lemmatized = simplemma.lemmatize(word_lower_cased, "en")
        elif language == "cs":
            word_lemmatized = simplemma.lemmatize(word_lower_cased, "cs")
        else:
            print("Error: Unknown language")

        word_lemmatized = word_lemmatized.lower()
        return word_lemmatized

    def preprocess_sentence(sentence: str, language: str):
        words = sentence.split()
        sentence_preprocessed = []
        for word in words:
            sentence_preprocessed.append(ibm1.preprocess_word(word, language))

        sentence_preprocessed = list(filter(None, sentence_preprocessed))
        return " ".join(sentence_preprocessed)



    def process_n_sentence_pairs(n: int):
        with open('czenali.txt', mode="r", encoding="utf-8") as f:
            print("Data retrieved.")
            for i, line in enumerate(f):
                if i >= n:
                    break
                if ((i + 1) % 10 == 0):
                    print("Processing line " + str(i + 1))
                sentences = line.split('\t')
                # preprocess the sentences
                english_sentence = ibm1.preprocess_sentence(sentences[0], "en")
                czech_sentence = ibm1.preprocess_sentence(sentences[1], "cs")
                # add the sentences to the respective sets
                sentence_tuple = (english_sentence, czech_sentence)
                ibm1.sentence_pairs.add(sentence_tuple)
                # add the words to the respective sets
                for word in english_sentence.split():
                    ibm1.english_words.add(word)
                for word in czech_sentence.split():
                    ibm1.czech_words.add(word)

        print("Done reading sentences")


    def train_ibm1(iterations: int):

        print("Starting the training process.")
        # Initialize translation probability t(e|c) uniformly
        t = {}
        for c in ibm1.czech_words:
            t[c] = {}
            for e in ibm1.english_words:
                t[c][e] = 1 / len(ibm1.english_words)

        for i in range(iterations):
            print(f"Running training iteration {i + 1} out of {iterations}:")
            # Initialize count(e|c) and total(c) to 0 for all e, c
            count = {}
            total = {}

            # Iterate over sentence pairs (e_s, c_s)
            for e_s, c_s in ibm1.sentence_pairs:
                e_words = e_s.split()
                c_words = c_s.split()

                # Calculate normalization factors total_s(e)
                total_s = {}
                for e in e_words:
                    total_s[e] = 0
                    for c in c_words:
                        total_s[e] += t[c][e]

                # Update counts and totals
                for e in e_words:
                    for c in c_words:
                        delta = t[c][e] / total_s[e]
                        if c not in count:
                            count[c] = {}
                        if e not in count[c]:
                            count[c][e] = 0
                        count[c][e] += delta
                        if c not in total:
                            total[c] = 0
                        total[c] += delta

            # Update translation probabilities t(e|c)
            for c in ibm1.czech_words:
                for e in ibm1.english_words:
                    t[c][e] = count[c].get(e, 0) / total[c]

        return t

    def extract_top_three_translations_for_words(translation_probabilities):

        with open('translation_dictionary.txt', mode="w", encoding="utf-8") as f:
            #sort the words according to their highest translation probability

            words_sorted = sorted(ibm1.czech_words, key=lambda x: max(translation_probabilities[x].values()), reverse=True)

            for c in words_sorted:
                #for each czech word, get a tuple with three english words with the highest translation probability and their corresponding probabilities
                 top_three = sorted(translation_probabilities[c].items(), key=lambda x: x[1], reverse=True)[:3]

                 #write the results to the file "translation_dictionary.txt"
                 f.write(c + ": ")
                 #use 3 decimal places for the probabilities
                 for e, prob in top_three:
                     f.write(e + " - " + str(round(prob, 3)) + " " )
                 f.write("\n")
        print("Translation dictionary extracted to the file translation_dictionary.txt")




def main():

    # The provided dictionary was created with 1000 sentence pairs and 10 iterations
    if len(sys.argv) != 3:
        sentence_pairs = 1000
        iterations = 10
    else:
        sentence_pairs = int(sys.argv[1])
        iterations = int(sys.argv[2])
    ibm1.process_n_sentence_pairs(sentence_pairs)
    translation_probabilities = ibm1.train_ibm1(iterations)
    ibm1.extract_top_three_translations_for_words(translation_probabilities)


if __name__ == "__main__":
    main()