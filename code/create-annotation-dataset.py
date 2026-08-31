import csv
import morphology as mir

INPUT_PATH = "resources/raw-sentences/max_opts_4.txt"
OUTPUT_PATH = "resources/samples.csv"

for i in range(1):
    with open(INPUT_PATH, "r", encoding='utf-8') as f:
        sentences = f.read().split("\n")

    data = []
    for sent_id, sentence in enumerate(sentences):
        sent_options = mir.sentence_word_options(sentence)
        if len(sent_options) == len(sentence.split()):
            for word_id, element in enumerate(sent_options):
                word = element[0]
                options = element[1]
                options.append("None of the above")
                row = [sent_id, word_id, sentence, word]
                row.append(options)
                data.append(row)
        else:
            continue

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["sentence-id", "word-id", "sentence", "word", "options"])
        for d in data:
            writer.writerow(d)