from math import sqrt

from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance


def task_1_a(sentences):
    print("\nTASK 1 a) Multimi si vectori bag-of-words")
    print("Folosim litere mici, eliminam punctul si pastram cuvantul 'the'.")
    print("Multimile retin cuvintele distincte; vectorii numara aparitiile.")

    tokens = [sentence.lower().replace(".", "").split() for sentence in sentences]
    word_sets = [set(words) for words in tokens]
    vocabulary = sorted(word_sets[0] | word_sets[1] | word_sets[2])
    vectors = []

    for words in tokens:
        vector = [words.count(word) for word in vocabulary]
        vectors.append(vector)

    print("Vocabular comun:", vocabulary)

    for index in range(len(sentences)):
        print(f"\nS{index + 1}: {sentences[index]}")
        print("Tokenuri:", tokens[index])
        print("Multime: {" + ", ".join(sorted(word_sets[index])) + "}")
        print("Vector bag-of-words:", vectors[index])

    return tokens, word_sets, vectors


def task_1_b(word_sets, vectors):
    print("\nTASK 1 b) Similaritatea Jaccard si similaritatea cosinus")
    print("Jaccard compara multimile; cosinus compara vectorii bag-of-words.")
    print("Jaccard = marimea intersectiei / marimea reuniunii.")
    print("Cosinus = produs scalar / produsul normelor vectorilor.")

    for index in [1, 2]:
        intersection = word_sets[0] & word_sets[index]
        union = word_sets[0] | word_sets[index]
        jaccard = len(intersection) / len(union)

        vector1 = vectors[0]
        vector2 = vectors[index]
        products = [x * y for x, y in zip(vector1, vector2)]
        dot_product = sum(products)
        square_sum1 = sum(value ** 2 for value in vector1)
        square_sum2 = sum(value ** 2 for value in vector2)
        cosine = dot_product / (sqrt(square_sum1) * sqrt(square_sum2))

        print(f"\nS1-S{index + 1}")
        print("Intersectie: {" + ", ".join(sorted(intersection)) + "}")
        print("Reuniune: {" + ", ".join(sorted(union)) + "}")
        print(f"Jaccard = {len(intersection)} / {len(union)} = {jaccard:.4f}")
        print(
            "Produs scalar = "
            + " + ".join(str(value) for value in products)
            + f" = {dot_product}"
        )
        print(f"Norma S1 = sqrt({square_sum1})")
        print(f"Norma S{index + 1} = sqrt({square_sum2})")
        print(
            f"Cosinus = {dot_product} / "
            f"(sqrt({square_sum1}) * sqrt({square_sum2})) = {cosine:.4f}"
        )


def task_1_c(tokens):
    print("\nTASK 1 c) Distanta de editare la nivel de token")
    print("Comparam liste de cuvinte, nu siruri de caractere.")
    print("Inserarea, stergerea si substitutia unui token au fiecare cost 1.")
    print("Nu folosim transpozitii. Calculam distanta Levenshtein.")

    for index in [1, 2]:
        distance = edit_distance(
            tokens[0], tokens[index], substitution_cost=1, transpositions=False
        )

        print(f"\nS1-S{index + 1}: distanta de editare = {distance}")
        print("O transformare minima pentru aceasta pereche:")

        for word1, word2 in zip(tokens[0], tokens[index]):
            if word1 != word2:
                print(f"Substitutie: {word1} -> {word2}")


def task_1_d():
    print("\nTASK 1 d) Relatii si similaritate in WordNet")
    print("Alegem sensurile de substantive: medic si copil, persoana tanara.")
    print("Afisam identificatorul cerut si numele canonic al synset-ului.")

    synset_pairs = [
        ("doctor.n.01", "physician.n.01"),
        ("child.n.01", "kid.n.01"),
    ]

    for name1, name2 in synset_pairs:
        synset1 = wordnet.synset(name1)
        synset2 = wordnet.synset(name2)

        print(f"\n{name1} -> {synset1.name()}")
        print("Definitie:", synset1.definition())
        print(f"{name2} -> {synset2.name()}")
        print("Definitie:", synset2.definition())
        print("Leme in primul synset:", ", ".join(synset1.lemma_names()))
        print("Sinonime in sensurile alese:", synset1 == synset2)
        print("Path similarity:", synset1.path_similarity(synset2))

    print("\nPath similarity este 1 pentru doua sensuri din acelasi synset.")


def task_1_e():
    print("\nTASK 1 e) Interpretarea rezultatelor")
    print("Perechea S1-S2 este mai apropiata semantic.")
    print("Doctor si physician sunt sinonime in sensul de medic.")
    print("Child si kid sunt sinonime in sensul de persoana tanara.")
    print("In S1 si S2, medicul examineaza copilul.")
    print("In S3, copilul examineaza medicul: rolurile sunt inversate.")

    print("\nReprezentarile prin multimi si bag-of-words nu surprind corect diferenta.")
    print("S1 si S3 au aceleasi cuvinte si aceleasi frecvente.")
    print("De aceea, Jaccard si cosinus dau 1, desi propozitiile nu spun acelasi lucru.")
    print("Aceste reprezentari pierd ordinea cuvintelor si rolurile participantilor.")

    print("\nDistanta de editare tine cont de ordinea tokenurilor, dar nu de sinonime.")
    print("Ambele perechi au distanta 2, deci aceasta masura nu prefera S1-S2.")
    print("WordNet identifica sinonimele, dar singur nu stabileste rolurile in propozitie.")
    print("Nici un bag-of-words de synset-uri nu ar pastra aceste roluri.")

    print("\nUn exemplu conceptual de reprezentare (subiect, actiune, obiect):")
    print("S1: (medic, examineaza, copil)")
    print("S2: (medic, examineaza, copil)")
    print("S3: (copil, examineaza, medic)")
    print("Combinarea sensurilor cu rolurile surprinde corect diferenta.")


if __name__ == "__main__":
    sentences = [
        "The doctor examined the child.",
        "The physician examined the kid.",
        "The child examined the doctor.",
    ]

    tokens, word_sets, vectors = task_1_a(sentences)
    task_1_b(word_sets, vectors)
    task_1_c(tokens)
    task_1_d()
    task_1_e()
