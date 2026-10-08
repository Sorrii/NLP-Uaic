from task_2.common import get_related_words


def task_2_a():
    parts_of_speech = {
        "n": "substantiv",
        "v": "verb",
        "a": "adjectiv",
        "s": "adjectiv",
        "r": "adverb",
    }

    print("\nTASK 2 a) Explorarea relatiilor WordNet")
    print("Pentru fiecare sens afisam definitia si cuvintele asociate.")
    print("Scrie 0 pentru iesire.")

    while True:
        try:
            word = input("\nIntrodu un cuvant in engleza: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram incheiat.")
            break

        if word == "0":
            print("Program incheiat.")
            break

        if not word:
            print("Nu ai introdus un cuvant.")
            continue

        try:
            results = get_related_words(word)
        except LookupError:
            print("Lipsesc datele WordNet. Ruleaza:")
            print("python -m nltk.downloader wordnet")
            break

        if not results:
            print("Cuvantul nu a fost gasit in WordNet.")
            continue

        for index, result in enumerate(results, start=1):
            part_of_speech = parts_of_speech[result["part_of_speech"]]
            print(f"\nSens {index} ({part_of_speech})")
            print("Definitie:", result["definition"])

            for relation, words in result["relations"].items():
                text = ", ".join(sorted(words)) if words else "Nu au fost gasite."
                print(f"{relation}: {text}")


if __name__ == "__main__":
    task_2_a()