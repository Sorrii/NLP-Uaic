from pathlib import Path
from random import shuffle

from nltk.corpus import wordnet

from task_2.common import get_related_words


def evaluate_answer(target, answer):
    answer = "_".join(answer.lower().split())
    part_of_speech = target["part_of_speech"]

    if part_of_speech == "s":
        part_of_speech = "a"

    answer = wordnet.morphy(answer, part_of_speech) or answer
    target_synset = wordnet.synset(target["synset"])
    best_result = None

    for synset in wordnet.synsets(answer, pos=part_of_speech):
        similarity = target_synset.path_similarity(synset, simulate_root=False)
        relations = []

        for relation, pairs in target["relation_pairs"].items():
            if (synset.name(), answer) in pairs:
                relations.append(relation)

        similarity_points = round(100 * similarity) if similarity is not None else 0
        bonus = 20 if relations else 0

        result = {
            "answer": answer,
            "synset": synset.name(),
            "definition": synset.definition(),
            "similarity": similarity,
            "relations": relations,
            "similarity_points": similarity_points,
            "bonus": bonus,
            "points": similarity_points + bonus,
        }

        if best_result is None or result["points"] > best_result["points"]:
            best_result = result

    return best_result


def show_result(result, total_score):
    explanations = {
        "Sinonime": "Acelasi sens.",
        "Antonime": "Sens opus, dar asociere valida. Nu inseamna sinonim.",
        "Hiperonime": "O categorie mai generala.",
        "Hiponime": "Un tip mai specific.",
        "Meronime": "O parte, un membru sau un material component.",
    }

    print("\nSensul raspunsului:", result["definition"])
    similarity = result["similarity"]

    if similarity is None:
        print("Similaritate path: indisponibila intre aceste sensuri.")
    else:
        print(f"Similaritate path: {similarity:.4f}")

        if similarity >= 0.5:
            print("Apropiere mare in ierarhia WordNet.")
        elif similarity >= 0.2:
            print("Apropiere moderata in ierarhia WordNet.")
        else:
            print("Apropiere mica in ierarhia WordNet.")

    if result["relations"]:
        for relation in result["relations"]:
            print(f"{relation}: {explanations[relation]}")
    else:
        print("Nu am identificat o relatie directa dintre cele verificate.")

    print(
        f"Puncte: {result['similarity_points']} + bonus {result['bonus']} "
        f"= {result['points']}"
    )
    print(f"Scor total: {total_score}")


def save_score(player, total_score, history):
    lines = [
        f"Jucator: {player}",
        f"Scor final: {total_score}",
        "Cuvinte propuse -> raspunsuri:",
    ]

    for word, answers in history.items():
        text = ", ".join(answers) if answers else "(fara raspuns)"
        lines.append(f"{word} -> {text}")

    scores_file = Path(__file__).with_name("scores.txt")

    with scores_file.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n\n")


def task_2_bc():
    words = []
    parts_of_speech = {
        "n": "substantiv",
        "v": "verb",
        "a": "adjectiv",
        "s": "adjectiv",
        "r": "adverb",
    }

    player = ""
    total_score = 0
    history = {}
    playing = True

    print("\nTASK 2 b) si c) Joc de asociere WordNet")
    print("Introdu cat mai multe asocieri in engleza pentru cuvantul propus.")
    print("0 = cuvant nou. /stop = oprirea jocului.")
    print("Punctaj: round(100 * similaritate) + 20 pentru o relatie directa.")
    print("Cuvantul propus si raspunsurile repetate nu se puncteaza.")
    print("La oprire salvam un rezumat in scores.txt.")

    try:
        while not player:
            player = input("\nNumele jucatorului (/stop = oprire): ").strip()

            if player.lower() == "/stop":
                print("Joc oprit.")
                return

            if not player:
                print("Introdu un nume.")

        while playing:
            if not words:
                print("Incarcam cuvintele din WordNet...")
                words = list(wordnet.all_lemma_names())
                shuffle(words)

                if not words:
                    print("Nu am gasit cuvinte in WordNet.")
                    break

            word = words.pop().replace("_", " ")
            results = get_related_words(word)

            if not results:
                print(f"Nu am gasit sensuri pentru {word}.")
                break

            target = results[0]
            used_answers = set()
            part_of_speech = parts_of_speech[target["part_of_speech"]]
            history.setdefault(word, [])

            print(f"\nCuvant: {word} ({part_of_speech})")
            print("Definitie:", target["definition"])

            while True:
                answer = input("\nRaspuns (0 = alt cuvant, /stop = oprire): ").strip()

                if answer.lower() == "/stop":
                    playing = False
                    break

                if answer == "0":
                    break

                if not answer:
                    print("Nu ai introdus un cuvant.")
                    continue

                answer = " ".join(answer.lower().split())

                if answer not in history[word]:
                    history[word].append(answer)

                result = evaluate_answer(target, answer)

                if result is None:
                    print(f"Nu am gasit un sens de tip {part_of_speech}.")
                    continue

                if result["answer"].replace("_", " ") == word:
                    print("Cuvantul propus nu se puncteaza.")
                    continue

                if result["answer"] in used_answers:
                    print("Ai folosit deja acest raspuns pentru cuvantul curent.")
                    continue

                used_answers.add(result["answer"])
                total_score += result["points"]
                show_result(result, total_score)

    except (EOFError, KeyboardInterrupt):
        print("\nJoc oprit.")
    except LookupError:
        print("Lipsesc datele WordNet. Ruleaza:")
        print("python -m nltk.downloader wordnet")

    if player:
        print(f"\nJucator: {player}")
        print(f"Scor final: {total_score}")

        try:
            save_score(player, total_score, history)
            print("Rezumat salvat in scores.txt.")
        except OSError as error:
            print(f"Nu am putut salva in scores.txt: {error}")


if __name__ == "__main__":
    task_2_bc()
