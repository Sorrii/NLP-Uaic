"""English desktop interface for the NLP WordNet coursework."""

from math import sqrt
from random import shuffle

import customtkinter as ctk
from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance

from task_2.exercise_bc.exercise_bc import evaluate_answer, save_score

try:
    from task_2.common import get_related_words
except ModuleNotFoundError as error:
    if error.name != 'task_2.common':
        raise
    get_related_words = None

BG = '#0b1020'
PANEL = '#171f33'
PURPLE = '#8b7cf6'
TEXT = '#eaf0ff'
MUTED = '#aab7ce'
PARTS = {'n': 'Noun', 'v': 'Verb', 'a': 'Adjective', 's': 'Adjective', 'r': 'Adverb'}
RELATIONS = {'Sinonime': 'Synonyms', 'Antonime': 'Antonyms', 'Hiperonime': 'Hypernyms',
             'Hiponime': 'Hyponyms', 'Meronime': 'Meronyms'}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.title('LexiLab | WordNet Learning Studio')
        self.geometry('1080x720')
        self.minsize(820, 570)
        self.configure(fg_color=BG)
        self.player = ''
        self.score = 0
        self.history = {}
        self.used = set()
        self.target = None
        self.word = ''
        self.words = []
        self.game_active = False

        sidebar = ctk.CTkFrame(self, width=215, fg_color=PANEL, corner_radius=0)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        ctk.CTkLabel(sidebar, text='✦ LexiLab', text_color=TEXT,
                     font=ctk.CTkFont(size=25, weight='bold')).pack(pady=(36, 9))
        ctk.CTkLabel(sidebar, text='WORDNET STUDIO', text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(pady=(0, 35))
        for title, page in [
    ('⌂  Overview', 'home'),
    ('⌕  Word Explorer', 'explorer'),
    ('◈  Association Game', 'game')
]:
            ctk.CTkButton(sidebar, text=title, anchor='w', fg_color='transparent',
                          hover_color='#29334e', height=46,
                          command=lambda p=page: self.show(p)).pack(fill='x', padx=12, pady=4)
        ctk.CTkLabel(sidebar, text='Natural Language Processing',
                     text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side='bottom', pady=26)

        self.main = ctk.CTkScrollableFrame(self, fg_color=BG)
        self.main.pack(side='left', fill='both', expand=True, padx=28, pady=22)
        self.show('home')
        self.protocol('WM_DELETE_WINDOW', self.close)

    def clear(self):
        for widget in self.main.winfo_children():
            widget.destroy()

    def heading(self, title, subtitle):
        ctk.CTkLabel(self.main, text=title, anchor='w', text_color=TEXT,
                     font=ctk.CTkFont(size=31, weight='bold')).pack(fill='x', pady=(8, 6))
        ctk.CTkLabel(self.main, text=subtitle, anchor='w', text_color=MUTED,
                     font=ctk.CTkFont(size=14)).pack(fill='x', pady=(0, 24))

    def panel(self):
        frame = ctk.CTkFrame(self.main, fg_color=PANEL, corner_radius=16)
        frame.pack(fill='x', pady=8)
        return frame

    def note(self, parent, title, description):
        ctk.CTkLabel(parent, text=title, anchor='w', font=ctk.CTkFont(size=18, weight='bold'),
                     text_color=TEXT).pack(fill='x', padx=22, pady=(19, 5))
        ctk.CTkLabel(parent, text=description, anchor='w', justify='left', wraplength=640,
                     text_color=MUTED).pack(fill='x', padx=22, pady=(0, 19))

    def show(self, page):
        self.clear()
        {'home': self.home,  'explorer': self.explorer,
         'game': self.game}[page]()

    def home(self):
        self.heading('Learn words. Explore meaning.',
                     'A simple workspace for natural language processing with WordNet.')
        for title, desc, page in [
            
            ('01  Word Explorer', 'Discover definitions, synonyms and semantic relationships.', 'explorer'),
            ('02  Association Game', 'Think of related words, earn points and track your progress.', 'game'),
        ]:
            card = self.panel()
            self.note(card, title, desc)
            ctk.CTkButton(card, text='Open →', width=120, fg_color=PURPLE,
                          command=lambda p=page: self.show(p)).pack(anchor='w', padx=22, pady=(0, 20))

    def analysis(self):
        self.heading('Sentence Analysis', 'Task 1 · Bag-of-words, similarities and WordNet')
        sentences = ['The doctor examined the child.', 'The physician examined the kid.',
                     'The child examined the doctor.']
        tokens = [s.lower().replace('.', '').split() for s in sentences]
        sets = [set(x) for x in tokens]
        vocab = sorted(set.union(*sets))
        vectors = [[row.count(word) for word in vocab] for row in tokens]
        self.note(self.panel(), 'Input sentences', '\n'.join(f'S{i}: {s}' for i, s in enumerate(sentences, 1)))
        self.note(self.panel(), 'Bag-of-words vocabulary', ', '.join(vocab))
        for i in (1, 2):
            union = sets[0] | sets[i]
            jaccard = len(sets[0] & sets[i]) / len(union)
            dot = sum(a * b for a, b in zip(vectors[0], vectors[i]))
            norms = sqrt(sum(a * a for a in vectors[0]) * sum(b * b for b in vectors[i]))
            cosine = dot / norms
            distance = edit_distance(tokens[0], tokens[i], substitution_cost=1, transpositions=False)
            self.note(self.panel(), f'S1 vs S{i+1}',
                      f'Jaccard similarity: {jaccard:.4f}\nCosine similarity: {cosine:.4f}\n'
                      f'Token edit distance: {distance}\n'
                      f'Bag-of-words vectors: {vectors[0]} vs {vectors[i]}')
        try:
            pairs = [('doctor.n.01', 'physician.n.01'), ('child.n.01', 'kid.n.01')]
            for left, right in pairs:
                a, b = wordnet.synset(left), wordnet.synset(right)
                self.note(self.panel(), f'{left} / {right}',
                          f'{a.definition()}\n{b.definition()}\n'
                          f'Same synset: {a == b} · Path similarity: {a.path_similarity(b)}')
        except LookupError:
            self.note(self.panel(), 'WordNet data missing', 'Run: python -m nltk.downloader wordnet')
        self.note(self.panel(), 'Interpretation',
                  'S1 and S2 describe the same roles using synonyms. S3 reverses the roles. '
                  'Bag-of-words ignores order, while token edit distance accounts for order '
                  'but cannot recognize synonyms. WordNet captures meaning relations but not sentence roles.')

    def explorer(self):
        self.heading('Word Explorer', 'Task 2a · Discover the different meanings of a word.')
        if get_related_words is None:
            self.note(self.panel(), 'Missing source file',
                      'Add your original task_2/common.py file to enable WordNet Explorer and the game.')
            return
        controls = self.panel()
        self.search_entry = ctk.CTkEntry(controls, placeholder_text='Enter an English word...', height=44)
        self.search_entry.pack(side='left', fill='x', expand=True, padx=(17, 10), pady=17)
        ctk.CTkButton(controls, text='Search', fg_color=PURPLE, height=44,
                      command=self.search).pack(side='right', padx=(0, 17))
        self.search_entry.bind('<Return>', lambda _: self.search())
        self.results = ctk.CTkFrame(self.main, fg_color='transparent')
        self.results.pack(fill='x')

    def search(self):
        for widget in self.results.winfo_children():
            widget.destroy()
        word = self.search_entry.get().strip()
        if not word:
            ctk.CTkLabel(self.results, text='Please enter a word.', text_color=MUTED).pack()
            return
        try:
            results = get_related_words(word)
        except LookupError:
            results = []
        if not results:
            ctk.CTkLabel(self.results, text='No WordNet entries found (or WordNet data is missing).',
                         text_color=MUTED).pack(pady=20)
            return
        for index, item in enumerate(results, 1):
            card = ctk.CTkFrame(self.results, fg_color=PANEL, corner_radius=14)
            card.pack(fill='x', pady=8)
            title = f'Sense {index}  ·  {PARTS.get(item["part_of_speech"], "Unknown")}'
            lines = [item['definition']]
            for relation, values in item['relations'].items():
                words = ', '.join(sorted(values)) if values else 'None'
                lines.append(f'{RELATIONS.get(relation, relation)}: {words}')
            self.note(card, title, '\n\n'.join(lines))

    def game(self):
        self.heading('Word Association Game', 'Tasks 2b & 2c · Find related words and collect points.')
        if get_related_words is None:
            self.note(self.panel(), 'Missing source file',
                      'Add your original task_2/common.py file to enable this game.')
            return
        if not self.game_active:
            intro = self.panel()
            self.note(intro, 'Start a new session',
                      'Earn round(100 × path similarity) plus 20 points for a direct relationship.')
            self.name_entry = ctk.CTkEntry(intro, placeholder_text='Player name', height=42)
            self.name_entry.pack(fill='x', padx=22, pady=8)
            ctk.CTkButton(intro, text='Start Game', fg_color=PURPLE,
                          command=self.start_game).pack(anchor='w', padx=22, pady=(5, 22))
            return
        self.score_label = ctk.CTkLabel(self.main, text=f'{self.player}   ·   {self.score} points',
                                        text_color=PURPLE, font=ctk.CTkFont(size=20, weight='bold'))
        self.score_label.pack(anchor='w', pady=(0, 15))
        self.target_frame = self.panel()
        self.answer_entry = ctk.CTkEntry(self.main, placeholder_text='Type an associated English word...', height=46)
        self.answer_entry.pack(fill='x', pady=12)
        self.answer_entry.bind('<Return>', lambda _: self.submit_answer())
        buttons = ctk.CTkFrame(self.main, fg_color='transparent')
        buttons.pack(fill='x')
        ctk.CTkButton(buttons, text='Submit Answer', fg_color=PURPLE,
                      command=self.submit_answer).pack(side='left', padx=(0, 9))
        ctk.CTkButton(buttons, text='Skip Word', fg_color='#34405b',
                      command=self.next_word).pack(side='left', padx=9)
        ctk.CTkButton(buttons, text='End Game', fg_color='#713d5d',
                      command=self.end_game).pack(side='right')
        self.feedback = ctk.CTkTextbox(self.main, fg_color=PANEL, height=200, wrap='word',
                                       font=ctk.CTkFont(size=13))
        self.feedback.pack(fill='x', pady=20)
        if self.target is None:
            self.next_word()
        else:
            self.render_target()

    def start_game(self):
        player = self.name_entry.get().strip()
        if not player:
            return
        self.player, self.score, self.history = player, 0, {}
        self.words, self.target, self.game_active = [], None, True
        self.show('game')

    def render_target(self):
        for widget in self.target_frame.winfo_children():
            widget.destroy()
        self.note(self.target_frame, self.word.title(),
                  f'{PARTS.get(self.target["part_of_speech"], "Unknown")}\n{self.target["definition"]}')

    def next_word(self):
        try:
            if not self.words:
                self.words = list(wordnet.all_lemma_names())
                shuffle(self.words)
            while self.words:
                word = self.words.pop().replace('_', ' ')
                results = get_related_words(word)
                if results:
                    self.word, self.target = word, results[0]
                    self.used = set()
                    self.history.setdefault(word, [])
                    self.render_target()
                    self.feedback.delete('1.0', 'end')
                    self.answer_entry.delete(0, 'end')
                    return
            self.feedback.insert('end', 'No more words available.')
        except LookupError:
            self.feedback.insert('end', 'WordNet data missing. Run: python -m nltk.downloader wordnet')

    def submit_answer(self):
        if self.target is None:
            return
        answer = ' '.join(self.answer_entry.get().lower().split())
        self.answer_entry.delete(0, 'end')
        if not answer:
            return
        if answer not in self.history[self.word]:
            self.history[self.word].append(answer)
        try:
            result = evaluate_answer(self.target, answer)
        except LookupError:
            self.feedback.insert('end', 'WordNet data missing.\n')
            return
        if result is None:
            message = 'No matching WordNet sense for this part of speech.'
        elif result['answer'].replace('_', ' ') == self.word:
            message = 'The target word does not earn points.'
        elif result['answer'] in self.used:
            message = 'You already used that answer for this word.'
        else:
            self.used.add(result['answer'])
            self.score += result['points']
            self.score_label.configure(text=f'{self.player}   ·   {self.score} points')
            relation_names = [RELATIONS.get(name, name) for name in result['relations']]
            similarity = ('Unavailable' if result['similarity'] is None
                          else f'{result["similarity"]:.4f}')
            message = (f'{answer}: +{result["points"]} points '
                       f'({result["similarity_points"]} similarity + {result["bonus"]} bonus)\n'
                       f'Path similarity: {similarity}\n'
                       f'Relations: {", ".join(relation_names) if relation_names else "No direct relation"}\n'
                       f'Meaning: {result["definition"]}')
        self.feedback.insert('end', message + '\n\n')
        self.feedback.see('end')

    def end_game(self):
        if self.game_active:
            try:
                save_score(self.player, self.score, self.history)
            except OSError as error:
                self.feedback.insert('end', f'Unable to save scores: {error}\n')
                return
            self.game_active = False
            self.target = None
            self.show('home')

    def close(self):
        if self.game_active:
            try:
                save_score(self.player, self.score, self.history)
            except OSError:
                pass
            self.game_active = False
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
