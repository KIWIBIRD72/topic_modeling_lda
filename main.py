import pandas as pd
from janome.tokenizer import Tokenizer
from gensim import corpora
import gensim
import multiprocessing
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import webbrowser
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_stopwords(stopwords_file_path: str) -> list:
    with open(stopwords_file_path, "r") as stopwords_file:
        stop_words = [word.strip() for word in stopwords_file.readlines()]
        return stop_words


def tokenise(text: str, tokenizer: Tokenizer, stopwords: list[str]) -> list[str]:
    """
    Сегметация японского текста. Удаление стоп слов и знаков пунктуации
    :param text: неформатированный текст на японском
    :param tokenizer: токенизатор
    :param stopwords: список стоп слов
    :return: лематизированный и разбитый на сегметы текст
    """
    return [
        token.surface
        for token in tokenizer.tokenize(text)
        if len(token.surface) > 1 and token.surface not in stopwords  # Убираем стоп-слова
    ]


def create_dictionary(documents: list, no_above: float, no_below: int) -> dict:
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_above=no_above, no_below=no_below)
    dictionary.compactify()

    return dictionary


def create_corpus(dictionary: corpora.Dictionary, preprocessed_docs: list[str]) -> list[str]:
    return [dictionary.doc2bow(text) for text in preprocessed_docs]


def print_topics(lda_topics: gensim.models.LdaMulticore):
    print('Topics:')
    for topic in lda_topics.print_topics():
        print(f'Topic {topic[0]}:', str(topic))


def coherence_score(dictionary: corpora.Dictionary, corpus: list[str], texts: list[str], max: int, start=2, step=3,
                    measure ="c_uci", ax=None):
    """
    Функция вычисляет метрики для оценки тем и рисует график на переданных осях.
    :param dictionary: словарь для тематического моделирования
    :param corpus: корпус в виде мешка слов
    :param texts: тексты документов
    :param max: максимальное количество топиков
    :param start: стартовое количество топиков
    :param step: промежуток, с которым вычисляются топики
    :param measure: метрика 'u_mass'| 'c_v'| 'c_uci'| 'c_npmi'
    :param ax: ось, на которой будет строиться график
    """
    coherence_values = []
    for num_topics in range(start, max, step):
        model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, passes=10, num_topics=num_topics,
                                           random_state=6457)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                                                      coherence=measure)
        coherence_values.append(coherencemodel.get_coherence())

    x = range(start, max, step)
    ax.plot(x, coherence_values)
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel(measure + " score")
    ax.legend([measure], loc='best')


def plot_multiple_graphs(dictionary, corpus, texts, start=2, max=15, step=3, title:str=''):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, measure in enumerate(['u_mass', 'c_v', 'c_uci', 'c_npmi']):
        coherence_score(dictionary=dictionary, corpus=corpus, texts=texts, start=start, max=max, step=step,
                        measure=measure, ax=axes.flatten()[i])

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def gensim_visualise(lda_topics: gensim.models.LdaMulticore, corpus: list[str], dictionary: list[str]):
    print("Gensim prepare visualisation...", end='\n\n')
    vis_lda = gensimvis.prepare(lda_topics, corpus, dictionary)
    pyLDAvis.save_html(vis_lda, "./temp/lda_vis_10.html")
    webbrowser.open("file://" + os.path.abspath("./temp/lda_vis_10.html"))


if __name__ == "__main__":
    # Enable multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    multiprocessing.freeze_support()

    DATASET_PATH = './assets/datasets/japanese_train.csv'
    STOP_WORDS_JA_PATH = './assets/stopwords/ja.txt'

    stopwords = get_stopwords(STOP_WORDS_JA_PATH)
    df = pd.read_csv(DATASET_PATH)

    # Токенизация текста
    tokenizer = Tokenizer()
    dataset_ja = df[df['lang'] == 'ja']['text']
    preprocessed_docs = dataset_ja.apply(lambda doc: tokenise(doc, tokenizer, stopwords))
    print('Text preprocessed:\n', preprocessed_docs, end='\n\n')

    # Создание словаря
    NO_ABOVE=0.1
    NO_BELOW=20
    dictionary = create_dictionary(preprocessed_docs, no_above=NO_ABOVE, no_below=NO_BELOW)
    print('Dictionary:', dictionary, end='\n\n')
    corpus = create_corpus(dictionary, preprocessed_docs)

    # Coreherence modeling
    # plot_multiple_graphs(dictionary=dictionary, corpus=corpus, texts=preprocessed_docs, title=f"no_above: {NO_ABOVE} no_below: {NO_BELOW}")

    # Topic modeling
    lda_topics = gensim.models.LdaMulticore(
        corpus=corpus,
        num_topics=12,
        id2word=dictionary,
        passes=10,
        random_state=6457)

    gensim_visualise(
        lda_topics=lda_topics,
        corpus=corpus,
        dictionary=dictionary)
