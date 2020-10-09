import numpy as np, pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_short
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import os


class LDA_Explainer:
    """An LDA wrapper for explaining a predictor's predictions.
    Currently supports only binary predictors.
    Optionally supports domain-ruled data (see methods API and demo).

    Parameters
    ----------
    num_topics : int
        Number of topics of the LDA model.
    
    Attributes
    ----------
    num_topics : int
        Number of LDA topics that was passed at initialization.
    lda : gensim.models.LdaModel
        The underlying LDA model.
        `None` before `fit()` is called.
    domain_names : list of str
        The domain names that were either given to `fit()` or defaulted to numbering.
        `None` before `fit()` is called in any case.
    domain_labels : list of int
        The domain names that were either given to `fit()` (or `None` if not given).
        `None` before `fit()` is called.
    model_confidence : list of float
        Model confidence that was passed to `fit()`.
        `None` before `fit()` is called.
    doc_topic_mx : 2-D numpy.ndarray of type float
        A `(num_documents, num_topics)` matrix that describe each document given to `fit()`
        as a mixture of topics.
        `None` before `fit()` is called.
    sep_topics : list of tuple triplets
        Seperating topics chosen for each domain and their scores.
        See documentation of `fit()` for further explanation.
        A list of `(domain_name, seperating_topic_number, topic_score)` triplets.
        Always contains the "All" domain (even when no domains are given).
        `None` before `fit()` is called.
    """


    def __init__(self, num_topics):
        # Input check
        try:
            assert float(num_topics) == int(num_topics) and num_topics > 0
            num_topics = int(num_topics)
        except (AssertionError, TypeError):
            raise ValueError('Argument "num_topics" should be a positive integer.')

        self.num_topics = num_topics
        self.lda = self.domain_names = self.domain_labels = self.model_confidence = self.doc_topic_mx = self.sep_topics = None


    def save(self, fname):
        """Saves the model to files with the prefix `fname`.
        
        Parameters
        ----------
        fname : str
            Optional directory + file names prefix.
            For example, if `fname = "./saved_models/explainer"`, multiple files in "./saved_models/"
            will be written with the prefix "explainer".

        Raises
        ------
        RuntimeError
            If `fit()` was not called for the model earlier.

        Notes
        -----
        9 files are saved.
        """

        if self.lda is None:
            raise RuntimeError('Tried saving model without fitting (nothing to save).')
        
        directory = os.path.split(fname)[0]
        if directory and not os.path.exists(directory):
            os.mkdir(directory)

        self.lda.save(fname + '.lda')
        self.doc_topic_mx.tofile(fname + '.doc_topic_mx')
        self.model_confidence.tofile(fname + '.model_confidence')
        if self.domain_labels is None:
            with open(fname + '.domain_labels', 'w') as f:
                pass  # Empty file
        else:
            self.domain_labels.tofile(fname + '.domain_labels')
        with open(fname + '.domain_names', 'w') as f:
            if self.domain_names is not None:  # Empty file otherwise
                f.write('\n'.join(self.domain_names))
        with open(fname + '.sep_topics', 'w') as f:
            for triplet in self.sep_topics:
                for item in triplet:
                    f.write(str(item) + '\n')


    @classmethod
    def load(cls, fname):
        """Loads a saved model from files with the prefix `fname`.
        
        Parameters
        ----------
        fname : str
            Optional directory + file names prefix.
            See `save()` for example.
        
        Returns
        -------
        LDA_explainer
            The loaded model.
        
        Raises
        ------
        FileNotFoundError
            If one or more of the files is not found.
        """

        obj = cls(1)  # Temporary
        obj.lda = LdaModel.load(fname + '.lda')
        obj.num_topics = obj.lda.num_topics
        obj.doc_topic_mx = np.fromfile(fname + '.doc_topic_mx').reshape(-1, obj.num_topics)
        obj.model_confidence = np.fromfile(fname + '.model_confidence')
        obj.domain_labels = np.fromfile(fname + '.domain_labels', dtype = np.int)
        if obj.domain_labels.size == 0:  # Empty file
            obj.domain_labels = None
        else:
            with open(fname + '.domain_names') as f:
                obj.domain_names = np.array(f.read().split('\n'))
        with open(fname + '.sep_topics') as f:
            obj.sep_topics = []
            while True:
                domain_name = f.readline().rstrip('\n')  # Including "All"
                if not domain_name:
                    break
                topic_no = int(f.readline().rstrip('\n'))
                score = float(f.readline().rstrip('\n'))
                obj.sep_topics.append((domain_name, topic_no, score))

        return obj


    def fit(self, texts, model_confidence, domain_labels = None, domain_names = None):
        r"""Fits the explaining LDA model and evaluate topics (possibly for each domain).
        The support for domains is completely optional and seems integral since the class
        was designed for a domain-ruled data.

        Parameters
        ----------
        texts : list of str
            List (or array-like) of the classified texts (as strings).
            This is used as input to the LDA model and maybe to the explained model as well.
        model_confidence : list of float
            List (or array-like) containing the confidence of the explained model that
            the text is classified positively (1), for each text.
        domain_labels : list of int, optional
            Optional list of the domain label for each entry.
            If given, should contain values in {0, 1, ..., num_domains}.
        domain_names : list of str, optional
            Optional list of the domain names, ignored if `domain_labels` is not given.
            Should be of length `numpy.max(domain_labels) + 1`.
            The name in index `i` corresponds to the domain `i` in `domain_labels`.
            If `domain_labels` is given and `domain_names` is not, simple numbering is used.
            Cannot contain `"All"`, as it is saved for all the domains.

        Returns
        -------
        LDA_explainer
            self

        Raises
        ------
        ValueError
            If any of the parameters given is not as specified.

        RuntimeError
            If model is already fit (e.g., if `fit()` is called twice or if a loaded model is fit).

        Notes
        -----
        Preprocesses the texts (lower casing, removing punctuations, stopwords and words of length <3) before fitting
        the LDA model, but input for `model` (if given) is passed as-is.
        
        If :math:`Z` is the group of all topics, the seperating topic (for each domain) is chosen by

        .. math:: z^{sep} = \arg\max_{z \in Z} \left| \sum_{i \in I_{test}} \hat{y}_i \theta_z^i \right|

        Where :math:`\hat{y}_i` is the prediction of the explained model for the :math:`i^{th}` document
        (1 for positive class and -1 for negative class) and :math:`\theta_z^i` is the probability of
        topic :math:`z` in document :math:`i`.
        
        Note that the sign of the score (without absolute value) is saved.
        
        This definition induces symmetry between positive and negative classes.
        """
        if self.lda is not None:
            raise RuntimeError('Model is already fit. Please create a new object.')

        # Input Check
        try:
            assert not isinstance(texts, str)
            for text in texts:
                assert isinstance(text, str)
        except (AssertionError, TypeError):
            raise ValueError('Argument "texts" should be an array-like of strings')
        if model_confidence is not None:  # True after changes.
            try:
                assert not isinstance(model_confidence, str)
                model_confidence = np.array(model_confidence, dtype = np.float)
                assert model_confidence.size == len(texts)
            except (AssertionError, ValueError):
                raise ValueError(f'Argument "model_confidence" should be an array-like of floats with size {len(texts)}')
        if domain_labels is not None:
            try:
                assert not isinstance(domain_labels, str)
                domains = np.unique(domain_labels)
                assert (domains.astype(np.int) == domains).all() and (domains >= 0).all()
                domain_labels = np.array(domain_labels, dtype = np.int)
                assert domain_labels.size == len(texts)
            except AssertionError:
                raise ValueError(f'Argument "domain_labels" is not as required (see documentation).')
            if domain_names is None:
                domain_names = np.arange(1, domains.max() + 2)
            if isinstance(domain_names, str) or len(domain_names) != domains.max() + 1:
                raise ValueError(f'Argument "domain_names" should be either None or an array-like with size {np.unique(domain_labels).size}.')
            if 'All' in domain_names:
                raise ValueError('Argument "domain_names" cannot contain "All".')
            self.domain_names = np.array(domain_names, dtype = str)
            self.domain_labels = domain_labels

        self.model_confidence = model_confidence

        # Preprocess
        texts = [strip_short(remove_stopwords(strip_punctuation(text.lower()))).split(' ')
                 for text in texts]
        
        # Fit the LDA model
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        self.lda = LdaModel(corpus, num_topics = self.num_topics, id2word = dictionary)
        
        # Get topic probabilities of each document as a nice matrix
        self.doc_topic_mx = np.zeros((len(corpus), self.num_topics))
        for i, doc in enumerate(corpus):
            for topic, dominance in self.lda[doc]:
                self.doc_topic_mx[i, topic] = dominance

        # Rank the topics
        prediction = (model_confidence >= 0.5) * 2 - 1  # 1 or -1
        topic_scores_all = self.doc_topic_mx.T @ prediction  # Signed scores
        sep_topic_all = np.abs(topic_scores_all).argmax()
        self.sep_topics = [('All', sep_topic_all, topic_scores_all[sep_topic_all])]  # (domain, topic, score) triplets
        if self.domain_names is not None:
            for domain, domain_name in enumerate(self.domain_names):
                idxs = self.domain_labels == domain
                topic_scores = self.doc_topic_mx[idxs].T @ prediction[idxs]
                sep_topic = np.abs(topic_scores).argmax()
                self.sep_topics.append((domain_name, sep_topic, topic_scores[sep_topic]))
    
        return self
    
    
    def __show_topic(self, topic_id, topn):
        return [term for term, _ in self.lda.show_topic(topic_id, topn = topn)]


    def display_seperating_topics(self, topn = 15):
        """Presents the seperating topics (one topic if no domains are given) in a pandas DataFrame.
        
        Parameters
        ----------
        topn : int
            Number of words to present for each topic (default = 15).
        
        Returns
        -------
        pandas.DataFrame
            The presentive DataFrame. Each row corresponds to a domain (including "All").
            Each row consist of the domain name as the index, the topic number, its score
            and `topn` entries with the `topn` top words of the topic.
    
        Notes
        -----
        Topics are numbered from 1 to `num_topics`.
        
        See `fit` for explanation on the score and its sign.
        """

        if self.lda is None:
            raise RuntimeError('Call `fit` before `get_seperating_topics`.')
        
        df = pd.DataFrame(columns = ['#', 'score'] + list(range(1, topn + 1)))
        
        for domain_name, sep_topic, score in self.sep_topics:
            df.loc[domain_name, ['#', 'score']] = sep_topic + 1, score
            df.loc[domain_name].iloc[2:] = self.__show_topic(sep_topic, topn)

        return df


    def display_topics(self, topn = 15, colors = None):
        """Presents all topics in a pandas DataFrame.
        
        Parameters
        ----------
        topn : int
            Number of words to present for each topic (default = 15).
        colors : dict of (str, list), optional
            Instructions for coloring specific rows in specific colors.
            If `n` is in `colors[c]`, then the font in the row of topic `n` will be colored in `c`.


        Returns
        -------
        pandas.io.formats.style.Styler
            The presentive table. Each row corresponds to a topic.
            Each row consist of the topic name (number) as the index and `topn`
            entries with the `topn` top words of the topic.
            The table is captioned `f"Top {topn} Words"`.

        Notes
        -----
        Topics are numbered from 1 to `num_topics`.
        """

        df = pd.DataFrame([self.__show_topic(i, topn) for i in range(self.num_topics)],
                          index = pd.Index([f'topic #{i}' for i in np.arange(1, self.num_topics + 1)]),
                          columns = range(1, topn + 1)
                         )
        if colors is None:
            styler = df.style
        else:
            def highlight(row):
                for color, rows in colors.items():
                    if int(row.name.split('#')[-1]) in rows:
                        return [f'color: {color}'] * len(row)
                return [''] * len(row)
            styler = df.style.apply(highlight, axis = 1)
        styler.set_caption(f'<h3>Top {topn} Words</h3>')

        return styler


    def plot_topic_confidence_trends(self, save_fname = None, colors = None):
        r"""Plots topic-confidence trend line for all documents and for each domain.
        
        Parameters
        ----------
        save_fname : str, optional
            The file name and path for the saved figure (in PNG format).
            If not specified, the figure will not be saved.
        colors : OrderedDict of (str, str) or list of str, optional
            A `(color_name : color)` dictionary or just colors list.
            These colors determine the trend line colors of the domains.
            The "All" trend line is always black.
            If there are more domains than colors, colors are reused rotationally.
            (default = matplotlib.colors.TABLEAU_COLORS)

        Returns
        -------
        matplotlib.figure.Figure
            The figure in which the trend lines are plotted.

        Notes
        -----
        The figure is generated as in [1]_: If :math:`\theta^i_z` is the probability
        of topic :math:`z` in document :math:`i`, then for each
        :math:`j \in J = \{0.1, 0.2, ..., 1\}` we take the average
        confidence of the explained model over
        :math:`I^j := \{i|\theta^i \in (j-0.1, j]\}`, i.e.,

        .. math:: f(\theta; j) = \frac{\sum_{i \in I^j} \hat{p}_i}{|I^j|}

        Where :math:`\hat{p}_i` is the confidence of the model that document :math:`i`
        belongs to the positive class.
        
        If there are more than 10 domains and no color scheme is specified,
        colors will be reused for trend lines.


        References
        ----------
        .. [1] Oved, N., Feder, A. and Reichart, R., 2020.
           Predicting In-Game Actions from Interviews of NBA Players.
           Computational Linguistics, pp.1-46.
        """

        if colors is None:
            colors = TABLEAU_COLORS
        if isinstance(colors, dict):
            colors = list(colors.values())

        def generate_trend(confidence, topic_probs, interval = 0.1, max_prob = 1):
            conf_ubs = np.arange(interval, max_prob + interval, interval)  # Upper bounds
            mean_confs = []
            for conf_ub in conf_ubs:
                idxs = (conf_ub - interval < topic_probs) & (topic_probs <= conf_ub)
                mean_confs.append(confidence[idxs].mean())

            return conf_ubs, np.array(mean_confs)

        fig = plt.figure()
        for domain, (domain_name, sep_topic, _) in enumerate(self.sep_topics, start = -1):  # "All" is first
            if domain == -1:
                assert domain_name == 'All'
                x, y = generate_trend(self.model_confidence, self.doc_topic_mx[:, sep_topic])
                plt.plot(x, y, color = 'black', label = domain_name)
                plt.xticks(x)
            else:
                idxs = self.domain_labels == domain
                if not idxs.any():
                    continue
                plt.plot(*generate_trend(self.model_confidence[idxs], self.doc_topic_mx[idxs, sep_topic]),
                         color = colors[domain % len(colors)], label = domain_name)
        plt.legend()
        plt.xlabel('Seperating Topic Probability')
        plt.ylabel('Positive Class Confidence')
        plt.title('Topic-Confidence Trends')
        if save_fname is not None:
            plt.savefig(save_fname, format = 'png')

        return fig


    def plot_topics_dominance(self, save_fstr = None):
        """Plots the average dominance of each topic in each domain (and in all domains).
        Draws a seperate lolipop chart for each domain and one for all domains.
        
        Parameters
        ----------
        save_fstr : str, optional
            A formatable string for saving the figures. Must contain "%s".
            E.g., "topic_dominance_%s.png".
            Domain names will replace "%s".
        
        Returns
        -------
        list of matplotlib.figure.Figure
            List of the drawn figures.
        
        Notes
        -----
        Width of the figures is `6 * num_topics / 20` inches, height is 4 inches.
        
        ".png" suffix is added to `save_fstr` if missing.
        """

        if save_fstr is not None:
            if '%s' not in save_fstr:
                raise ValueError('Argument "save_fstr" must contain the substring "%s".')
            if not save_fstr.lower().endswith('.png'):
                save_fstr += '.png'

        figures = []
        x = np.arange(self.num_topics) + 1
        for domain, (domain_name, _, _) in enumerate(self.sep_topics, start = -1):
            fig = plt.figure(figsize = (6 * self.num_topics / 20, 4))
            if domain_name == 'All':
                y = self.doc_topic_mx.mean(axis = 0)
            else:
                y = self.doc_topic_mx[self.domain_labels == domain].mean(axis = 0)
            plt.vlines(x, 0, y, colors = 'skyblue')
            plt.plot(x, y, 'o')
            plt.xlabel('topic')
            plt.xticks(x)
            plt.ylabel(f'average probability in the domain')
            plt.title(f'Topic Dominance in Domain "{domain_name}"')
            if save_fstr is not None:
                plt.savefig(save_fstr % domain_name)
            figures.append(fig)

        return figures
