import collections
import math

import numpy as np
import pandas as pd
from scipy import sparse

from river.base import tags

from . import base

__all__ = ["ComplementNB"]


class ComplementNB(base.BaseNB):
    """Naive Bayes classifier for multinomial models.

    Complement Naive Bayes model learns from occurrences between features such as word counting
    and discrete classes. ComplementNB is suitable for imbalance dataset.
    The input vector must contain positive values, such as counts or TF-IDF values.

    Parameters
    ----------
    alpha
        Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).

    Attributes
    ----------
    class_dist : proba.Multinomial
        Class prior probability distribution.
    feature_counts : collections.defaultdict
        Total frequencies per feature and class.
    class_totals : collections.Counter
        Total frequencies per class.

    Examples
    --------

    >>> from river import feature_extraction
    >>> from river import naive_bayes

    >>> sentences = [
    ...     ('food food meat brain', 'health'),
    ...     ('food meat ' + 'kitchen ' * 9 + 'job' * 5, 'butcher'),
    ...     ('food food meat job', 'health')
    ... ]

    >>> model = feature_extraction.BagOfWords() | ('nb', naive_bayes.ComplementNB)

    >>> for sentence, label in sentences:
    ...     model = model.learn_one(sentence, label)

    >>> model['nb'].p_class('health') == 2 / 3
    True
    >>> model['nb'].p_class('butcher') == 1 / 3
    True

    >>> model.predict_proba_one('food job meat')
    {'health': 0.9409689355477155, 'butcher': 0.05903106445228467}

    >>> import pandas as pd

    >>> docs = [
    ...     ('food food meat brain', 'health'),
    ...     ('food meat ' + 'kitchen ' * 9 + 'job' * 5, 'butcher'),
    ...     ('food food meat job', 'health')
    ... ]

    >>> docs = pd.DataFrame(docs, columns = ['X', 'y'])

    >>> X, y = docs['X'], docs['y']

    >>> model = feature_extraction.BagOfWords() | ('nb', naive_bayes.ComplementNB)

    >>> model = model.learn_many(X, y)

    >>> model['nb'].p_class('health') == 2 / 3
    True

    >>> model['nb'].p_class('butcher') == 1 / 3
    True

    >>> model['nb'].p_class_many()
        butcher    health
    0  0.333333  0.666667

    >>> model.predict_proba_one('food job meat')
    {'butcher': 0.05903106445228467, 'health': 0.9409689355477155}

    >>> model.predict_proba_one('Taiwanese Taipei')
    {'butcher': 0.3769230769230768, 'health': 0.6230769230769229}

    >>> unseen_data = pd.Series(
    ...    ['food job meat', 'Taiwanese Taipei'], name = 'X', index = ['river', 'rocks'])

    >>> model.predict_proba_many(unseen_data)
            butcher    health
    river  0.059031  0.940969
    rocks  0.376923  0.623077

    >>> model.predict_many(unseen_data)
    river    health
    rocks    health
    dtype: object

    References
    ----------
    [^1]: [Rennie, J.D., Shih, L., Teevan, J. and Karger, D.R., 2003. Tackling the poor assumptions of naive bayes text classifiers. In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 616-623)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
    [^2]: [StackExchange discussion](https://stats.stackexchange.com/questions/126009/complement-naive-bayes)

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)
        self.feature_totals = collections.Counter()
        self.class_totals = collections.Counter()

    def _more_tags(self):
        return {tags.POSITIVE_INPUT}

    def learn_one(self, x, y):
        """Updates the model with a single observation.

        Args:
            x: Dictionary of term frequencies.
            y: Target class.

        Returns:
            self

        """
        self.class_counts.update((y,))

        for f, frequency in x.items():
            self.feature_counts[f].update({y: frequency})
            self.feature_totals.update({f: frequency})
            self.class_totals.update({y: frequency})

        return self

    def p_class(self, c):
        return self.class_counts[c] / sum(self.class_counts.values())

    def p_class_many(self) -> pd.DataFrame:
        return base.from_dict(self.class_counts).T[self.class_counts] / sum(
            self.class_counts.values()
        )

    def joint_log_likelihood(self, x):
        """Computes the joint log likelihood of input features.

        Args:
            x: Dictionary of term frequencies.

        Returns:
            Mapping between classes and joint log likelihood.

        """
        cc = {
            c: {
                f: self.feature_totals[f] + self.alpha - frequency.get(c, 0)
                for f, frequency in self.feature_counts.items()
            }
            for c in self.class_counts
        }

        return {
            c: sum(
                {
                    f: frequency
                    * -math.log(cc[c].get(f, self.alpha) / sum(cc[c].values()))
                    for f, frequency in x.items()
                }.values()
            )
            for c in self.class_counts
        }

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        """Updates the model with a term-frequency or TF-IDF pandas dataframe.

        Args:
            X: Term-frequency or TF-IDF pandas dataframe.
            y: Target classes.

        """
        y = base.one_hot_encode(y)
        columns, classes = X.columns, y.columns
        y = sparse.csc_matrix(y.sparse.to_coo()).T

        self.class_counts.update(
            {c: count.item() for c, count in zip(classes, y.sum(axis=1))}
        )

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        fc = y @ X

        self.class_totals.update(
            {c: count.item() for c, count in zip(classes, fc.sum(axis=1))}
        )

        self.feature_totals.update(
            {
                c: count.item()
                for c, count in zip(columns, np.array(fc.sum(axis=0)).flatten())
            }
        )

        # Update feature counts by slicing the sparse matrix per column.
        # Each column correspond to a class.
        for c, i in zip(classes, range(fc.shape[0])):

            counts = {
                c: {columns[f]: count for f, count in zip(fc[i].indices, fc[i].data)}
            }

            # Transform {classe_i: {token_1: f_1, ... token_n: f_n}} into:
            # [{token_1: {classe_i: f_1}},.. {token_n: {class_i: f_n}}]
            for dict_count in [
                {token: {c: f} for token, f in frequencies.items()}
                for c, frequencies in counts.items()
            ]:

                for f, count in dict_count.items():
                    self.feature_counts[f].update(count)

        return self

    def _feature_log_prob(self, unknown: list, columns: list) -> pd.DataFrame:
        """Compute log probabilities of input features.

        Args:
            unknown: List of features that are not part the vocabulary.
            columns: List of input features.

        Returns:
            Log probabilities of input features.

        """
        cc = (
            base.from_dict(self.feature_totals).squeeze().T
            + self.alpha
            - base.from_dict(self.feature_counts).fillna(0).T
        )

        sum_cc = cc.sum(axis=1).values

        cc[unknown] = self.alpha

        return -np.log(cc[columns].T / sum_cc)

    def joint_log_likelihood_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Computes the joint log likelihood of input features.

        Args:
            X: Term-frequency or TF-IDF pandas dataframe.

        Returns:
            Input samples joint log likelihood.

        """
        index, columns = X.index, X.columns
        unknown = [x for x in columns if x not in self.feature_counts]

        if hasattr(X, "sparse"):
            X = sparse.csr_matrix(X.sparse.to_coo())

        return pd.DataFrame(
            X @ self._feature_log_prob(unknown=unknown, columns=columns),
            index=index,
            columns=self.class_counts.keys(),
        )
