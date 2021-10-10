# Machine Learning Safari

[![Language](https://img.shields.io/badge/language-Python_(3.8%2B)-orange.svg?style=for-the-badge)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-informational?style=for-the-badge)](https://www.python.org/dev/peps/pep-0008/)

[![Read the Docs](https://readthedocs.org/projects/machine-learning-safari/badge/)](https://machine-learning-safari.readthedocs.io/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/THargreaves/machine-learning-safari/Tests?logo=github&style=for-the-badge)](https://github.com/THargreaves/machine-learning-safari/actions?workflow=Tests)
[![Codecov](https://img.shields.io/codecov/c/github/THargreaves/machine-learning-safari?logo=codecov&style=for-the-badge)](https://codecov.io/gh/THargreaves/machine-learning-safari)

## Overview

![Machine Learning Safari Logo](https://user-images.githubusercontent.com/38204689/132847544-bb36bd1c-7390-4351-b694-269e873cd16c.png)

Machine Learning Safari is an esoteric Python comprised of from-scratch implementations of popular machine learning algorithms.

The goal of the package is to provide efficient and easy to understand implementations of popular machine learning algorithms, whilst maintaining a simple package structure. This makes it easier for beginners to comprehend the purpose of different elements of the package, compared to, say, [scikit-learn](https://github.com/scikit-learn/scikit-learn) which can be harder to navigate.

The focus of this package is largely on the following types of machine learning algorithms:

- Regression and Classification
- Dimensionality Reduction
- Clustering

We hope that this package can be used for both the practical application of machine learning as well as a demonstration of the implementation of such methods for educational purposes.

## Package Design

The package is composed of nested (sub-)modules, each corresponding to a particular machine learning algorithm implemented as a class. An instance of such a class is called a model and (loosely inspired, by [scikit-learn](https://scikit-learn.org/stable/)) has `fit` and `apply` methods. On top of this, each class has an `inspect` method which can be used to display information about the models internal state. Note, that we use the method `apply` in both supervised and unsupervised settings, rather than including separate `predict`/`transform` methods.

For supervised models, rather than having separate classifiers and regressors, we bundle up functionality into one class and provide an `objective` parameter which can be either `regression` or `classification`.

For example, we can fit and apply the null model for classification as so.

```python
import numpy as np
import mlsafari as mls

X_train = np.empty((4, 2))
y_train = np.array([1, 2, 3, 3])
X_test = np.empty((3, 2))

mod = mls.NullModel(objective='classification')
mod.fit(None, np.array())
mod.apply(X_test)
#> array([3, 3, 3])
```

The package is developed using the methods discussed in [Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/). Most notably, this includes using [Poetry](https://python-poetry.org) for packaging and dependcy management.

## Contributing

Contributions to the package are welcome. Before writing code, we suggest opening an issue detailing the algorithm you wish to implement or selecting an already open issue.

Please ensure that all contributions are documented, have full coverage with unit tests, and follow [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) code style.
