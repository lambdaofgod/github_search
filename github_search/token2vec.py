import gensim
import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from sklearn import feature_extraction

from github_search import neptune_util

torch.manual_seed(1)


def train_token2vec(upstream, product, embedding_dim, n_iterations, n_positive_imports):
    model_path = str(product["model_path"])

    def get_module_import_corpus(import_corpus_path):
        return pd.read_csv(import_corpus_path)["imports"].dropna()

    # %%
    import_corpus_path = str(upstream["prepare_module_corpus"])
    import_corpus = get_module_import_corpus(import_corpus_path)

    # %%

    class TokenDataset:
        def __init__(
            self, import_corpus, min_import_frequency=5, min_imports=2, use_cuda=True
        ):
            filtered_import_corpus = self._filter_corpus(import_corpus, min_imports)
            self._vectorizer = feature_extraction.text.CountVectorizer(
                min_df=min_import_frequency, binary=True
            )
            occurrence_matrix = self._vectorizer.fit_transform(filtered_import_corpus)
            n_imports = np.array((occurrence_matrix.sum(axis=1) > 1)).reshape(-1)
            valid_indices = np.where(n_imports)[0]
            occurrence_matrix = occurrence_matrix[valid_indices, :]
            self._occurrence_matrix = occurrence_matrix
            self.corpus_size = occurrence_matrix.shape[0]
            self.vocabulary_size = occurrence_matrix.shape[1]
            self.use_cuda = use_cuda

        def sample_imports(self, n_positive_imports, n_negative_imports="same"):
            if n_negative_imports == "same":
                n_negative_imports = n_positive_imports

            (
                positive_import_contexts,
                positive_import_targets,
            ) = self._sample_positive_or_negative_imports(
                n_positive_imports, positive=True
            )
            (
                negative_import_contexts,
                negative_import_targets,
            ) = self._sample_positive_or_negative_imports(
                n_negative_imports, positive=False
            )
            positive_import_contexts, positive_import_targets = torch.tensor(
                positive_import_contexts
            ), torch.tensor(positive_import_targets)
            negative_import_contexts, negative_import_targets = torch.tensor(
                negative_import_contexts
            ), torch.tensor(negative_import_targets)
            predictions = torch.cat(
                (torch.ones(n_positive_imports), torch.zeros(n_positive_imports)),
                axis=0,
            )
            contexts = torch.cat(
                (positive_import_contexts, negative_import_contexts), axis=0
            )
            targets = torch.cat(
                (positive_import_targets, negative_import_targets), axis=0
            )
            if self.use_cuda:
                return contexts.cuda(), targets.cuda(), predictions.cuda()
            else:
                return contexts, targets, predictions

        def _sample_positive_or_negative_imports(self, n_imports, positive):
            file_indices_sample = np.random.choice(
                range(self.corpus_size), size=n_imports
            )
            context_indices = []
            target_indices = []
            for idx in file_indices_sample:
                sample_row = np.array(self._occurrence_matrix[idx].todense())[0]
                import_indices = np.where(sample_row)[0]
                context_index = np.random.choice(import_indices, size=1)[0]
                if positive:
                    sample_row[context_index] = 0
                    possible_target_indices = np.where(sample_row)[0]
                else:
                    possible_target_indices = np.where(sample_row == 0)[0]
                target_index = np.random.choice(possible_target_indices)
                context_indices.append(context_index)
                target_indices.append(target_index)
            return context_indices, target_indices

        def _filter_corpus(self, import_corpus, min_imports):
            return import_corpus[
                import_corpus.str.split().apply(set).apply(len) >= min_imports
            ]

    # %%

    class Token2Vec(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(Token2Vec, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        def forward(self, context, target):
            context_embeddings = self.embeddings(context)
            target_embeddings = self.embeddings(target)
            similarities = (context_embeddings * target_embeddings).sum(axis=1)
            return similarities

    # %%

    do_nothing_callback = lambda *arg: None

    class Token2VecModeler:
        def __init__(self, token_dataset, embedding_size, use_cuda=True):
            self.token_dataset = token_dataset
            self.token2vec = Token2Vec(token_dataset.vocabulary_size, embedding_size)
            if use_cuda:
                self.token2vec.cuda()

        def fit(
            self,
            n_positive_imports,
            n_iterations,
            lr=0.001,
            loss_function=nn.BCEWithLogitsLoss(),
            evaluation_period=100,
            optimizer_cls=optim.Adam,
            training_loss_callback=do_nothing_callback,
            valid_loss_callback=do_nothing_callback,
            training_end_callback=do_nothing_callback,
        ):
            losses = []
            eval_losses = []
            self.loss_function = nn.BCEWithLogitsLoss()
            optimizer = optimizer_cls(self.token2vec.parameters(), lr=0.001)
            self.optimizer = optimizer

            for iteration in tqdm.tqdm(range(n_iterations)):
                training_loss = self._get_sample_loss(
                    n_positive_imports, training=True, optimizer=optimizer
                )
                training_loss_callback(training_loss)
                losses.append(training_loss)
                if iteration % evaluation_period == 0:
                    valid_loss = self._get_sample_loss(
                        n_positive_imports, training=False
                    )
                    eval_losses.append(valid_loss)
                    valid_loss_callback(valid_loss)

            history = {"train_loss": losses, "eval_loss": eval_losses}
            training_end_callback(self)
            return history

        def _get_sample_loss(self, n_positive_imports, training, optimizer=None):
            (context, target, pred) = self.token_dataset.sample_imports(
                n_positive_imports=n_positive_imports
            )
            self.token2vec.zero_grad()
            log_probs = self.token2vec(context, target)

            loss = self.loss_function(log_probs, pred)
            total_loss = 0
            total_loss += loss.item()
            if training:
                loss.backward()
                optimizer.step()
            return total_loss

        def make_keyed_vectors(self):
            vocab = self.token_dataset._vectorizer.vocabulary_
            word_vectors = self.token2vec.embeddings.weight.cpu().detach().numpy()
            kv = gensim.models.KeyedVectors(word_vectors.shape[1])
            words_list = list(vocab.keys())
            vectors = []
            for word in words_list:
                vectors.append(word_vectors[vocab[word]])
            kv.add(words_list, vectors)
            return kv

    # %%
    import_dataset = TokenDataset(import_corpus)
    token2vec_modeler = Token2VecModeler(import_dataset, embedding_dim)
    sampled_files_ratio = n_positive_imports * n_iterations / import_dataset.corpus_size
    sampled_tokens_ratio = (
        n_positive_imports * n_iterations / import_dataset.vocabulary_size
    )

    # %%
    sampled_files_ratio, sampled_tokens_ratio

    # %%
    hyperparameters = dict(
        min_import_frequency=5,
        n_iterations=n_iterations,
        n_positive_imports=n_positive_imports,
        embedding_dim=embedding_dim,
        lr=0.001,
    )

    # %%

    def run_training_experiment(hyperparameters):
        def training_end_callback(model):
            model.make_keyed_vectors().save(model_path)

        history = token2vec_modeler.fit(
            n_positive_imports,
            n_iterations,
            lr=hyperparameters["lr"],
            training_end_callback=training_end_callback,
        )
        return history

    # %%
    print(hyperparameters)
    history = run_training_experiment(hyperparameters)

    # %%
    plt.plot(history["train_loss"])
    plt.show()

    # %%
    plt.plot(pd.Series(history["train_loss"]).rolling(10).mean())
    plt.show()

    # %%
    plt.plot(history["eval_loss"])
    plt.show()
