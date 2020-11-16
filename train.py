import argparse
from itertools import islice

from torch import nn
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers import evaluation
from sentence_transformers import models


def load_train_data(dataset):
    if dataset == 'msrp':
        return load_msrp_train()
    elif dataset == 'sts':
        return load_sts_train()
    else:
        return load_chbank_train()


def load_chbank_train():
    train_exmples = []
    bank_file = 'datasets/chinese/bank_train.txt'
    positive_count = 0
    negative_count = 0
    sentences1 = []
    sentences2 = []
    scores = []
    with open(bank_file) as file:
        for line in islice(file, 0, None):
            content = line.split('\t')
            label = int(content[2])
            sent_1 = content[0]
            sent_2 = content[1]
            sentences1.append(sent_1)
            sentences2.append(sent_2)
            scores.append(label)
            examples = InputExample(texts=[sent_1, sent_2], label=label)
            train_exmples.append(examples)
            if int(label) == 0:
                negative_count += 1
            else:
                positive_count += 1
    print(
        'positive sample:{} and negative sample:{} and its positive ratio is :{}'.format(positive_count, negative_count,
                                                                                         positive_count / (
                                                                                                 negative_count + positive_count)))
    return train_exmples, sentences1, sentences2, scores


def load_sts_train():
    train_exmples = []
    sts_file = 'datasets/sts/sts-train.csv'
    sentences1 = []
    sentences2 = []
    scores = []
    with open(sts_file) as file:
        for line in islice(file, 0, None):
            content = line.split('\t')
            label = int(content[4])
            sent_1 = content[5]
            sent_2 = content[6]
            sentences1.append(sent_1)
            sentences2.append(sent_2)
            scores.append(label)
            examples = InputExample(texts=[sent_1, sent_2], label=label)
            train_exmples.append(examples)

    return train_exmples, sentences1, sentences2, scores


def load_msrp_train():
    train_exmples = []
    msrp_file = 'datasets/msrp/msr_paraphrase_train.txt'
    positive_count = 0
    negative_count = 0
    sentences1 = []
    sentences2 = []
    scores = []
    with open(msrp_file) as file:
        for line in islice(file, 1, None):
            content = line.split('\t')
            label = int(content[0])
            sent_1 = content[3]
            sent_2 = content[4]
            sentences1.append(sent_1)
            sentences2.append(sent_2)
            scores.append(label)
            examples = InputExample(texts=[sent_1, sent_2], label=label)
            train_exmples.append(examples)
            if int(label) == 0:
                negative_count += 1
            else:
                positive_count += 1
    print(
        'positive sample:{} and negative sample:{} and its positive ratio is :{}'.format(positive_count, negative_count,
                                                                                         positive_count / (
                                                                                                 negative_count + positive_count)))
    return train_exmples, sentences1, sentences2, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Some available pre_trained models are in huggingface,
    # Click this link for detail:
    # https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens#how-to-use

    parser.add_argument('--base_model', type=str, default='bert-base-uncased')
    parser.add_argument('--sentence_embedding_dim', type=int, default=256)
    parser.add_argument('--model_save_path', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='msrp', choices=['msrp', 'sts', 'chinese_bank'])
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'])
    parser.add_argument('--masked', type=bool, default=False, choices=[False, True])
    args = parser.parse_args()

    base_model = args.base_model
    sentence_embedding_dim = args.sentence_embedding_dim
    model_save_path = args.model_save_path
    batch_size = args.batch_size
    epochs = args.epochs
    dataset = args.dataset
    task_type = args.task_type
    masked = args.masked

    word_embedding_model = models.Transformer(base_model, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                               out_features=sentence_embedding_dim,
                               activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    train_examples, sentences1, sentences2, scores = load_train_data(dataset)

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    train_loss = losses.ContrastiveLoss(model=model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100,
              output_path=model_save_path, evaluator=evaluator)
