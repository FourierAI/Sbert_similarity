import argparse
from itertools import islice

from torch import nn
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers import evaluation
from sentence_transformers import models
import load_data as ld


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

    train_examples = ld.load_dataset(dataset_name=dataset, dataset_type='train')

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    train_loss = losses.ContrastiveLoss(model=model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(train_examples)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100,
              output_path=model_save_path, evaluator=evaluator)
