import argparse

from sentence_transformers import SentenceTransformer, util, evaluation

import load_data as ld

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', type=str, default='./model_save')
    parser.add_argument('--output_dir', type=str, default='./performance/')
    parser.add_argument('--dataset', type=str, default='msrp', choices=['msrp', 'sts', 'atec', 'ccks', 'chsts'])
    parser.add_argument('--task_type', type=str, default='', choices=['classification', 'regression'])
    args = parser.parse_args()

    trained_model_path = args.trained_model_path
    output_dir = args.output_dir
    dataset = args.dataset
    task_type = args.task_type

    test_examples = ld.load_dataset(dataset_name=dataset, dataset_type='test')

    if task_type == "classification":
        evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(test_examples)
    else:
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

    model = SentenceTransformer(trained_model_path)
    model.evaluate(evaluator, output_dir)
