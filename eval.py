import argparse
import json
from itertools import islice

from sklearn import metrics
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from sentence_transformers import SentenceTransformer, util


def eval_classification(scores, predicted_label, output_file):
    precision, recall, f1, count = metrics.precision_recall_fscore_support(scores, predicted_label)
    precision_macro, recall_macro, f1_macro, count_macro = metrics.precision_recall_fscore_support(scores,
                                                                                                   predicted_label,
                                                                                                   average='macro')
    accuracy = metrics.accuracy_score(scores, predicted_label)
    result = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "count": count.tolist(),
        "total": len(scores),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "accuracy": accuracy
    }

    with open(output_file, 'w') as file:
        json.dump(result, file)


def load_chbank_test():
    bank_file = 'datasets/chinese/bank_test.txt'
    negative_count=0
    positive_count = 0
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
            if int(label) == 0:
                negative_count += 1
            else:
                positive_count += 1

    threshold = positive_count / (negative_count + positive_count)

    return sentences1, sentences2, scores, threshold


def load_sts_test():
    sts_file = 'datasets/sts/sts-test.csv'
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

    return sentences1, sentences2, scores

def load_msrp_test():
    msrp_file = 'datasets/msrp/msr_paraphrase_test.txt'
    sentences1 = []
    sentences2 = []
    scores = []
    positive_count = 0
    negative_count = 0
    with open(msrp_file) as file:
        for line in islice(file, 1, None):
            content = line.split('\t')
            label = content[0]
            sent_1 = content[3]
            sent_2 = content[4]
            sentences1.append(sent_1)
            sentences2.append(sent_2)
            scores.append(int(label))
            if int(label) == 0:
                negative_count += 1
            else:
                positive_count += 1

    threshold = positive_count / (negative_count + positive_count)

    return sentences1, sentences2, scores, threshold


def eval_regression(scores, cosine_scores_list, output_file):
    spearman_coe = spearmanr(scores, cosine_scores_list)
    pearson_coe = pearsonr(scores, cosine_scores_list)
    result = {
        "spearman": spearman_coe,
        "pearson": pearson_coe
    }
    with open(output_file, 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', type=str, default='./')
    parser.add_argument('--output_file', type=str, default='performance.json')
    parser.add_argument('--dataset', type=str, default='msrp', choices=['msrp', 'sts', 'chinese_bank'])
    parser.add_argument('--task_type', type=str, default='', choices=['classification', 'regression'])
    args = parser.parse_args()

    trained_model_path = args.trained_model_path
    output_file = args.output_file
    dataset = args.dataset
    task_type = args.task_type

    if dataset == 'msrp':
        sentences1, sentences2, scores, threshold = load_msrp_test()
    elif dataset == 'sts':
        sentences1, sentences2, scores = load_sts_test()
    else:
        sentences1, sentences2, scores, threshold = load_chbank_test()

    model = SentenceTransformer(trained_model_path)

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    cosine_scores_list = [cosine_scores[i][i] for i in range(len(sentences1))]

    if task_type == "classification":
        predicted_label = [1 if score > threshold else 0 for score in cosine_scores_list]
        eval_classification(scores, predicted_label, output_file)
    else:
        eval_regression(scores, cosine_scores_list, output_file)