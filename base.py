import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    input_filepath = args.input
    output_filepath = args.output

    sentences1 = []
    sentences2 = []
    scores = []
    with open(input_filepath, 'r') as file:
        for line in file:
            if not line.startswith('index'):
                contents = line.split('\t')
                sentences1.append(contents[7])
                sentences2.append(contents[8])
                scores.append(contents[9])

    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    cosine_scores_list = [cosine_scores[i][i] for i in range(len(sentences1))]

    with open(output_filepath, 'a+') as file:
        for index, score in enumerate(cosine_scores_list):
            file.write(str(score.item()) + ' ' + str(scores[index]))
