from model.rhyme_generator import *
import pandas as pd
from sentence_transformers import SentenceTransformer, util



parser = argparse.ArgumentParser(description='Korean Rhyme')


parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--rhyme',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='hiphop_data/train.csv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='hiphop_data/test.csv',
                            help='test file')

        parser.add_argument('--tokenizer_path',
                            type=str,
                            default='tokenizer',
                            help='tokenizer')
        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_seq_len',
                            type=int,
                            default=32,
                            help='max seq len')
        return parser




if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()


    model = KoBARTConditionalGeneration(args)





    if args.rhyme:
        model.model.eval()
        with open("data/lyrics.txt", "r") as f:
            corpus = [l.strip() for l in f.readlines()]
        corpus_embeddings = torch.load("data/sentence_embeddings.pt").to("cuda")
        embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to("cuda")

        while 1:
            q = input('context > ').strip()
            if q == 'quit':
                break
            # 텍스트 변환
            transferred_text = model.rhyme(q)
            transferred_text = transferred_text.replace("<usr>", "").strip()
            print("Rhyme  > {}".format(transferred_text))
            top_k = 5
            embedding = embedder.encode([transferred_text], convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(embedding, corpus_embeddings).flatten()
            top_results = torch.topk(cos_scores, k=top_k)

            print("\n\n======================\n\n")
            print("source sentence:", q)
            print("generated sentence:", transferred_text)
            print("\nTop 5 most similar sentences in corpus:")
            for score, idx in zip(top_results[0], top_results[1]):
                print(corpus[idx], "(Score: {:.4f})".format(score))
