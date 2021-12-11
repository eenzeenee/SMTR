import argparse
import logging
import os
from model.kobart_rap import *



parser = argparse.ArgumentParser(description='Korean Rap')


parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--rap',
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
                            default=36,
                            help='max seq len')
        return parser




if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()


    model = KoBARTConditionalGeneration(args)





    if args.rap:
        model.model.eval()
        while 1:
            q = input('context > ').strip()
            if q == 'quit':
                break
            print("Rap  > {}".format(model.rap(q)))
