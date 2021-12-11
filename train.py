import argparse
import logging
import os

from pytorch_lightning import loggers as pl_loggers
from model.rhyme_generator import *
from utils.loader import RhymeDataModule



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
                            default=36,
                            help='max seq len')
        return parser








if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = RhymeDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = KoBARTConditionalGeneration(args)

    dm = RhymeDataModule(args.train_file,
                        args.test_file,
                        os.path.join(args.tokenizer_path, 'model.json'),
                        max_seq_len=args.max_seq_len,
                        num_workers=args.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1,
                                                       prefix='rhyme-kobart-full')
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)

    model.model.save_pretrained('pretrained_dir/rhyme-kobart-model')


    if args.rhyme:
        model.model.eval()
        while 1:
            q = input('context > ').strip()
            if q == 'quit':
                break
            print("Rhyme  > {}".format(model.rhyme(q)))
