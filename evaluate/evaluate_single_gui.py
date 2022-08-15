from __future__ import print_function
from __future__ import absolute_import

__author__ = 'Taneem Jan'

from argparse import ArgumentParser
from Evaluator import *


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--original_gui_filepath', type=str,
                        dest='original_gui_filepath', help='filepath of original gui file',
                        required=True)
    parser.add_argument('--predicted_gui_filepath', type=str,
                        dest='predicted_gui_filepath', help='filepath of original gui file',
                        required=True)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    original_gui_filepath = options.original_gui_filepath
    predicted_gui_filepath = options.predicted_gui_filepath

    bleu_score = Evaluator.get_sentence_bleu(original_gui_filepath, predicted_gui_filepath)
    print("BLEU score for single GUI: {}".format(bleu_score))


if __name__ == "__main__":
    main()
