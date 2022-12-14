#from bert import train_bert
#from xlnet import train_xlnet,evaluate_epoch
#from electra import train_electra
from gpt2 import train_gpt2
#from albert import train_albert
#from roberta import train_roberta
# import argparse
# import sys

# def main():
#     parser = argparse.ArgumentParser(
#         description='CLI for training models'
#     )
#     parser.add_argument(
#         '--model',
#         type=str,
#         choices=['bert', 'xlnet', 'gpt2'],
#         help='Model to train'
#     )
#     parser.add_argument(
#         '-t', '--train',
#         help='Train the model',
#         action='store_true'
#     )
#     parser.add_argument(
#         '-e', '--evaluate',
#         help='Evaluate the model',
#         action='store_true'
#     )
#     parser.add_argument(
#         '-re', '--resume-epoch',
#         type=int,
#         help='Resume training from this epoch'
#     )

#     args = parser.parse_args()
#     model = args.model or None
#     if args.train and model and model == 'bert':
#         print('Training BERT')
#         train_bert()
#     elif args.train and model and model == 'xlnet':
#         print('Training XLNet')
#         train_xlnet()
#     elif args.train and model and model == 'gpt2':
#         print('Training GPT2')
#         train_gpt2()
#     else:
#         print('No model specified')
#         sys.exit(1)

# if __name__ == '__main__':
#     main()

#train_xlnet()
#train_bert()
#evaluate_epoch(2)
train_gpt2()
#train_electra()
#train_albert()
#train_roberta()