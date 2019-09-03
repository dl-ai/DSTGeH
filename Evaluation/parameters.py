import argparse

parser=argparse.ArgumentParser(description="A description of what the program does")
parser.add_argument('--toy','-t',action='store_true',help='Use only 50K samples of data')
parser.add_argument('--num_epochs',choices=[5,10,20],default=5,type=int,help='Number of epochs.')
parser.add_argument("--num_layers", type=int, required=True, help="Network depth.")

args=parser.parse_args()
print(args)
print(args.toy,args.num_epochs,args.num_layers)