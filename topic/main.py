import argparse
from train import trainIters
from evaluate import runTest

def parse():
    parser = argparse.ArgumentParser(description='coarse-to-fine')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-ts', '--test', help='Test the model with corpus')
    parser.add_argument('-ld', '--load', help='Load the saved model and train')
    
    parser.add_argument('-md', '--model', help='the saved model')  
    
    parser.add_argument('-bs', '--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-ly', '--layer', type=int, default=2, help='Number of layers in encoder and decoder')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512, help='Hidden size in encoder and decoder')
    parser.add_argument('-es', '--embed_size', type=int, default=512, help='embedding size of topic')
    parser.add_argument('-as', '--attr_size', type=int, default=512, help='embedding size of attribute, e.g. user, product')
    parser.add_argument('-an', '--attr_num', type=int, default=3, help='number of attributes, user, product, rating')
    parser.add_argument('-bm', '--beam_size', type=int, default=4, help='beam size in decoder')
    parser.add_argument('-or', '--overall', type=int, default=5, help='overall scale, change with the dataset')
    parser.add_argument('-dr', '--lr_decay_ratio', type=float, default=0.8, help='learning rate decay ratio')
    parser.add_argument('-de', '--lr_decay_epoch', type=int, default=5, help='learning rate decay epoch')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00002, help='Learning rate')

    parser.add_argument('-mx', '--max_length', type=int, default=10, help='max length of sequence')
    parser.add_argument('-mn', '--min_length', type=int, default=2, help='min length of sequence')

    parser.add_argument('-sd', '--save_dir', help='saved directory of model')

    args = parser.parse_args()
    return args

def parseFilename(filename):
    filename = filename.split('/')
    layers, hidden, batch_size = filename[-2].split('_')
    n_layers = int(layers)
    hidden_size = int(hidden)
    return n_layers, hidden_size

def run(args):
    learning_rate, lr_decay_epoch, lr_decay_ratio, n_layers, hidden_size, embed_size, \
        attr_size, attr_num, batch_size, beam_size, overall, max_length, min_length, save_dir = \
            args.learning_rate, args.lr_decay_epoch, args.lr_decay_ratio, args.layer, args.hidden_size, args.embed_size, \
                args.attr_size, args.attr_num, args.batch_size, args.beam_size, args.overall, args.max_length, args.min_length, args.save_dir
        
    if args.train and not args.load:
        trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, \
                n_layers, hidden_size, embed_size, attr_size, attr_num, overall, save_dir)
    # load saved model and train
    elif args.load:
        n_layers, hidden_size = parseFilename(args.load)
        trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, \
                n_layers, hidden_size, embed_size, attr_size, attr_num, overall, save_dir, loadFilename=args.load)
    # load parameters from filename
    elif args.test: 
        n_layers, hidden_size = parseFilename(args.model)
        runTest(args.test, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, \
            args.model, beam_size, max_length, min_length, save_dir)

if __name__ == '__main__':
    args = parse()
    run(args)
