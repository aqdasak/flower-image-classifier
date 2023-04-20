import argparse


def get_train_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir', type=str, help='Data directory on which model will be trained'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='.',
        help='Folder in which checkpoints will be saved',
    )
    parser.add_argument(
        '--arch', type=str, default='vgg11', help='CNN Model Architecture'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.003, help='Learning rate of model'
    )
    parser.add_argument(
        '--hidden_units', type=int, default=2048, help='Number of nodes in hidden layer'
    )
    parser.add_argument(
        '--epochs', type=int, default=5, help='Number of training iterations'
    )
    parser.add_argument('--gpu', action='store_true', help='Train with GPU')

    return parser.parse_args()


def get_predict_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to saved model checkpoint')
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='Top predictions',
    )
    parser.add_argument(
        '--category_names',
        type=str,
        default='',
        help='Json file containing mapping of category to name of flower',
    )
    parser.add_argument('--gpu', action='store_true', help='Train with GPU')

    return parser.parse_args()
