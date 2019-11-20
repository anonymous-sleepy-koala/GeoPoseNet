from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser, isTrain=True):
        parser = BaseOptions.initialize(self, parser)
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--valid_freq', type=int, default=5, help='frequency of checking the performance')

        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        # parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--epoch_start', type=int, default=0, help='epoch to start training')
        parser.add_argument('--epoch_end', type=int, default=500, help='epoch to end training')
        #parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size (default: 4)')
        # parser.add_argument('--val_skip_frames', type=int, default=4, help='skip frames in validation dataset (default: 4)')
        # parser.add_argument('--crop', type=str, default="random", help='bottom | random')        
        parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling images (default: 50)')
        parser.set_defaults(load_gt=True)

        #-------------------optimizer---------------------------
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--step_freq', type=int, default='8', help='frequency at which optimizer steps')
        # lr_parsers = parser.add_subparsers(dest='lr_policy', help='command help for different learning rate policy')
        # parser_linear = lr_parsers.add_parser('linear', help='learning rate policy: linear')
        # parser_linear.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        # parser_linear.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        
        # parser_step = lr_parsers.add_parser('step', help='learning rate policy: step')
        # parser_step.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # parser_step.add_argument('--lr_decay_gamma', type=float, default=0.5, help='multiply by a gamma every lr_decay_iters iterations')

        # parser_plateau = lr_parsers.add_parser('plateau', help='learning rate policy: plateau')
        # parser_plateau.add_argument('--factor', type=float, default=0.3, \
        #     help='Factor by which the learning rate will be reduced')
        # parser_plateau.add_argument('--patience', type=int, default=5, \
        #     help='Number of epochs with no improvement after which learning rate will be reduced')
        # parser_plateau.add_argument('--threshold', type=float, default=0.001, \
        #     help='Number of epochs with no improvement after which learning rate will be reduced')

        # parser_cosine = lr_parsers.add_parser('cosine', help='learning rate policy: cosine')
        # parser_cosine.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        #-------------------optimizer---------------------------

        self.isTrain = isTrain
        return parser
