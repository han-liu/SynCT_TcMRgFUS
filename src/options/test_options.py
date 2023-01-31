from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.set_defaults(model='test')
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        parser.add_argument('--overlap_ratio', type=float, default=0.6, help='overlapping ratio: higher produces better results but takes longer inference time.')
        parser.add_argument('--input_dir', type=str, help='input directory that stores T1-weighted MRIs (nii.gz)')
        parser.add_argument('--output_dir', type=str, help='output directory to save synthetic CTs')


        self.isTrain = False
        return parser
