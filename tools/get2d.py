import libs.dataset.h36m.data_utils2 as data_utils
import libs.parser.parse as parse

opt = parse.parse_arg()
if __name__ == '__main__':
    train_dataset, eval_dataset, stats, action_eval_list = \
        data_utils.prepare_dataset(opt)