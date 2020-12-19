# -*- coding: utf-8 -*-

"""
tools for read config
"""

import yaml


class ConfigParser():
    """
    Config Reader
    """
    def __init__(self, config_path, tag_num):
        # load
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        # config
        self.model_name = config['model_name']
        self.dataset_dir = config['dataset_dir']
        self.random_split_num = config['random_split_num']
        self.label_file = config['params'][tag_num]['label_file']
        self.stepwise = config['params'][tag_num]['stepwise']
        self.fine_tune_model = config['params'][tag_num]['fine_tune_model']
        self.tag = config['params'][tag_num]['tag']
        self.EPOCH = config['params'][tag_num]['EPOCH']
        self.LR = config['params'][tag_num]['LR']
        self.batch_size = config['params'][tag_num]['batch_size']
        self.label_file_pred = config['params'][tag_num]['label_file']
        self.test_points_pos = config['params'][tag_num]['test_points_pos']
        self.metric_list = config['metric_list']
        self.interval = config['interval']
        self.test_index_list = config['test_index_list']
        self.fix_test_dataset = config['params'][tag_num].get('fix_test_dataset', None)
        
        
def test():
    """
    test
    """
    config_path = '../config.yml'
    config = ConfigParser(config_path, '1')
    print 'config.metric_list', config.metric_list


if __name__ == "__main__":
    test()
