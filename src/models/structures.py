from collections import namedtuple

Dataset = namedtuple(
    'dataset',
    [
        'name',
        'x_train',
        'x_test',
        'y_train',
        'y_test'
    ])

Layer = namedtuple(
    'layer',
    [
        'size',
        'activation'
    ])

generator_config = namedtuple(
    'gen_config', 
    [ 
        'in_dim',
        'n_gen',
        'n_classes',
        'scale',
        'intermediate',
        'merge'
    ]
)

Config = namedtuple(
    'configuration',
    [
        'generator',
        'merge',
        'output'
    ]
)

Result = namedtuple(
    'fit_result',
    [
        'acc_store',
        'val_acc_store',
        'test_acc',
        'history'
    ]
)