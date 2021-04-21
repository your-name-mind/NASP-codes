from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PARAMS = {
    "none": 0,
    "max_pool_3x3": 0,
    "max_pool_5x5": 0,
    "max_pool_7x7": 0,
    "avg_pool_3x3": 0,
    "avg_pool_5x5": 0,
    "skip_connect": 0,
    "sep_conv_3x3": 504,
    "sep_conv_5x5": 888,
    "sep_conv_7x7": 1464,
    "dil_conv_3x3": 252,
    "dil_conv_5x5": 444,
    "conv_7x1_1x7": 2016,
    "conv 3x3": 1296,
    "conv 5x5": 3600,
    "skip_conv_3x3" : 1308,
    "skip_conv_5x5" : 3612,
    "skip_sep_conv_3x3": 504,
    "skip_sep_conv_5x5": 888,
    "skip_sep_conv_7x7": 1464,
    "skip_dil_conv_3x3": 252,
    "skip_dil_conv_5x5": 444,
    "skip_spat_conv_3x1_1x3": 864,
    "skip_spat_conv_5x1_1x5": 1440,
    "skip_spat_conv_7x1_1x7": 2016
}
PRIMITIVES = [
    "skip_conv_3x3",
    "skip_conv_5x5",
    "skip_sep_conv_3x3",
    "skip_sep_conv_5x5",
    # "skip_sep_conv_7x7",
    "skip_dil_conv_3x3",
    "skip_dil_conv_5x5",
    "skip_spat_conv_3x1_1x3",
    "skip_spat_conv_5x1_1x5",
    # "skip_spat_conv_7x1_1x7",
    "max_pool_3x3",
    "max_pool_5x5",
    "avg_pool_3x3",
    "avg_pool_5x5"
]

# PRIMITIVES_REDUCE = [
#     "skip_connect",
#     "avg_pool_3x3",
#     "max_pool_3x3",
#     "max_pool_5x5",
#     "max_pool_7x7"]

NASNet = Genotype(
    normal=[
        ('sep_5x5', 1),
        ('sep_3x3', 0),
        ('sep_5x5', 0),
        ('sep_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_5x5', 1),
        ('sep_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_3x3', 0),
        ('sep_5x5', 2),
        ('sep_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_7x7', 2),
        ('sep_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

NASP = Genotype(normal=[('conv_3x1_1x3', 0), ('conv 3x3', 1), ('dil_conv_3x3', 2), ('conv 3x3', 1), ('dil_conv_3x3', 2),
                        ('conv 3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6),
                reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_5x5', 1),
                        ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 1)],
                reduce_concat=range(2, 6))
