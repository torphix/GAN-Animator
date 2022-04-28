import yaml
from math import ceil

def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def calculate_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def calculate_output_length(in_size, kernel_size, stride, padding):
    # Calculate final length of features
    return int((in_size + padding - kernel_size) / stride) + 2


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def get_configs():
    with open('configs/models.yaml','r') as f:
        models_config = yaml.load(f.read(), yaml.FullLoader)
    with open('configs/data.yaml','r') as f:
        data_config = yaml.load(f.read(), yaml.FullLoader)
    with open('configs/train.yaml','r') as f:
        train_config = yaml.load(f.read(), yaml.FullLoader)
    return models_config, data_config, train_config
    