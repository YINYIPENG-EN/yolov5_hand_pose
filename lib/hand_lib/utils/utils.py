import os

"""Parses the data configuration file"""
def parse_data_cfg(path):
    print('data_cfg ： ',path)
    options = dict()
    with open(path, 'r',encoding='UTF-8') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
