import re
import argparse
import numpy as np

def parse(log_path):
    with open(log_path) as f:
       text = f.read()

    float_pattern = r'\d+\.\d+'
    mean_pattern = r'AdjustSmoothL1\(mean\): ({}), ({}), ({}), ({})'.format(
        float_pattern, float_pattern, float_pattern, float_pattern)
    var_pattern = r'AdjustSmoothL1\(var\): ({}), ({}), ({}), ({})'.format(
        float_pattern, float_pattern, float_pattern, float_pattern)
    pattern = mean_pattern + r'.*\n.*' + var_pattern + r'.*\n.*' + \
        r'iter: (\d+)  ' + \
        r'loss: ({}) \(({})\)  '.format(float_pattern, float_pattern) + \
        r'loss_retina_cls: ({}) \(({})\)  '.format(float_pattern, float_pattern) + \
        r'loss_retina_reg: ({}) \(({})\)  '.format(float_pattern, float_pattern) + \
        r'loss_mask: ({}) \(({})\)  '.format(float_pattern, float_pattern) + \
        r'time: ({}) \(({})\)  '.format(float_pattern, float_pattern) + \
        r'data: ({}) \(({})\)  '.format(float_pattern, float_pattern) + \
        r'lr: ({})  '.format(float_pattern) + \
        r'max mem: (\d+)'
    reg_exp = re.compile(pattern)

    headers = ['smooth_l1_mean', 'smooth_l1_var', 'iter', 'loss',
               'loss_retina_cls', 'loss_retina_reg', 'loss_mask',
               'time', 'data', 'lr', 'max_mem']

    iterations = list()
    means = list()
    variations = list()
    running_losses = list()
    for args in reg_exp.findall(text):
        mean = [float(v) for v in args[0:4]]
        var = [float(v) for v in args[5:8]]
        iteration = int(args[8])
        point_loss = float(args[9])
        running_loss = float(args[10])
        point_loss_retina_cls = float(args[11])
        running_loss_retina_cls = float(args[12])
        point_loss_retina_reg = float(args[13])
        running_loss_retina_reg = float(args[14])
        point_loss_mask = float(args[15])
        running_loss_mask = float(args[16])
        point_time = float(args[17])
        running_time = float(args[18])
        point_data = float(args[19])
        running_data = float(args[20])
        lr = float(args[21])
        max_mem = int(args[22])

        iterations.append(iteration)
        means.append(mean)
        variations.append(var)
        running_losses.append(running_loss)

    iterations = np.asarray(iterations)
    means = np.asarray(means)
    variations = np.asarray(variations)
    running_losses = np.asarray(running_losses)
    print(iterations)
    print(means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse log file')
    parser.add_argument('log_path', metavar='P', help='path to the log file')
    args = parser.parse_args()

    parse(args.log_path)

