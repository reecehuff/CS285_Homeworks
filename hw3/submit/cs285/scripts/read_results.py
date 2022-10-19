import glob
import tensorflow as tf

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

import glob

logdir = '../../data/q1_MsPacman-v0_10-10-2022_16-04-36/events*'
eventfile = glob.glob(logdir)[0]
print(eventfile)
X, Y = get_section_results(eventfile)
for i, (x, y) in enumerate(zip(X, Y)):
    print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))