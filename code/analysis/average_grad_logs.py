import numpy as np
import re

job = "TEST SET"
print("=" * (len(job) + 4))
print("| %s |" % job)
print("=" * (len(job) + 4))
print('\n')

re_exps = [
    ("avg_grad", re.compile(r'Average\sgradients\s:-\s(\d*\.?\d*)')),
    ("avg_grad_no_pad", re.compile(r'Average\sno\spad\sgradients\s:-\s(\d*\.?\d*)')),
    ("avg_A_grad", re.compile(r'Average\sA\sgradients\s:-\s(\d*\.?\d*)')),
    ("avg_A_grad_no_pad", re.compile(r'Average\sA\sno\spad\sgradients\s:-\s(\d*\.?\d*)')),
    ("avg_B_grad", re.compile(r'Average\sB\sgradients\s:-\s(\d*\.?\d*)')),
    ("avg_B_grad_no_pad", re.compile(r'Average\sB\sno\spad\sgradients\s:-\s(\d*\.?\d*)'))
]

jobs = ['grad_100', 'grad2_99']

results = {}

for job in jobs:
    results[job] = {key: [] for key, _ in re_exps}

    for i in range(100):
        with open('logs/test_%s_seed_%d.log' % (job, i), 'r') as f:
            data = f.read().split('\n')[-7:-1]
        for (key, regex), text in zip(re_exps, data):
            matches = regex.findall(text)
            results[job][key].append(float(matches[0]))

    print("=" * (len(job) + 4))
    print("| %s |" % job)
    print("=" * (len(job) + 4))

    for key, _ in re_exps:
        print("%s :- %.4f" % (key, np.mean(results[job][key])))

job = "SEEDS COMPARISON"
print("=" * (len(job) + 4))
print("| %s |" % job)
print("=" * (len(job) + 4))

for key, _ in re_exps:
    count = 0
    for r1, r2 in zip(results['grad_100'][key], results['grad2_99'][key]):
        if r1 > r2:
            count += 1
    print("%s :- %d grad_100, %d grad2_99" % (key, count, len(results['grad_100'][key]) - count))
