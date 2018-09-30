import cPickle
import glob
import numpy as np
import os
import re

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages


pyplot.rc('axes', labelsize=12)

pp = PdfPages('analysis/avg_epochs.pdf')
pyplot.figure()
pyplot.clf()

ci_list = np.array([x / 18.0 for x in range(9, 18)])

baseline_p = np.array([81.38, 81.32, 81.11, 82.66, 83.55, 84.37, 87.00, 88.94, 91.51])
pyplot.plot(ci_list, baseline_p, color='#FFAAAA', linewidth=2, label='no-distill, no-project')

baseline_q = np.array([85.53, 85.34, 86.28, 87.09, 87.34, 87.41, 87.25, 85.54, 84.49])
pyplot.plot(ci_list, baseline_q, color='#FF0000', linewidth=2, label='no-distill, project')

distilll_p = np.array([81.83, 81.77, 81.74, 83.32, 84.16, 85.07, 87.71, 89.64, 91.55])
pyplot.plot(ci_list, distilll_p, color='#AAAAFF', linewidth=2, label='distill, no-project')

distilll_q = np.array([85.36, 85.16, 86.16, 86.94, 87.09, 87.12, 86.91, 85.09, 84.00])
pyplot.plot(ci_list, distilll_q, color='#0000FF', linewidth=2, label='distill, project')

elmooooo_p = np.array([87.25, 87.48, 89.02, 88.98, 90.50, 92.09, 92.65, 94.31, 95.05])
pyplot.plot(ci_list, elmooooo_p, color='#00FF00', linewidth=2, label='ELMo, no-project')

elmooooo_q = np.array([88.19, 88.31, 90.01, 89.99, 91.40, 92.66, 92.81, 93.54, 93.44])
pyplot.plot(ci_list, elmooooo_q, color='#008800', linewidth=2, label='ELMo, project')

# legend = pyplot.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
legend = pyplot.legend(bbox_to_anchor=(0, 1), loc="upper left", mode="expand", borderaxespad=0)
pyplot.xlabel('threshold')
pyplot.ylabel('test performance')
pp.savefig(bbox_inches="tight")
pp.close()
