"""
A test to visualize outputs.
Should be made better in neat future.
"""

import matplotlib.pyplot as plt
import numpy as np
import doce
import nmf_audio_benchmark.benchmarks.msa_benchmark as msa_benchmark

selector = {"rank":[10,20],"beta":[2,1,0]}
    

(data, settings, header) = msa_benchmark.experiment.get_output(
  output = 'f_mes_05',
  selector = selector,
  path = 'output'
  )

data = np.array(data)#[:,:,2]

settingIds = np.arange(len(settings))

fig, ax = plt.subplots()
ax.barh(settingIds, np.mean(data, axis=1), xerr=np.std(data, axis=1), align='center')
ax.set_yticks(settingIds)
ax.set_yticklabels(settings)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Accuracy')
ax.set_title(header)

fig.tight_layout()
plt.savefig('metric_display.png')
print('The figure is available here: ./metric_display.png')