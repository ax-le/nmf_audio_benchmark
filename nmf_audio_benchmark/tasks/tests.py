import numpy as np

from nmf_audio_benchmark.tasks.music.msa import CBMEstimator

#TODO: do real tests.
dumb_cbm = CBMEstimator()
print(dumb_cbm.score(predictions=np.array([[0, 1],[1,4], [4,6]]), annotations=np.array([[0, 2],[2,4], [4,6]])))