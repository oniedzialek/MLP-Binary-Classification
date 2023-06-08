import numpy as np

np.set_printoptions(threshold=np.inf)

filename = "E:\\Studia\\AI\\ProjektAI\\moje skrypty\\agaricus-lepiota.data"
data = np.loadtxt(filename, delimiter=',', dtype=str)
print((data=='?').sum(axis=0))

data = data[(data=='?').any(axis=1)==0]
print(data.shape)

filename_out = "E:\\Studia\\AI\\ProjektAI\\moje skrypty\\agaricus-lepiota-po-usunieciu-pyta.data"
np.savetxt(filename_out, data, delimiter=',', fmt='%s')

