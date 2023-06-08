import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

filename = "E:\\Studia\\AI\\ProjektAI\\moje skrypty\\agaricus-lepiota-w-liczbach-posortowane-wg-cechy.data"
data = np.loadtxt(filename, delimiter=',', dtype=str)

x = data[:,1:].astype(float).T
y_t = data[:,0].astype(float)
y_t = y_t.reshape(1,y_t.shape[0]) # dla kodowanie klas naturalno-liczbowego
# min i max dla każdej z cech przed normalizacją
np.transpose([np.array(range(x.shape[0])), x.min(axis=1), x.max(axis=1)])

#usuwanie cechy o takiej samej wartosci w kazdym recordzie
x = np.delete(x, 15, 0)

# normalizacja do przedziału <-1,1> wg zależnoci
# x_norm = (x_norm_max-x_norm_min)*(x-x_min)/(x_max-x_min) + x_norm_min
# w której x_norm_max oraz x_norm_min są docelowymi wartociami rozpiętoci cechy
x_min = x.min(axis=1)
x_max = x.max(axis=1)
x_norm_max = 1
x_norm_min = -1
x_norm = np.zeros(x.shape)
for i in range(x.shape[0]):
    x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])* \
        (x[i,:]-x_min[i]) + x_norm_min
# sprawdzenie rozpiętoci cech po normalizacji
np.transpose([np.array(range(x.shape[0])), x_norm.min(axis=1), x_norm.max(axis=1)])
# sortowanie y_t
y_t_s_ind = np.argsort(y_t)
x_n_s = np.zeros(x.shape)
y_t_s = np.zeros(y_t.shape)
for i in range(x.shape[1]):
    y_t_s[0,i] = y_t[0,y_t_s_ind[0,i]]
    x_n_s[:,i] = x_norm[:,y_t_s_ind[0,i]]
plt.plot(y_t_s[0])
hkl.dump([x,y_t,x_norm],'agaricus-lepiota-po-normalizacji.hkl')
# x,y_t,x_norm,x_n_s,y_t_s = hkl.load('zoo.hkl')
