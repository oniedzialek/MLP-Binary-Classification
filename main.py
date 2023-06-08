import hickle as hkl
import numpy as np
import nnet_jit as net
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from matplotlib.ticker import MaxNLocator


class mlp_3w:
    def __init__(self, x, y_t, K1, K2, lr, err_goal, \
                 disp_freq, max_epoch):
        self.x = x
        self.L = self.x.shape[0]
        self.y_t = y_t
        self.K1 = K1
        self.K2 = K2
        self.lr = lr
        self.err_goal = err_goal
        self.disp_freq = disp_freq
        self.max_epoch = max_epoch
        self.K3 = y_t.shape[0]
        self.SSE_vec = []
        self.PK_vec = []
        self.data = x.T
        self.target = y_t

        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        hkl.dump([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], 'weights3l.hkl')
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = hkl.load('weights3l.hkl')
        self.SSE = 0
        self.lr_vec = list()

    def predict(self, x):
        n = np.dot(self.w1, x)
        self.y1 = net.tansig(n, self.b1 * np.ones(n.shape))
        n = np.dot(self.w2, self.y1)
        self.y2 = net.tansig(n, self.b2 * np.ones(n.shape))
        n = np.dot(self.w3, self.y2)
        self.y3 = net.purelin(n, self.b3 * np.ones(n.shape))
        return self.y3

    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch + 1):

            # Przewidywana wartosc koncowa
            self.y3 = self.predict(x_train)

            # Roznica miedzy wartoscia oczekiwana a bledem
            self.e = y_train - self.y3

            # Przypisanie aktualnego bledu do SEE_t_1
            self.SSE_t_1 = self.SSE
            # Suma kwadratu bledow. Miara bledu, ktora jest uzywana do oceny wydajnosci sieci neuronowej
            self.SSE = net.sumsqr(self.e)

            # Tworzy wektor, gdzie 1 oznacza, że błąd prognozy jest większy lub równy 0.5, a 0 w przeciwnym wypadku.
            # Dodaje te wartości, dzięki czemu uzyskujemy liczbę prognoz, które miały błąd większy lub równy 0.5.
            # Dzielimy przez liczbe wszystkich prognoz, odejmujemy to od 1 i mnozymy przez 100 by uzyskac wynik w procentach.
            # Dzieki temu uzyskujemy prognoze ile danych miało pomyłkę mniejszą niż 0.5 i możemy uznać je za poprawnie przewidziane
            self.PK = (1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[1]) * 100

            # Dodanie PK do wektora wszysktich PK by monitorowac dokladnosc modelu w czasie
            self.PK_vec.append(self.PK)

            # Konczy proces trenowania jesli osiagnieto oczekiwany blad lub 100 procent poprawnosci
            if self.SSE < self.err_goal or self.PK == 100:
                break

            # Jezeli suma kwadratow błędów jest wartościa NaN (Not a Number), oznacza to, ze cos poszlo nie tak w procesie uczenia
            if np.isnan(self.SSE):
                break

            # Implementacja propagacji wstecznej, która jest stosowana do aktualizacji wag w sieci neuronowej.
            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)

            # Bazując na gradiencie, learnbp zmienia wagi oraz bias dla x
            # Wagi są aktualizowane na podstawie błędu d (kierunek i wartość) mnożonego przez lr (dokładność-szybkość uczenia czyli z jaką dokładnością zmieniamy wartości wag)
            self.dw1, self.db1 = net.learnbp(x_train, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            self.w1 += self.dw1
            self.b1 += self.db1
            self.w2 += self.dw2
            self.b2 += self.db2
            self.w3 += self.dw3
            self.b3 += self.db3

            self.SSE_vec.append(self.SSE)

    def train_CV(self, CV, skfold):
         PK_vec = np.zeros(CVN)

         # Start petli uczacej
         # skfold dzieli data i target na podgrupy (foldy), które beda uczone
         # train i test sa zakresem indeksów z tabel data i target. Pozwalaja na przydzielenie odpowiednich zbiorow do zestawow treningowych i testowych
         for i, (train, test) in enumerate(skfold.split(self.data, np.squeeze(self.target)), start=0):
             x_train, x_test = self.data[train], self.data[test]
             y_train, y_test = np.squeeze(self.target)[train], np.squeeze(self.target)[test]

             # Czesc trenujaca - cale przejscie przez funckje oraz metoda gradientowa czyli propagacja wsteczna
             # Przestawiane są biasy i wagi dla danych połączeń
             self.train(x_train.T, y_train.T)
             # Nowo ustawione wagi i biasy testowane są nie dla zbioru treningowe lecz dla zbioru testowego, by stwierdzić czy szkolenie umożliwiło uniwersalne zastosowania
             result = mlpnet.predict(x_test.T)

             n_test_samples = test.size
             temp = abs(result - y_test)>=0.5
             temp[np.isnan(result)] = True

             # Poprawna predykcja podana w procentach zapisywana jest do wektora
             PK_vec[i] = (1 - sum(temp.astype(int)[0]) / n_test_samples) * 100

         # Oblicza stednia wartosc elementow tablicy
         PK = np.mean(PK_vec)
         return PK



#   zaladowanie przygotowanych danych

x, y_t, x_norm = hkl.load("agaricus-lepiota-normalized-2.hkl")

plt.ion()
plt.switch_backend('Qt5Agg')

max_epoch = 1000
err_goal = 0.25
disp_freq = 100

lr_vec = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
K1_vec = np.arange(1, 21, 1)
K2_vec = np.arange(1, 11, 1)

start = timer()

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

PK_2D_K1K2 = np.zeros([len(K1_vec), len(K2_vec)])
PK_2D_lr = np.zeros([len(lr_vec)])
PK_2D_K1K2_max = 0
k1_ind_max = 0
k2_ind_max = 0

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        mlpnet = mlp_3w(x_norm, y_t, int(K1_vec[k1_ind]), int(K2_vec[k2_ind]), \
                        lr_vec[-3], err_goal, disp_freq, \
                        max_epoch)
        PK = mlpnet.train_CV(CVN, skfold)
        print("K1 {} | K2 {} | PK {}".format(K1_vec[k1_ind], K2_vec[k2_ind], PK))
        PK_2D_K1K2[k1_ind, k2_ind] = PK
        if PK > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK
            k1_ind_max = k1_ind
            k2_ind_max = k2_ind

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')
ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK')
ax.view_init(30, 200)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("Fig.1_K1K2_PK_mushroom.png", bbox_inches='tight')
plt.show(block=True)

PK_2D_lr_max = 0
lr_ind_max = 0

for lr_ind in range(len(lr_vec)):
    mlpnet = mlp_3w(x_norm, y_t, int(K1_vec[k1_ind_max]), int(K2_vec[k2_ind_max]), \
                    lr_vec[lr_ind], err_goal, disp_freq, \
                    max_epoch)
    PK = mlpnet.train_CV(CVN, skfold)
    print("lr {} | PK {}".format(lr_vec[lr_ind], PK))
    PK_2D_lr[lr_ind] = PK

    if PK > PK_2D_lr_max:
        PK_2D_lr_max = PK
        lr_ind_max = lr_ind

fig = plt.figure(figsize=(8, 8))

plt.plot(np.log10(lr_vec), PK_2D_lr)
plt.savefig("Fig.2_lr_PK_mushroom.png", bbox_inches='tight')
plt.show(block=True)

print("Optimal metaparameters: K1={} | K2={} | lr={} | PK={}". \
      format(K1_vec[k1_ind_max], K2_vec[k2_ind_max], lr_vec[lr_ind_max], \
             PK_2D_lr[lr_ind_max]))
print("Execution time:", timer() - start)

