# Legenda
legend = {
    'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 1, 'f': 6, 'g': 7,
    'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14,
    'o': 15, 'p': 0, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
    'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25
}

with open("E:\\Studia\\AI\\ProjektAI\\moje skrypty\\agaricus-lepiota-po-usunieciu-pyta.data", 'r') as filein:
    data = filein.readlines()

with open("E:\\Studia\\AI\\ProjektAI\\moje skrypty\\agaricus-lepiota-w-liczbach.data", 'w') as fileout:
    for line in data:
        line_numbers = [str(legend.get(c.lower())) for c in line if c.isalpha()]
        numbers = ','.join(line_numbers)
        fileout.write(numbers + '\n')


for letter, number in legend.items():
    print(f'{letter}: {number}')
