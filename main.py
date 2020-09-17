from cnn.cnn import MyCnn
from cnn.layers import Conv2D, Pooling, Dense, Flatten
import numpy as np

if __name__ == "__main__":
    # Import testing purposes
    cnn = MyCnn()
    inp = []
    for i in range(6):
        inp.append(np.random.randint(0, 5, size=(6, 6)))
    print('input: ')
    print(inp)

    cnn.add(Conv2D(0, 2, [5,5], 1, np.array([[6,6] for _ in range(5)])))
    res = cnn.feed_forward(inp)

    print('Conv2D: ')
    print(res)

    cnn.add(Pooling([2, 2], 2, 'avg'))
    res = cnn.feed_forward(inp)

    print('Pooling: ')
    print(res)

    cnn.add(Flatten())
    res = cnn.feed_forward(inp)

    print('Flatten: ')
    print(res)

    cnn.add(Dense(2))
    res = cnn.feed_forward(inp)

    print('Dense: ')
    print(res)
