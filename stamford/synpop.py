import argparse
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, GRU, LSTM
from tensorflow.keras.utils import Sequence
from stamford import graph
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def household_data(g):
    sexes = nx.get_node_attributes(g, "sex")
    ages  = nx.get_node_attributes(g, "age")

    households = graph.households(g)
    x = []
    y = []
    hh_max = max(d for (_, d) in nx.degree(g, households))
    batch  = hh_max+1

    def pad(a):
        return a + [[0,0]]*(batch-len(a))

    for hh in households:
        members = [[-1 if sexes[m] == "male" else 1, ages[m]] for m in graph.members(g, hh)]
        for i in range(1, len(members)):
            x.extend(pad(members[:i]))
            y.extend(members[i])
        x.extend(pad(members))
        y.extend([0, 0])

    x = np.array(x, dtype=np.float32).reshape(int(len(x)/batch), batch, 2)
    y = np.array(y, dtype=np.float32).reshape(int(len(y)/2), 1, 2)

    return x, y, hh_max+1

def command():
    logging.basicConfig
    parser = argparse.ArgumentParser("stamford_graph")
    parser.add_argument("--train", "-t", action="store_true", help="Train predictive model")
    parser.add_argument("--generate", "-g", type=int, help="Generate N households")
    parser.add_argument("--bins", "-b", type=int, default=10, help="Bins for age-banding")
    parser.add_argument("graph", help="Sampled population graph")
    parser.add_argument("model", help="Neural network model file (e.g. model.h5)")
    args = parser.parse_args()

    g = nx.read_graphml(args.graph)

    x, y, batch = household_data(g)

    for i in range(2*batch):
        print(x[i], y[i])

    if args.train:
        model = Sequential([
            Input(shape=(batch,)),
            Embedding(input_dim=100, input_length=2, output_dim=2, mask_zero=True),
            Dense(16), #LSTM(64, activation="relu"),
            Dense(2),
        ])

        model.summary()
        model.compile(optimizer='adam', loss='mse')

        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                     ModelCheckpoint(args.model, save_best_only=True, save_weights_only=False)]
        model.fit(x, y, epochs=100, batch_size=batch, validation_split=0.2, callbacks=callbacks)

    if args.generate:
        model = load_model(args.model)

        def pad(a):
            return np.array(a + [[0,0]]*(batch-len(a)), dtype=np.float32).reshape(1, batch, 2)


        a = pad([[1, 38]])
        for i in range(1, 10):
            prediction = model.predict(a)
            a[0, i, :] = prediction[0]
            print(prediction)

        for sv, age in a[0]:
            if sv < 0: sex = "male"
            elif sv > 0: sex = "female"
            else: sex = "unk"
            print("%s\t%d" % (sex, age))


if __name__ == '__main__':
    command()
