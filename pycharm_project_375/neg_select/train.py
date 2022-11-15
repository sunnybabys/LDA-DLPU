import pickle
import time
import random
import mxnet.model
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl


from utils import build_graph, sample, load_data
from model import GNNMDA, GraphEncoder, BilinearDecoder


def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    g, all_sample, p_sample_df, unknown_associations, random_positive, IL_shape = build_graph(directory, random_seed=random_seed,
                                                                                    ctx=ctx)

    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## lncrna nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## disease nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    label_edata = g.edata['rating']
    src, dst = g.all_edges()

    # Train the model
    model = GNNMDA(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g, aggregator=aggregator,
                                dropout=dropout, slope=slope, ctx=ctx),
                   BilinearDecoder(feature_size=embedding_size))  # 指定编码器和解码器

    model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
    cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    for epoch in range(epochs):
        start = time.time()
        for _ in range(10):
            with mx.autograd.record():
                score_train = model(g, src, dst)

                loss_train = cross_entropy(score_train, label_edata).mean()
                loss_train.backward()
            trainer.step(1)
        end = time.time()
        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(), 'Time: %.2f' % (end - start))
    print('## Training Finished !')
    h = model.encoder(g)
    src_test, dst_test = g.all_edges()
    print(src_test, dst_test)
    sc = model.decoder(h[src_test], h[dst_test])
    src_test = src_test.asnumpy()
    dst_test = dst_test.asnumpy()
    sc = sc.asnumpy()
    a = []
    for i in range(len(dst_test)):
        a.append([src_test[i], dst_test[i], sc[i]])
    b = np.array(a).reshape(-1, 3)
    c = []
    length = int(b.shape[0] / 2)
    for i in range(length):
        print(b[i, 0], b[i, 1], '*' * 5, b[length + i, 0], b[length + i, 1])
        c.append([b[i, 0], b[i, 1], (b[length + i, 2] + b[i, 2]) / 2])
    prob = pd.DataFrame(data=np.array(c).reshape(-1, 3), columns=['e1', 'e2', 'probability'])
    return prob, p_sample_df, unknown_associations, random_positive ,IL_shape
