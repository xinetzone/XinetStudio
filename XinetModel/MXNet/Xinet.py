from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from time import time
import matplotlib.pyplot as plt


class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """

    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = self.dataset[:]
        X = nd.array(data[0])
        y = nd.array(data[1])
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            np.random.shuffle(idx)

        for i in range(0, n, self.batch_size):
            j = nd.array(idx[i: min(i + self.batch_size, n)])
            yield nd.take(X, j), nd.take(y, j)

    def __len__(self):
        return len(self.dataset)//self.batch_size

def transform3D(data, label, resize=None):
    # transform a batch of examples
    if resize:
        n = data.shape[0]
        new_data = nd.zeros((n, resize, resize, data.shape[3]))
        for i in range(n):
            new_data[i] = image.imresize(data[i], resize, resize)
        data = new_data
    # change data from batch x height x weight x channel to batch x channel x height x weight
    return nd.transpose(data.astype('float32'), (0, 3, 1, 2))/255, label.astype('float32')

def transform2D(data, label, resize=None):
    # transform a batch of examples
    if resize:
        n = data.shape[0]
        new_data = nd.zeros((n, resize, resize, data.shape[3]))
        for i in range(n):
            new_data[i] = image.imresize(data[i], resize, resize)
        data = new_data
    # change data from batch x height x weight x channel to batch x channel x height x weight
    return nd.transpose(data.astype('float32'), (2, 0, 1))/255, label.astype('float32')


def transform(data, label, resize=None):
    # transform a batch of examples
    if resize:
        n = data.shape[0]
        new_data = nd.zeros((n, resize, resize, data.shape[3]))
        for i in range(n):
            new_data[i] = image.imresize(data[i], resize, resize)
        data = new_data
    
    return data.astype('float32')/255, label.astype('float32')


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list


def SGD(params, lr, batch_size):
    for param in params:
        param -= lr * param.grad / batch_size


def batch_norm2D(X, gamma, beta, is_training, moving_mean, moving_variance,
               eps = 1e-5, moving_momentum = 0.9):
    '''
    事实上，在测试时我们还是需要继续使用批量归一化的，只是需要做些改动。
    在测试时，我们需要把原先训练时用到的批量均值和方差替换成**整个**训练数据的均值和方差。
    但是当训练数据极大时，这个计算开销很大。因此，我们用移动平均的方法来近似计算
    '''
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持 4D 形状使得可以正确的广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
        # 变形使得可以正确的广播
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)

    # 均一化
    if is_training:
        X_hat = (X - mean) / nd.sqrt(variance + eps)
        #!!! 更新全局的均值和方差
        moving_mean[:] = moving_momentum * moving_mean + (
            1.0 - moving_momentum) * mean
        moving_variance[:] = moving_momentum * moving_variance + (
            1.0 - moving_momentum) * variance
    else:
        #!!! 测试阶段使用全局的均值和方差
        X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)

    # 伸缩和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx):
    '''
    `is_training = False` 表示对测试数据进行了 Batch Normlization 处理
    
    '''
    acc = nd.array([0.], ctx= ctx)
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for data, label in data_iterator:
        label = label.as_in_context(ctx)
        data = data.as_in_context(ctx)
        acc += nd.sum(net(data).argmax(axis=1)==label)
        n += len(label)
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n


def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, batch_size, print_batches=None):
    """
    Train a network
    `BN = True` 表示对训练数据进行了 Batch Normlization 处理
    """
    print(("Start training on ", ctx))
    if isinstance(train_data, mx.io.MXDataIter):
        train_data.reset()

    n = len(train_data) // batch_size  
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.

        start = time()
        for data, label in train_data:
            label = label.as_in_context(ctx)
            data = data.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            L.backward()
            
            if isinstance(trainer, gluon.Trainer):
                trainer.step(batch_size)
                trainer.set_learning_rate(trainer.learning_rate * 0.01)
            else:
                # 将梯度做平均，这样学习率会对 batch size 不那么敏感
                trainer
                
            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)

        test_acc = evaluate_accuracy(test_data, net, ctx)

        print(("Epoch %d. Loss: %g, Train acc %g, Test acc %g, Time %g sec" % (
                epoch, train_loss/n, train_acc/n, test_acc, time() - start)))


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                   strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                       strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


def resnet18(num_classes):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.BatchNorm(),
            nn.Conv2D(64, kernel_size=3, strides=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            Residual(64),
            Residual(64),
            Residual(128, same_shape=False),
            Residual(128),
            Residual(256, same_shape=False),
            Residual(256),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes)
        )
    return net


def show_images(imgs, nrows, ncols, figsize=None):
    """plot a list of images"""
    if not figsize:
        figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i][j].imshow(imgs[i*ncols+j].asnumpy())
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a random order from sequential data."""
    # Subtract 1 because label indices are corresponding input indices + 1.
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # Randomize samples.
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # Read batch_size random samples each time.
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    # Subtract 1 because label indices are corresponding input indices + 1.
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label


def grad_clipping(params, clipping_norm, ctx):
    """Gradient clipping."""
    if clipping_norm is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > clipping_norm:
            for p in params:
                p.grad[:] *= clipping_norm / norm


def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                char_to_idx, get_inputs, is_lstm=False):
    """Predict the next chars given the prefix."""
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        if is_lstm:
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)
        if i < len(prefix)-1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim,
                          learning_rate, clipping_norm, batch_size,
                          pred_period, pred_len, seqs, get_params, get_inputs,
                          ctx, corpus_indices, idx_to_char, char_to_idx,
                          is_lstm=False):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(1, epochs + 1):
        # If consecutive sampling is used, in the same epoch, the hidden state
        # is initialized only at the beginning of the epoch.
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps,
                                     ctx):
            # If random sampling is used, the hidden state has to be
            # initialized for each mini-batch.
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs shape：(batch_size, vocab_size)
                if is_lstm:
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h,
                                                    state_c, *params)
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # Let t_ib_j be the j-th element of the mini-batch at time i.
                # label shape：（batch_size * num_steps）
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ].
                label = label.T.reshape((-1,))
                # Concatenate outputs:
                # shape: (batch_size * num_steps, vocab_size).
                outputs = nd.concat(*outputs, dim=0)
                # Now outputs and label are aligned.
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, clipping_norm, ctx)
            SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            print(("Epoch %d. Training perplexity %f" % (e,
                                                        exp(train_loss/num_examples))))
            for seq in seqs:
                print((' - ', predict_rnn(rnn, seq, pred_len, params,
                                         hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                                         is_lstm)))
            print()