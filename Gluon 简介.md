MXNet 0.11 发布，加入动态图接口 `Gluon`，还有两位 CMU 教授的亲笔教程。

0.11 是 MXNet 正式加入 Apache 以后的第一个版本，官方网站搬到了[Apache](https://mxnet.incubator.apache.org/versions/master/)的服务器（注意：要在最上方 Version 处选择 master 才能看到包含 Gluon 的最新文档）。

这次最大的改进是加入了动态图接口 `Gluon`。Gluon 学习了 Keras，Chainer，和 Pytorch 的优点，并加以改进。接口更简单，且支持动态图（Imperative）编程。相比 TF，Caffe2 等静态图（Symbolic）框架更加灵活易用。同时 Gluon 还继承了 MXNet ==速度快，省显存==[^1]，并行效率高的优点，并支持静、动态图混用，比 Pytorch 更快。

[^1]: 这是MXNet自身拥有的优势。

同时为了彻底解决 MXNet 文档不全的弱点，还特地邀请了前 CMU 知名教授 Alex Smola 和即将出任 CMU 教授的小网红 Zachary Lipton 联手为 Gluon 打造[文档](http://gluon.mxnet.io/#)！

以下内容摘自：[MXNet 0.11发布，加入动态图接口Gluon，还有两位CMU教授的亲笔教程](https://zhuanlan.zhihu.com/p/28648399)
## 接口更简洁
Gluon 采用 Keras 和 Numpy 风格 API，并且 Layer 可以自动判断输入长度。用过 Chainer 和 Pytorch 的人想必都体会过每一层都要记住前一层输出长度的麻烦，从卷积层到全联接层过渡时长度计算更是痛苦，往往要运行一遍才知道。在 Gluon 里则没有这种问题，每层只要指定输出长度，输入长度则可以由系统自动计算。

## 速度更快

深度学习框架大体分为两类：以 TensorFlow，caffe2 为代表的静态图（Symbolic）框架和以 Chainer，Pytorch 为代表的动态图（Imperative）框架。静态图的优势在于速度快，省内存，便于线上部署。而动态图框架的优势是灵活，易用，debug 方便，特别是在自然语言处理和增强学习等领域，比起静态图框架有显著优势。

Gluon 同时支持灵活的动态图和高效的静态图，让你在享受动态编程的灵活易用的同时最小化性能的损失。而 Gluon 的 HybridBlock 和 hybridize 接口让你可以在静态动态间一键切换。0.11 版 Gluon 比 0.20 版 Pytorch 快 $10\%$ 以上，在未来的一两个月我们会加入更多优化，再提高 $10\%$ 以上的性能。

## 既是文档，又是教材

深度学习的教材和样例虽多，但是教材往往重理论轻实践，而样例重实践却不系统。为了填补理论和实践之间的空白，并一举解决 MXNet 文档不全的弱项，我们特邀两位 CMU 教授 Alex Smola 和 Zachary Lipton 为 Gluon 撰写一部兼顾深度学习理论，动手编程，和实战应用的[文档+教材](https://github.com/zackchase/mxnet-the-straight-dope)。

Gluon 教程包括深度学习理论讲解和代码实践。前五章每个例子都包括了两个版本。从零开始（from scratch）版本深入讲解所有细节，Gluon 版本则着重演示高级封装的灵活高效。建议刚开始接触深度学习的同学从头开始顺序阅读，而已经有一定经验的同学可以跳过基础教程只看 Gluon 版。这套教程现在在 Github 上公开写作，共计划 18 章，已经完成了前五章。印刷出版和中文翻译也在计划中。我们保证每天更新，绝不弃坑，欢迎大家试读，也欢迎参与创作！

## Gluon与其他框架的对比
引自：[深度炼丹](https://zhuanlan.zhihu.com/c_94953554)
Tensorflow：Gluon 同时支持静态图和动态图，在灵活性和速度上都有优势。但由于 Gluon 刚刚面市，在成熟度和线上部署方便还有不足。总的来说在做深度学习研究的同学不妨一试。

Pytorch：Gluon 与 Pytorch 的相似度很高，而 Gluon 独特的静、动态图混合功能可以在不牺牲灵活性的前提下提高性能。如果你喜欢 pytorch 的简单易用又在乎性能，那么强烈建议你试一试 Gluon。

### Gluon的定位
PyTorch 定位于科研，在 Facebook 内部使用 Caffe2 作为产品的部署和应用，那么 Gluon 是如何定位的呢？
官方称 Gluon 不仅定位于科研，同时也可用于产品。这无疑比 PyTorch 更好，因为这样不需要再重写代码，而且两个框架之间转化也容易丢掉一些细节，从而模型达不到之前的精度，能够有一套统一的框架，不仅能做科研，同时能够用于产品部署无疑是最好的解决方案。我相信以 dmlc 的实力肯定能做出了，毕竟 MXNet 的设计理念是非常领先的。

### Gluon的优势
大致看完 Gluon 的 api，和 PyTorch 非常非常像，开发者也说了 Gluon 学习了 Keras，Chainer 和 PyTorch 的优点并加以改进，相似的 api 无疑是一个优势。之前[我](https://zhuanlan.zhihu.com/p/28752061)是 PyTorch 使用者，这两天我尝试着将之前的 PyTorch 教程移植到 Gluon 下面，发现非常方便，几乎大体的框架都不用改动，只需要该一些小的 api 就可以了，所以非常方便用户迁移过来，这是我的 [PyTorch](https://github.com/SherlockLiao/pytorch-beginner) 教程和移植的 [Gluon教程](https://github.com/SherlockLiao/mxnet-gluon-tutorial)。

第二个优势是 MXNet 的优势，就是速度快，省显存，并行效率高，分布式简单等等。这些优势并不是 Gluon 带来的，而是 MXNet 一直以来的优势。

第三个优势就是静态图和动态图的转换，PyTorch 只有动态图的模式，有的时候我们的网络结构其实是一个静态图，但是通过 PyTorch 每次都会重新构建动态图，而 Gluon 提供了一个静态图和动态图之间切换的方式。Gluon 中的模块 `gluon.nn.Sequential` 和 `gluon.Block` 分别与 PyTorch 中的 `torch.nn.Sequential` 和 `torch.nn.Module` 对应，他们都是动态图的构建，而 Gluon 中还提供了 `gluon.nn.hybridSequential` 和 `gluon.HybridBlock`，这两个模块就可以在动态图和静态图之间转换，使用者可以先用 `imperatvie` 的方式写网络，debug，最后跑通网络之后，如果网络是一个静态图结构，就可以用 `net.hybridize()` 的方式将其转换成静态图，众所周知静态图的运算会比动态图快，所以这是 Gluon 比 PyTorch 更好的地方。

第四个优势就是前面提到过的，科研和工业界可以都用同一个框架开发，避免的代码的重写，减少劳动效率，避免出现问题。
