import chainer


class StandardUnit(chainer.Chain):
    def __init__(self, stage, nb_filter, kernel_size=3):

        with self.init_scope():
            self.conv1
            self.conv2

    def __call__(self, x):
        h = F.dropout(self.conv1(x))
        out = F.dropout(self.conv2(x))
        return out


class NestedUNet(chainer.Chain):
    def __init__(self):
        with self.init_scope():
            self.conv0_0 = StandardUnit(nb_filter=nb_filter[0])
            self.conv1_0
            self.conv2_0
            self.conv3_0
            self.conv4_0

    def __call__(self, x):
        h0_0 = self.conv0_0(x)
        h1_0 = self.conv1_0(h0_0)
        # concatenate
        h0_1 = self.conv0_1()


        h2_0 = self.conv2_0(h1_0)
            
