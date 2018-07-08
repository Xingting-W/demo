function periodic() {
    var d = document.getElementById('egdiv');
    d.innerHTML = 'Random number: ' + Math.random();
}

function start() {
    // 创建一个网
    var net = new convnetjs.Net();

    setInterval(periodic, 1000);

    // Example: 神经网络分类
    // 定义神经网络的结构
    var layer_defs = [];
    // 输入层。input layer of size 1x1x2 (all volumes are 3D)
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    // 两个全连接类型的隐藏层，隐藏节点为20，激活函数为relu。some fully connected layers
    layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
    layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
    // 输出层，二分类。a softmax classifier predicting probabilities for two classes: 0,1
    layer_defs.push({type:'softmax', num_classes:2});

    // 将上面定义的layer层统一为一个神经网络Net
    net.makeLayers(layer_defs);

    // the network always works on Vol() elements. These are essentially simple wrappers around lists, but also contain gradients and dimensions line below will create a 1x1x2 volume and fill it with 0.5 and -1.3
    // 建立了一个Vol类型的数据，[0.5,-1.3]是数据点。Vol是代码的基本数据结构
    var x = new convnetjs.Vol([0.5, -1.3]);
    // 使用net的前向传播算法，给出数据点x的输出
    var probability_volume = net.forward(x);
    // 网络的输出，probability_volume也是一个Vol类型的变量，probability_volume.w[0]是指数据点x分到第一类的概率
    console.log('probability that x is class 0: ' + probability_volume.w[0]);
    // prints 0.50101


    // 使用类Trainer定义训练的相关参数
    var trainer = new convnetjs.Trainer(net, {learning_rate:0.01, l2_decay:0.001});
    // 使用期方法train对数据点x,分类为0进行训练
    trainer.train(x, 0);

    var probability_volume2 = net.forward(x);
    console.log('probability that x is class 0: ' + probability_volume2.w[0]);
    // prints 0.50374 训练导致输出概率变大



    // Example: Neural Net Regression
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
    layer_defs.push({type:'fc', num_neurons:5, activation:'sigmoid'});
    layer_defs.push({type:'regression', num_neurons:1});
    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    var x = new convnetjs.Vol([0.5, -1.3]);

// train on this datapoint, saying [0.5, -1.3] should map to value 0.7:
// note that in this case we are passing it a list, because in general
// we may want to  regress multiple outputs and in this special case we
// used num_neurons:1 for the regression to only regress one.
    var trainer = new convnetjs.SGDTrainer(net,
        {learning_rate:0.01, momentum:0.0, batch_size:1, l2_decay:0.001});
    trainer.train(x, [0.7]);

// evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
// actual output by looking into its 'w' field:
    var predicted_values = net.forward(x);
    console.log('predicted value: ' + predicted_values.w[0]);
}

