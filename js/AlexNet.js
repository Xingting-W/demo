(function (convnetjs, document) {
    //构建层
    let layer_defs = [];
    // 输入一个227*227的RGB图片
    layer_defs.push({type: 'input', out_sx: 227, out_sy: 227, out_depth: 3});
    // conv1阶段
    // 卷积：使用96个11*11的过滤器，步长为4，经过relu的处理，生成激活像素层，会生成55*55*96个卷积后的像素层
    layer_defs.push({type: 'conv', sx: 11, filters: 96, stride: 4, activation: 'relu'});
    // 池化：池化运算的尺度为3*3，运算的步长为2，池化后像素的规模为27*27*96
    layer_defs.push({type: 'pool', sx: 3, stride: 2});
    // 归一化处理，归一化运算的尺度为5*5
    layer_defs.push({type:'lrn', k:2, n:5, alpha:0.0001, beta:0.75});

    // conv2阶段
    layer_defs.push({type: 'conv', sx: 5, filters: 256, stride: 1, pad: 2, group_size: 2,activation: 'relu'});
    layer_defs.push({type: 'pool', sx: 3, stride: 2});
    layer_defs.push({type:'lrn', k:2, n:5, alpha:0.0001, beta:0.75});

    // conv3阶段
    layer_defs.push({type: 'conv', sx: 3, filters: 384, stride: 1, pad: 1, activation: 'relu'});

    // conv4阶段
    layer_defs.push({type: 'conv', sx: 3, filters: 384, stride: 1, pad: 1, activation: 'relu'});

    // conv5阶段
    layer_defs.push({type: 'conv', sx: 3, filters: 256, stride: 1, pad: 1, activation: 'relu'});
    layer_defs.push({type: 'pool', sx: 3, stride: 2});

    // fc6阶段
    layer_defs.push({type: 'fc', num_neurons: 4096, activation: 'relu', drop_prob: 0.5});

    // fc7阶段
    layer_defs.push({type: 'fc', num_neurons: 4096, activation: 'relu', drop_prob: 0.5});

    // fc8阶段
    layer_defs.push({type: 'fc', num_neurons: 1000});
    layer_defs.push({type: 'softmax', num_classes: 2});

    // 创建一个神经网络
    const net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    // 创建一个trainer
    var trainer = new convnetjs.Trainer(net, {method: 'adadelta', l2_decay: 0.001, batch_size: 10});
    // trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});

    var visnet = document.getElementById('visnet');

    var catEle = document.createElement('img');
    catEle.src = 'images/cat.png';

    var preImage = function (img, callback) {
        if (img.complete) { // 如果图片已经存在于浏览器缓存，直接调用回调函数
            callback(img);
            return; // 直接返回，不用再处理onload事件
        }

        img.onload = function () { //图片下载完毕时异步调用callback函数。
            callback(img);//将回调函数的this替换为Image对象
        };
    };

    var show = function (img) {
        var cat = convnetjs.img_to_vol(img, 227);
        console.log(cat);

        // 对cat进行训练
        trainer.train(cat, 0);
        // var stats = trainer.train(cat, 0);

        for (var i = 0; i < net.layers.length; i++) {
            console.log(net.layers[i]);
        }

        var div = document.createElement('div');
        convnetjs.draw_activations_COLOR(div, net.layers[0].out_act, 1);
        visnet.appendChild(div);
        for (var i = 1; i < 10; i++) {
            visnet.appendChild(document.createTextNode(net.layers[i].layer_type));
            visnet.appendChild(document.createElement('br'));
            var div = document.createElement('div');
            convnetjs.draw_activations(div, net.layers[i].out_act, 1);
            visnet.appendChild(div);
        }
    }

    preImage(catEle, show);

})(convnetjs, document);