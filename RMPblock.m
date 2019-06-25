function LayerGraph = RMPblock(LayerGraph, InputLayerName)
% create RMP block, based on CEnet
% This code was written by Mr zhipan Wang, if you have any question, Please
% contact me,Email:1044625113@qq.com,@BeiJing,China Remote Sensing
% Application Center,2019-6-24

% input: 14*14*512 in original paper
% output: 14*14*516 in original paper

% InputLayerName: layer name to connect!

stride_size = [2, 3, 5, 6]; % stride_size: conv kernel size, the author's code suggested  value!


% 2*2 pool layer
maxPoolLayer1 = maxPooling2dLayer(2, 'Stride', stride_size(1), 'Padding','same', 'Name','RMP_maxPool_1'); % origianl pool size is 2*2

RMP_conv1 = convolution2dLayer(1,1,...  % use 1*1 conv to the number of parameter
    'Stride',[1 1],...
    'Padding','same',...
    'BiasL2Factor',0,...
    'Name','RMP_conv1');

deconvLayer1 = transposedConv2dLayer(1,1,...
    'Stride',stride_size(1),...
    'Cropping','same',...
    'BiasL2Factor',0,...
    'Name','RMP_deconv_1');

RMP_crop1 = crop2dLayer('centercrop','Name','RMP_crop1'); % make output layer the same size as input layer

net1 = [maxPoolLayer1; RMP_conv1; deconvLayer1; RMP_crop1];
LayerGraph = addLayers(LayerGraph, net1);
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_crop1/ref');


% 3*3 pool layer
maxPoolLayer2 = maxPooling2dLayer(3, 'Stride', stride_size(2), 'Padding','same', 'Name','RMP_maxPool_2');

RMP_conv2 = convolution2dLayer(1,1,...  % use 1*1 conv to the number of parameter
    'Stride',[1 1],...
    'Padding','same',...
    'BiasL2Factor',0,...
    'Name','RMP_conv2');

deconvLayer2 = transposedConv2dLayer(1,1,...
    'Stride',stride_size(2),...
    'Cropping','same',...
    'BiasL2Factor',0,...
    'Name','RMP_deconv_2');

RMP_crop2 = crop2dLayer('centercrop','Name','RMP_crop2'); % make output layer the same size as input layer

net2 = [maxPoolLayer2; RMP_conv2; deconvLayer2; RMP_crop2];
LayerGraph = addLayers(LayerGraph, net2);
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_crop2/ref');

% 5*5 pool layer
maxPoolLayer3 = maxPooling2dLayer(5, 'Stride', stride_size(3), 'Padding','same', 'Name','RMP_maxPool_3');

RMP_conv3 = convolution2dLayer(1,1,...  % use 1*1 conv to the number of parameter
    'Stride',[1 1],...
    'Padding','same',...
    'BiasL2Factor',0,...
    'Name','RMP_conv3');

deconvLayer3 = transposedConv2dLayer(1,1,...
    'Stride',stride_size(3),...
    'Cropping','same',...
    'BiasL2Factor',0,...
    'Name','RMP_deconv_3');

RMP_crop3 = crop2dLayer('centercrop','Name','RMP_crop3'); % make output layer the same size as input layer

net3 = [maxPoolLayer3; RMP_conv3; deconvLayer3; RMP_crop3];
LayerGraph = addLayers(LayerGraph, net3);
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_crop3/ref');


% 6*6 pool layer
maxPoolLayer4 = maxPooling2dLayer(6, 'Stride', stride_size(4), 'Padding','same', 'Name','RMP_maxPool_4');

RMP_conv4 = convolution2dLayer(1,1,...  % use 1*1 conv to the number of parameter
    'Stride',[1 1],...
    'Padding','same',...
    'BiasL2Factor',0,...
    'Name','RMP_conv4');

deconvLayer4 = transposedConv2dLayer(1,1,...
    'Stride',stride_size(4),...
    'Cropping','same',...
    'BiasL2Factor',0,...
    'Name','RMP_deconv_4');

RMP_crop4 = crop2dLayer('centercrop','Name','RMP_crop4'); % make output layer the same size as input layer

net4 = [maxPoolLayer4; RMP_conv4; deconvLayer4; RMP_crop4];
LayerGraph = addLayers(LayerGraph, net4);
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_crop4/ref');



% 完成连接
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_maxPool_1');
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_maxPool_2');
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_maxPool_3');
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_maxPool_4');


% 特征融合
add = depthConcatenationLayer(5,'Name', 'RMP_cat');
LayerGraph = addLayers(LayerGraph, add);

LayerGraph = connectLayers(LayerGraph, 'RMP_crop1', 'RMP_cat/in1');
LayerGraph = connectLayers(LayerGraph, 'RMP_crop2', 'RMP_cat/in2');
LayerGraph = connectLayers(LayerGraph, 'RMP_crop3', 'RMP_cat/in3');
LayerGraph = connectLayers(LayerGraph, 'RMP_crop4', 'RMP_cat/in4');
LayerGraph = connectLayers(LayerGraph, InputLayerName, 'RMP_cat/in5');


end

