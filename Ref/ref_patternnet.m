clear; close all; 

[x,t] = iris_dataset;
net = patternnet(1);
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y);


net.layers{1}.transferFcn 
net.layers{2}.transferFcn 
net.layers{3}.transferFcn 
