clear all
close all
clc

%% һ����������
[ftrain1,ftrain2,ftrain3,ftrain4,SpeciesTrain] = textread('train.txt','%f,%f,%f,%f,%f');
[ftest1,ftest2,ftest3,ftest4,SpeciesTest] = textread('test.txt','%f,%f,%f,%f,%f');

% ����
TrainIn = [ftrain1 ftrain2 ftrain3 ftrain4]';
TestIn = [ftest1 ftest2 ftest3 ftest4]';
% ��ǩ
TrainLabels = [[1;0;0]*ones([1 25]) [0;1;0]*ones([1 25]) [0;0;1]*ones([1 25])];

%% �������ݹ�һ��
TrainIn = mapminmax(TrainIn,-4,4);
TestIn = mapminmax(TestIn,-4,4);

%% ����ѵ��
NumHidden = 10;
NumIterate = 500;
lr = 0.04;
net = TrainBP(TrainIn, TrainLabels, NumHidden, NumIterate, 0.04);

%% �ġ�����
sigmoid = @(x) 1./(1 + exp(-x));
Predict = sigmoid(net.w2*sigmoid(net.w1*TestIn+net.b1*ones([1 75]))+net.b2*ones([1 75]));
cnt = 0;
for i = 1:75
    if SpeciesTest(i) == find(Predict(:,i) == max(Predict(:,i)))
        cnt = cnt + 1;
    end
end
fprintf('Correct predict:%d\nAccuracy:%f\n',cnt,cnt/75);