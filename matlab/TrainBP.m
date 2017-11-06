% Input������
% Output����ǩ
% HideNum��������ڵ���Ŀ
% epochs����������
% lr:ѧϰ����.
function net = TrainBP(Input, Output, HideNum, epochs, lr)

vecErr = [];

[InNum, TrainNum] = size(Input);
OutNum = size(Output, 1);
sigmoid = @(x) 1./(1 + exp(-x));
alpha = 0.9;

% Ȩֵ��ƫ�ó�ʼ��Ϊ���ֵ
w1 = rand([HideNum InNum]);
b1 = rand([HideNum 1]);
w2 = rand([OutNum HideNum]);
b2 = rand([OutNum 1]);

for i = 1:epochs
    % �����������
    Index = randperm(TrainNum);
    Input = Input(:,Index);
    Output = Output(:,Index);
    
    nabla2 = zeros([OutNum HideNum]);
    nabla1 = zeros([HideNum InNum]);
    for j = 1:TrainNum
        Y = sigmoid(w1*Input(:,j)+b1); % ���������.HideNum*1
        Predict = sigmoid(w2*Y+b2); % OutNum*1
        delta2 = (Output(:,j) - Predict).*Predict.*(1-Predict);   % OutNum*1
        delta1 = w2'*delta2.*Y.*(1-Y);   % HideNum*1
        nabla2 = lr*(1-alpha)*delta2*Y' + alpha*nabla2; % OutNum*HideNum
        nabla1 = lr*(1-alpha)*delta1*Input(:,j)' + alpha*nabla1; % HideNum*InNum
        w2 = w2 + nabla2;
        w1 = w1 + nabla1;
        b2 = b2 + lr*(1-alpha)*delta2 + alpha*delta2;
        b1 = b1 + lr*(1-alpha)*delta1 + alpha*delta1;
    end

        Predict = sigmoid(w2*sigmoid(w1*Input+b1*ones([1 TrainNum]))+b2*ones([1 TrainNum]));
        mserror = 0.5*sum(mean((Predict-Output).^2,2));
        fprintf('Epoch:%d error:%f\n',i,mserror);
        vecErr = [vecErr mserror];
   
end
plot(vecErr);title('����������');
xlabel('��������');
ylabel('������');
%     Predict = sigmoid(w2*sigmoid(w1*Input+b1*ones([1 TrainNum]))+b2*ones([1 TrainNum]));
%     mserror = 0.5*sum(mean((Predict-Output).^2,2));
%     fprintf('Epoch:%d error:%f\n',i,mserror);

net.w1 = w1;
net.b1 = b1;
net.w2 = w2;
net.b2 = b2;
end