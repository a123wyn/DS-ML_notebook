function [ w ] = weight( x )
%w为输出权重
[n,m]=size(x);
%读取初始数据矩阵规模，n个对象，m个指标
[X,ps]=mapminmax(x',0.1,1.1);
%数据归一化处理
X=X';

%计算第j个指标下，第i个对象所占比重p(i,j)
for i=1:n
    for j=1:m
        p(i,j)=X(i,j)/sum(X(:,j));
    end
end

%计算第j个指标的熵值e(j)
k=1/log(n);
for j=1:m
    e(j)=-k*sum(p(:,j).*log(p(:,j)));
end
d=ones(1,m)-e;%求信息熵冗余度
w=d./sum(d);%求权值
w=w';
X=X;
end

