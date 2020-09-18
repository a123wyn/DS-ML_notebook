function [ Q ] = AHP(A)
%Q为各指标对应的权值，A为对比矩阵
[m,n]=size(A);
for i=1:n %判别矩阵是否具有完全一致性
    for j=1:n
        if A(i,j)*A(j,i)~=1
        fprintf('no')
        end
    end
end
[x,y]=eig(A);%求A的特征值，特征向量，并找到最大特征值对应的特征向量
t=max(y);
p=max(t);
c1=find(y(1,:)==max(t));
T=x(:,c1);%特征向量
q=zeros(n,1);%权
for i=1:n
    q(i,1)=T(i,1)/sum(T);
end
Q=q;
CI=(p-n)/(n-1);%判断是否通过一致性检验
RI=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59];
CR=CI/RI(1,n);
if CR>=0.1
    fprintf('未通过一致性检验\n');
else
    fprintf('通过一致性检验\n');
end
end

