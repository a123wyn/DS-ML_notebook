function [ Q ] = AHP(A)
%QΪ��ָ���Ӧ��Ȩֵ��AΪ�ԱȾ���
[m,n]=size(A);
for i=1:n %�б�����Ƿ������ȫһ����
    for j=1:n
        if A(i,j)*A(j,i)~=1
        fprintf('no')
        end
    end
end
[x,y]=eig(A);%��A������ֵ���������������ҵ��������ֵ��Ӧ����������
t=max(y);
p=max(t);
c1=find(y(1,:)==max(t));
T=x(:,c1);%��������
q=zeros(n,1);%Ȩ
for i=1:n
    q(i,1)=T(i,1)/sum(T);
end
Q=q;
CI=(p-n)/(n-1);%�ж��Ƿ�ͨ��һ���Լ���
RI=[0,0,0.58,0.9,1.12,1.24,1.32,1.41,1.45,1.49,1.52,1.54,1.56,1.58,1.59];
CR=CI/RI(1,n);
if CR>=0.1
    fprintf('δͨ��һ���Լ���\n');
else
    fprintf('ͨ��һ���Լ���\n');
end
end

