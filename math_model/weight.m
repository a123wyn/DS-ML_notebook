function [ w ] = weight( x )
%wΪ���Ȩ��
[n,m]=size(x);
%��ȡ��ʼ���ݾ����ģ��n������m��ָ��
[X,ps]=mapminmax(x',0.1,1.1);
%���ݹ�һ������
X=X';

%�����j��ָ���£���i��������ռ����p(i,j)
for i=1:n
    for j=1:m
        p(i,j)=X(i,j)/sum(X(:,j));
    end
end

%�����j��ָ�����ֵe(j)
k=1/log(n);
for j=1:m
    e(j)=-k*sum(p(:,j).*log(p(:,j)));
end
d=ones(1,m)-e;%����Ϣ�������
w=d./sum(d);%��Ȩֵ
w=w';
X=X;
end

