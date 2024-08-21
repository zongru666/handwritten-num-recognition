
%Fisher决策手写数字识别%简化代码
%用时2分30秒
 %            *********Fisher()实现************
load('mnist_test.mat')
load('mnist_train.mat')
%function  [Rchar]=Fisher(lett)  %Fisher函数
 %训练集特征数据集处理
 step=1 %第1步
 %求各类样本最后一个在y中的行数
 a=1;
 b(1)=0;
 for n=1:9
    while (y(a,1)<=(n-1))
          a=a+1;
    end;
    b(n+1)=a-1;
 end;
 %b(10)=60000;
 
 step=step+1 %第2步
 %通过上下拼接，将特征向量变为矩阵
 for n=1:60000
	imagecell{n}=reshape(X(n,:),[28 28])';     %1*60000个28*28的矩阵(右上角的'转置) 
 end;

 step=step+1 %第3步
 %剪切图片（去掉数字的四周）
 for j=1:60000
     imagecell{j}(all(imagecell{j}==0,2),:)=[];  %将全0的每一行去掉
     imagecell{j}(:,all(imagecell{j}==0,1))=[];  %将全0的每一列去掉
	 imagecell2{j}=imresize(imagecell{j},[10 10]); %放缩为10*10的矩阵
	 imagecell3{j}=reshape(imagecell2{j}',[1 100]); %将矩阵转置后转为行向量
 end

 step=step+1 %第4步
 %将剪切后的每个数字样本分类（每类只取5000个样本）
 for j=1:10
     for i=1:5000
         num{j}(i,:)=imagecell3{i+b(j)}; %每一行为一个样本
     end
end

step=step+1 %第5步
%原样本空间
%计算类均值向量meani
for i=1:10
    meani{i}=mean(num{i});
end

step=step+1 %第6步
%求类内离散度矩阵Si
for j=1:10
    Si{j}=0;  %样本类内离散度
    for i=1:5000    
	    Si{j}=Si{j}+(num{j}(i,:)-meani{j})'*(num{j}(i,:)-meani{j});  %累加
    end
end

step=step+1 %第7步
%%对这10类分别进行两两类识别比较
%求两两类的总类内离散度、类间离散度(45种情况)
Sw=cell(10,10);
Sb=cell(10,10);
for i=1:9
    for j=i+1:10
        Sw{i,j}=Si{i}+Si{j};   %总类内离散度
        Sb{i,j}=(meani{i}-meani{j})'*(meani{i}-meani{j});   %类间离散度
    end
end

%*******************************************
%***********测试集特征数据集处理************
%******************************************* 
 step=step+1 %第8步
 %求各类样本最后一个在Ty中的行数
 Ta=1;
 Tb(1)=0;
 for Tn=1:9
    while (Ty(Ta,1)<=(Tn-1))
          Ta=Ta+1;
    end;
    Tb(Tn+1)=Ta-1;
 end;
 %b(10)=10000;
 
 step=step+1 %第9步
 %通过上下拼接，将特征向量变为矩阵
 for Tn=1:10000
	Timagecell{Tn}=reshape(TX(Tn,:),[28 28])';     %1*60000个28*28的矩阵(右上角的'转置) 
 end;

 step=step+1 %第10步
 %剪切图片（去掉数字的四周）
 for Tj=1:10000
     Timagecell{Tj}(all(Timagecell{Tj}==0,2),:)=[];  %将全0的每一行去掉
     Timagecell{Tj}(:,all(Timagecell{Tj}==0,1))=[];  %将全0的每一列去掉
	 Timagecell2{Tj}=imresize(Timagecell{Tj},[10 10]); %放缩为10*10的矩阵
	 Timagecell3{Tj}=reshape(Timagecell2{Tj}',[1 100]); %将矩阵转置后转为行向量
 end
 
 step=step+1 %第11步
 %将剪切后的每个数字样本分类（每类只取500个样本）
 for Tm=1:10
     for Ti=1:500
     Tnum{Tm}(Ti,:)=Timagecell3{Ti+Tb(Tm)}; %每一行为一个测试样本
	 end
 end

 step=step+1 %第12步
%判别分类
for p=1:10
    %测试某一类500个数据
    for Tk=1:500
        %求两两类判别的阈值，判别函数，最优投影方向。
        W=cell(10,10);
        Gx=cell(10,10);
        for i=1:9
            for j=i+1:10
                Sw{i,j}=Sw{i,j}+0.0001*eye(100);       %处理总类内离散度矩阵
                W{i,j}=inv(Sw{i,j})*(meani{i}-meani{j})';     %最优投影方向
                Gx{i,j}=(W{i,j}')*(Tnum{p}(Tk,:)-0.5*(meani{i}+meani{j}))';
     	    end
	    end

        count=1;
        k=0;
        for i=count:9   %从第1类开始两两比较
            for j=(count+1):10
                if Gx{i,j}<0      %不属于i类，则转为从第i+1类开始比较
                   if count==9        %已经确定不是8就是9，则停止继续往下比较（count不再加1）
                      char=10;
                   else
                       count=count+1;     %转为第i+1类
                       k=0;
                       break;
                   end
                else
                    char=count;    %将当前类的序号赋值给变量char
                    k=k+1;     % 计算判定的次数
                end
            end
            if k==10-count      %若判定完则跳出循环
               break;
            end
        end
    Tchar(Tk) = char-1;    %识别结果
    end

    %计算识别个数
    Tcount=0;
    for Tt=1:500
        if Tchar(Tt)==p-1
           Tcount=Tcount+1;
	    end
    end
    Tc(p)=Tcount;  %存放每类识别正确的个数
end

step=step+1 %第13步
%求识别率（92.26%）
rate=Tc/500; %每类数字的识别率
%总的识别率
answer=sum(Tc,2)/5000;  %总的识别率=总的正确识别个数/总的测试样本个数
for i=1:10
    fprintf(num2str(i-1));  %显示类别
    fprintf('类测试样本得到的识别率是%d\n',rate(i));  %显示各类的识别率
end
fprintf('每类500个（共5000个）测试样本得到的识别率是%d\n',answer);

%end