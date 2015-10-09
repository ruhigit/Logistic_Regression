function[accuracy]=batch_gradient_descent(train_data,train_label)


%remove missing values
[no_rows,no_cols]=size(train_data);
x_train=train_data;
for i=1:no_cols
    rows=find(~isnan(x_train(:,i)));
    x=x_train(rows(:),:);
    x_train=x;
end

[no_rows,no_cols]=size(x_train);
%need to add a column of ones
%to the beginning of the data when not using glmfit.
col1=ones(no_rows,1);
x_train=[col1 x_train];
[no_rows,no_cols]=size(x_train);

x=(x_train);
y=(cell2mat(train_label(:,1)));
theta=zeros(no_cols,1);
step_sizes=[0.01 0.001 0.002 0.005];
iterations=[10 300 200 100];
%repeat until converges
for i1=1:4
    step_size=step_sizes(i1);
    iter=iterations(i1);
    for outer=1:iter
        output=mysigmoid(x*theta);
        accuracy(i1)=cal_acc(y,output);
        %loss=cal_loss(x,y,theta);
        sig=mysigmoid(x(1,:)*theta);
        gradient=(sig-y(1)).*x(1,:);
        for i=2:length(x)
            sig=mysigmoid(x(i,:)*theta);
            gradient = gradient+(sig-y(i)).*x(i,:);
        end
    theta=theta-(step_size.*gradient)';
    end
end

function[answer]=mysigmoid(a)
answer=1./(1+exp(-a));

%function[loss]=cal_loss(x,y,theta)
%loss=y'*log(mysigmoid(x*theta))+(1-y')*log(1-mysigmoid(x*theta));

function[acc]=cal_acc(train_label,y_hat)
correct=0;
for i=1:length(y_hat)
            class = y_hat(i) > 0.5;
            if(class==train_label(i))
                correct=correct+1;
            end
end
acc=correct/length(y_hat);
