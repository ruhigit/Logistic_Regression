function[y_tr,y_te,feature_set,feature_set_te]=sequential_feature_selection(x_train,train_label,x_test,test_label)

%%greedy forward selection. Iteratively build up a set of 10 features. At each iteration add one column
%%to the feature set. Choose the column that works best in conjunction with the existing,
%%already-chosen features.

[no_rows,no_cols]=size(x_train);
training_accuracy=zeros(10,1);
feature_set=[];
feature_set_te=[];
features=[];
train_label=cell2mat(train_label(:,1));
test_label=cell2mat(test_label(:,1));
%the outer loop will run 10 times, adding a new feature each time.
for outer=1:10
    %The inner loop will train a new model as many times as there are
    %columns.
    for col=1:no_cols
        if(ismember(col,features))
            continue;
        end
        x_matrix=[feature_set x_train(:,col)];
        model= glmfit(x_matrix, train_label, 'binomial');
        y_hat = glmval(model, x_matrix, 'logit');
        %Trainig accuracy
        correct=accuracy(y_hat,train_label);
        %each time storing the training accuracy.
        accu(col)=correct/length(y_hat);
    end
    [value,feature]=max(accu);
    features(outer)=feature;
    feature_set=[feature_set x_train(:,feature)];
    feature_set_te=[feature_set_te x_test(:,feature)];
    training_accuracy(outer)=value;
    %Testing accuracy
    model= glmfit(feature_set, train_label, 'binomial');
    y_hat = glmval(model, feature_set_te, 'logit');
    correct=accuracy(y_hat,test_label);
    testing_accuracy(outer)=correct/length(y_hat);
    
end


y_tr=training_accuracy();

y_te=testing_accuracy();


function[correct]=accuracy(y_hat,train_label)
correct=0;
for i=1:length(y_hat)
            class = y_hat(i) > 0.5;
            if(class==train_label(i))
                correct=correct+1;
            end
end
