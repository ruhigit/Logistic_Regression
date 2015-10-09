[name,values,train_data,train_label,test_data,test_label]=load_data();
[y_pclass,x_age,y_age,x_sibsp,y_sibsp,x_parch,y_parch,x_fare,y_fare]=monotonic_relationship(train_data,train_label);
[global_names,global_mi]=mutual_information(train_data,train_label);
[m_tr,m_te,s_tr,s_te,x_train,x_test]=missing_values(train_data,train_label,test_data,test_label);
[cols,X_train,X_test]=basis_expansion(x_train,x_test);
[y_tr,y_te,feature_set_train,feature_set_test]=sequential_feature_selection(X_train,train_label,X_test,test_label);
[b_acc]=batch_gradient_descent(feature_set_train,train_label);
[n_acc]=newton(feature_set_train,train_label);


%%% graphs%%
%1. Monotonic relationship
x=[1 2 3];
y=y_pclass;
subplot(4,2,1);
bar(x,y);
xlabel('pclass');
ylabel('Prob(Surviving)');
subplot(4,2,2);
bar(x_age,y_age);
xlabel('age' );
ylabel( 'P(Surviving)');
subplot(4,2,3);
bar(x_sibsp,y_sibsp);
xlabel('sibsp' );
ylabel( 'P(Surviving)');
subplot(4,2,4);
bar(x_parch,y_parch);
xlabel('parch');
ylabel( 'P(Surviving)');
subplot(4,2,5);
bar(x_fare,y_fare);
xlabel('fare');
ylabel( 'P(Surviving)');
x=1:10;
subplot(4,2,6);
plot(x,y_tr);
ylabel('Training');
subplot(4,2,7);
plot(x,y_te);
ylabel('Testing');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp('===  Results ===');
disp('====a) Missing Values====');
for i=1:14
    disp(name{i});
    disp(values(i));
end
disp('====c) Mutual Information====');
[sorted,indices]=sort(global_mi,'descend');
for i=1:13
    disp(global_names{indices(i)});
    disp(sorted(i));
end
disp('=====d) Missing Values=====');
disp('==Multiple Models==');
disp('Training Accuracy');
disp(m_tr);
disp('Testing Accuracy');
disp(m_te);
disp('==Substituting Values==');
disp('Training Accuracy');
disp(s_tr);
disp('Testing Accuracy');
disp(s_te);
disp('==== Basis Expansion ====');
disp('No of columns=');
disp(cols);
step_sizes=[0.01 0.001 0.002 0.005];
iterations=[10 300 200 100];
disp('==== Batch Gradient Descent ====')
for i=1:4
    disp('step_size');
    disp(step_sizes(i));
    disp('Iterations');
    disp(iterations(i));
    disp('Training Accuracy');
    disp(b_acc(i));
end
disp('Newtons Method');
for i=1:4
    disp('Iterations:');
    disp(i);
    disp(n_acc(i));
end