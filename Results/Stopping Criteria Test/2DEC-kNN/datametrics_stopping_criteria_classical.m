close all;
clear; 
clc;


% filename = 'data_2_7.txt';
% filename = 'data_6_6.txt';
% filename = 'data_8_6.txt';
filename = 'data_10_7.txt';

f = fopen(filename);
data = textscan(f,'%s');
fclose(f);
variable = str2double(data{1}(1:1:end));
resultsdata_init = zeros(800,101);

for i = 1:800
    resultsdata_init(i,1) = variable(101*i-100);
    resultsdata_init(i,2:51) = variable(101*i-99:101*i-50);
    resultsdata_init(i,52:101) = variable(101*i-49:101*i);
    
end

%% Reducing result data 

resultsdata = zeros(8,101); 

for j = 1:8
    
   index1 = 100*(j-1) + 1;
   index2 = 100*j;
   
   accuracies = resultsdata_init(index1:index2, 2:51);
   stopping_probs = resultsdata_init(index1:index2, 52:101);
   
   
   resultsdata(j,1) = resultsdata_init(index1, 1);
   resultsdata(j,2:51) = mean(accuracies);
   resultsdata(j,52:101) = mean(stopping_probs);
   
end


%% Calculating data metrics 

% Data Metrics 
% # no. of points 
% # maximum accuracy from 1- 50 iterations
% # iteration number at which maximum accuracy was achieved


datametric = zeros(8,3);

for j = 1:8
    

   accuracies = resultsdata(j,2:51);
    
   
   datametric(j,1) = resultsdata(j, 1);
   [datametric(j,2) , datametric(j,3)] = max(accuracies);
    
   
end

datametric_ahead = datametric; 

datametric(:,2) = round(datametric(:,2),3);
datametric(:,3) = round(datametric(:,3),3);


%% Plotting  accuracy and probability of stopping for each number of points

figure
% labels = strings;
for k = 1:8
    
    
    plot(resultsdata(k,2:51))
    hold on 
    scatter(datametric(k,3), datametric(k,2), 'O')
    hold on
    grid on
    toadd  = append(string(datametric(k,1))); 
%     labels = append(labels,newline,toadd);
    labelpoints(datametric(k,3), datametric(k,2), toadd)
    

end
%legend(legends)
xlabel('Iteration Number')
ylabel('Accuracy')


figure

labels = strings;
for l = 1:8
    
    
    plot(resultsdata(l,52:101))
    hold on
    toadd  = append(string(datametric(l,1))); 
    labels = append(labels,newline,toadd);

    hold on
    grid on

end

xlabel('Iteration Number')
ylabel('Probabilty of stopping')
% legend
legend(labels,'Location','best')
%% storing processed result 
%  filenamestore = 'processed_result_stopping_criteria_classical_2_7.txt'; 
% filenamestore = 'processed_result_stopping_criteria_classical_6_6.txt'; 
% filenamestore = 'processed_result_stopping_criteria_classical_8_6.txt'; 
filenamestore = 'processed_result_stopping_criteria_classical_10_7.txt'; 

writematrix(datametric,filenamestore,'Delimiter','\t');  
%  type processed_result_stopping_criteria_classical_2_7.txt;
% type processed_result_stopping_criteria_classical_6_6.txt;
% type processed_result_stopping_criteria_classical_8_6.txt;
type processed_result_stopping_criteria_classical_10_7.txt;

%  xlfilenamestore = 'processed_result_stopping_criteria_classical_2_7.xlsx'; 
% xlfilenamestore = 'processed_result_stopping_criteria_classical_6_6.xlsx'; 
% xlfilenamestore = 'processed_result_stopping_criteria_classical_8_6.xlsx'; 
xlfilenamestore = 'processed_result_stopping_criteria_classical_10_7.xlsx'; 

xlswrite(xlfilenamestore,datametric);

