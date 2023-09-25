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
resultsdata = zeros(128,102);

for i = 1:128
    resultsdata(i,1) = variable(102*i-101);
    resultsdata(i,2) = variable(102*i-100);
    resultsdata(i,3:52) = variable(102*i-99:102*i-50);
    resultsdata(i,53:102) = variable(102*i-49:102*i);
    
end

% Data Metrics
% # radius 
% # no. of points 
% # maximum accuracy from 1- 50 iterations
% # iteration number at which maximum accuracy was achieved


datametric = zeros(128,4);

for j = 1:128
    

   accuracies = resultsdata(j,3:52);
    
   
   datametric(j,1) = resultsdata(j, 1);
   datametric(j,2) = resultsdata(j, 2);
   [datametric(j,3) , datametric(j,4)] = max(accuracies);
    
   
end

datametric_ahead = datametric; 

datametric(:,3) = round(datametric(:,3),3);
datametric(:,4) = round(datametric(:,4),3);


%% Plotting  accuracy and probability of stopping for each radius and number of points

% figure
% % labels = strings;
% for k = 113:128
%     
%     
%     plot(resultsdata(k,3:52))
%     hold on 
%     scatter(datametric(k,4), datametric(k,3), 'O')
%     hold on
%     grid on
%     toadd  = append(string(datametric(k,1))," ",string(datametric(k,2))); 
% %     labels = append(labels,newline,toadd);
%     labelpoints(datametric(k,4), datametric(k,3), toadd)
%     
% 
% end
% %legend(legends)
% xlabel('Iteration Number')
% ylabel('Accuracy')

% 
% figure
% 
% labels = strings;
% for l = 113:128
%     
%     
%     plot(resultsdata(l,53:102))
%     hold on
% %     toadd  = append(string(datametric(l,1))," ",string(datametric(l,2))); 
% %     labels = append(labels,newline,toadd);
% 
%     hold on
%     grid on
% 
% end
% legend
% xlabel('Iteration Number')
% ylabel('Probabilty of stopping')
% legend(labels,'Location','best','NumColumns',16)


%% Surface plots 
iterations = 1:50;
labels = strings; 
 C = ["black"; "blue"; "green"; "cyan" ;"red"; "magenta" ;"yellow"; "white"];

for iterator = [1 4 8]
        figure
    toadd  = string(resultsdata(16*iterator-15,2)); 
    labels = append(labels,newline,toadd);
    radii = log10(resultsdata(16*iterator-15 : 16*iterator , 1)); 
    surface = resultsdata(16*iterator-15 : 16*iterator,53:102) ;
%     surf(iterations,radii,surface,'FaceAlpha',0.5);
    surf(iterations,radii,surface);
    
xlabel('Iteration Number')
ylabel('log10(Radius of stereographic projection)')
zlabel('Probability of stopping')
    hold on
end
% legend('640')

xlabel('Iteration Number')
ylabel('log10(Radius of stereographic projection)')
zlabel('Probability of stopping')
% zlim([80.5 83.5])
% ylim([0 2])
% xlim([0 50])
% 
% radii2 = log10(resultsdata(1:16,1)); 
% surface2 = resultsdata(1:16,3:52) ;
% surf(iterations,radii2,surface2); 

% %% storing complete processed result 
% %  filenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_2_7.txt'; 
% % filenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_6_6.txt'; 
% % filenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_8_6.txt'; 
% filenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_10_7.txt'; 
% 
% writematrix(datametric,filenamestore,'Delimiter','\t');  
% %  type complete_processed_result_stopping_criteria_3d_stereo_short_2_7.txt;
% % type complete_processed_result_stopping_criteria_3d_stereo_short_6_6.txt;
% % type complete_processed_result_stopping_criteria_3d_stereo_short_8_6.txt;
% type complete_processed_result_stopping_criteria_3d_stereo_short_10_7.txt;
% 
% %  xlfilenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_2_7.xlsx'; 
% % xlfilenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_6_6.xlsx'; 
% % xlfilenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_8_6.xlsx'; 
% xlfilenamestore = 'complete_processed_result_stopping_criteria_3d_stereo_short_10_7.xlsx'; 
% 
% xlswrite(xlfilenamestore,datametric);
% 
% 
% 
% %% Further processing of results
% 
% datametric_acc = datametric_ahead;
% 
% for k = 1:8
%     
%    index3 = 16*(k-1) + 1;
%    index4 = 16*k;
%    A = datametric_acc(index3:index4, :);
%    A = sortrows(A,3,'descend');
%    datametric_acc(index3:index4, :) = A;
%    
% end
% 
% 
% datametric_acc(:,3) = round(datametric_acc(:,3),3);
% datametric_acc(:,4) = round(datametric_acc(:,4),3);
% 
% 
% winner_acc = zeros(8,4);
% 
% for m = 1:8
%     index2 = 16*(m-1)+1;
%     winner_acc(m,:) = datametric_acc(index2,:);
% end
% 
% %% storing processed result sorted by accuracy
% %  filenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_2_7.txt'; 
% % filenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_6_6.txt'; 
% % filenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_8_6.txt'; 
% filenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_10_7.txt'; 
% % 
% writematrix(winner_acc,filenamestore,'Delimiter','\t');  
% %  type accuracy_processed_result_stopping_criteria_3d_stereo_short_2_7.txt;
% % type accuracy_processed_result_stopping_criteria_3d_stereo_short_6_6.txt;
% % type accuracy_processed_result_stopping_criteria_3d_stereo_short_8_6.txt;
% type accuracy_processed_result_stopping_criteria_3d_stereo_short_10_7.txt;
% 
% %  xlfilenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_2_7.xlsx'; 
% % xlfilenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_6_6.xlsx'; 
% % xlfilenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_8_6.xlsx'; 
% xlfilenamestore = 'accuracy_processed_result_stopping_criteria_3d_stereo_short_10_7.xlsx'; 
% 
% xlswrite(xlfilenamestore,winner_acc);
% 
% % % 
% alphabet = round(alphabet,3);
% labels = arrayfun(@(x) sprintf('(%g + %gi)', real(x), imag(x)), alphabet, 'UniformOutput', false);
% scatter(real(alphabet),imag(alphabet),'filled');
% labelpoints(real(alphabet),imag(alphabet),labels,'S',0.15)
% grid on
% grid minor
% xlabel('real')
% ylabel('imag')
