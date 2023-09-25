close all;
clear; 
clc;

% v = [ 31.4162 26.8419 ]; 
% v = [54.9000 3.9747];
% v = [-122.2583 12.5888];
% v = [-97.875 17.4639594269209] ;

 filename = 'complete_processed_result_stereographic_overfitting_test_2_7.txt';
% filename = 'complete_processed_result_stereographic_overfitting_test_6_6.txt';
% filename = 'complete_processed_result_stereographic_overfitting_test_8_6.txt';
% filename = 'complete_processed_result_stereographic_overfitting_test_10_7.txt';

f = fopen(filename);
data = textscan(f,'%s');
fclose(f);
variable = str2double(data{1}(1:1:end));
resultsdata = zeros(120,3);

for i = 1:120

    resultsdata(i,1) = variable(15*i-14);
    resultsdata(i,2) = variable(15*i-13);
%     resultsdata(i,3) = variable(15*i-12); % Overfitting Parameter
%     resultsdata(i,3) = variable(15*i-11); % Training Accuracy
    resultsdata(i,3) = variable(15*i-9); % Testing Accuracy
%     resultsdata(i,3) = variable(15*i-7); % No. of iterations
%     resultsdata(i,3) = variable(15*i-3); % Training execution time
%     resultsdata(i,3) = variable(15*i-1); % Testing execution time
    
end

%% Surface plots 


% radii = log10(resultsdata(1:15,1)); 
% numpts = log10(resultsdata(1:15:120,2));
radii = resultsdata(1:15,1); 
numpts = resultsdata(1:15:120,2);
to_plot = reshape(resultsdata(:,3),[15,8]);


surfc(numpts, radii, to_plot)
% set(gca,'YScale','log')
set(gca,'XScale','log')
set(gca, 'Xdir', 'reverse')
xlabel('No. of points')
ylabel('Radius')
% zlabel('Overfitting Parameter')
% % zlim([15 25])
% zlabel('Training Accuracy (%)')
zlim([86 87.2])
zlabel('Testing Accuracy (%)')
% zlabel('No. of iterations')
% zlim([15 25])
% zticks([15 17 19 21 23 25])
set(gca, 'FontSize',15)
% zlabel('Training execution time (ms)')
% zlabel('Testing execution time (ms)')
ylim([1 5])

% clim([86 87])

grid on
hold on

% labels = strings; 
%  C = ["black"; "blue"; "green"; "cyan" ;"red"; "magenta" ;"yellow"; "white"];

% for iterator = 1:8
%         
% %     toadd  = string(resultsdata(16*iterator-15,2)); 
% %     labels = append(labels,newline,toadd);
%   
%     surface = resultsdata(15*iterator-14 : 15*iterator,3) ;
% %     surf(iterations,radii,surface,'FaceAlpha',0.5);
%     surf(iterations,radii,surface);
% %     surf(iterations,radii,surface,'FaceColor',C(iterator));
%     hold on
% end
% % legend('640','1280','2560','3200','6400','12800','25600','51200')
% 
% xlabel('No. of points')
% ylabel('log10(radius)')
% % zlabel('Probability of stopping')
% zlabel('No. of iterations')
% zlim([80.4 84.2])
% ylim([0 2])
% xlim([0 50])
