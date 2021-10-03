function plot_component(x,y,b,lines,X,Y,APF,C)
%PLOT_COMPONENT Summary of this function goes here
%   Detailed explanation goes here
    plot3(x, y, b, 'o', 'MarkerFaceColor', 'black', "MarkerSize", 10);
    hold on;
    for i = 1:6
       plot3(lines(:,1,i),lines(:,2,i),lines(:,3,i), 'black', "LineWidth", 3); 
    end
    mesh(X,Y,APF,C,'EdgeAlpha',0.5, "FaceAlpha", 0.5);
end

