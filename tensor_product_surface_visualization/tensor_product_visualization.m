%  Written 2019 by Ramon Witschi, ETH Computer Science BSc, for Bachelor Thesis @ CGL
%
%   STUART VORTEX DATASET
%
%   "dimX": 128,
%	"dimY": 128,
%	"dimZ": 128,
%
%	"minX": -6.0,
%	"minY": -2.0,
%	"minZ": 1.0,
%
%	"maxX": 3.0,
%	"maxY": 2.0,
%	"maxZ": 3.0,
%
%   "V0": "( 4.0*math.sinh(y)+4.0*math.cosh(y)-math.cos(x - z) ) / (4.0*math.cosh(y) - math.cos(x - z))",
%	"V1": "- math.sin(x - z) / (4.0*math.cosh(y) - math.cos(x - z))",
%	"V2": "1",

clc;
clear all;
close all;

color_less_0 = [0.1, 0.6, 1.0];
color_more_0 = [0.6, 1.0, 0.1];
color_equl_0 = [1.0, 0.2, 0.2];

add_to_output_string = "far";

xs = 1.5;
ys = -0.5;
z = 2.0;

dx = 1;
dy = 1;

xend = xs+dx;
yend = ys+dy;

[b1,b2,b3] = get_control_points(xs,ys,z,dx,dy);

xcoords = [xs, xs, xs, xs+0.5*dx, xs+0.5*dx, xs+dx, xs+dx, xs+0.5*dx, xs+dx];
ycoords = [ys, ys+0.5*dy, ys+dy, ys, ys+0.5*dy, ys+0.5*dy, ys, ys+dy, ys+dy];

X = xs:0.002:xend;
Y = ys:0.002:yend;

[X,Y] = meshgrid(X,Y);

N = size(X,1);
M = size(X,2);

F1 = zeros(N,M); 
F2 = zeros(N,M);
F3 = zeros(N,M);

APF1 = zeros(N,M); 
APF2 = zeros(N,M);
APF3 = zeros(N,M);

APFC1 = zeros(N,M,3);
APFC2 = zeros(N,M,3);
APFC3 = zeros(N,M,3);

FC1 = zeros(N,M,3);
FC2 = zeros(N,M,3);
FC3 = zeros(N,M,3);

INTAPF = [];
%INTAPFC = zeros(N,M,3);
INTF = [];
%INTFC = zeros(N,M,3);

eps_0 = 1E-3;
eps_1 = 1E-2;

for i = 1:N
    for j = 1:M

        x = X(i,j);
        y = Y(i,j);
        
        % The Bezier tensorproduct surface approximation
        % is assumed to be in the unit square to satisfy
        % the convex hull property.
        xrel = (x - xs)/dx;
        yrel = (y - ys)/dy;

        APF1(i,j) = compute_bi_quadratic_tensor_surface(xrel,yrel,b1);
        APF2(i,j) = compute_bi_quadratic_tensor_surface(xrel,yrel,b2);
        APF3(i,j) = compute_bi_quadratic_tensor_surface(xrel,yrel,b3);
        
        APFC1(i,j,:) = get_color(color_less_0,color_more_0,color_equl_0,eps_0,APF1(i,j));
        APFC2(i,j,:) = get_color(color_less_0,color_more_0,color_equl_0,eps_0,APF2(i,j));
        APFC3(i,j,:) = get_color(color_less_0,color_more_0,color_equl_0,eps_0,APF3(i,j));
        
        if ( abs(APF1(i,j) - APF2(i,j)) <= eps_1 && abs(APF1(i,j) - APF3(i,j)) <= eps_1 )
            INTAPF = [INTAPF; x,y,APF1(i,j)];
        end

        vxw = get_vxw(x,y,z);
        F1(i,j) = vxw(1);
        F2(i,j) = vxw(2);
        F3(i,j) = vxw(3);
        
        FC1(i,j,:) = get_color(color_less_0,color_more_0,color_equl_0,eps_0,vxw(1));
        FC2(i,j,:) = get_color(color_less_0,color_more_0,color_equl_0,eps_0,vxw(2));
        FC3(i,j,:) = get_color(color_less_0,color_more_0,color_equl_0,eps_0,vxw(3));
        
        if ( abs(F1(i,j) - F2(i,j)) <= eps_1 && abs(F1(i,j) - F3(i,j)) <= eps_1 )
            INTF = [INTF; x,y,F1(i,j)];
        end

    end
end

lines23 = get_control_point_lines_to_plot(xcoords,ycoords,b1);
lines31 = get_control_point_lines_to_plot(xcoords,ycoords,b2);
lines12 = get_control_point_lines_to_plot(xcoords,ycoords,b3);

nb_rows = 4;
nb_cols = 2;

f1 = figure(1);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.5, 0.5]);
plot_component(xcoords,ycoords,b1,lines23,X,Y,APF1,APFC1);
grid on;
saveas(f1,"stuart_vortex_tp_1_" + add_to_output_string + ".png");

f2 = figure(2);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.5, 0.5]);
plot_component(xcoords,ycoords,b2,lines31,X,Y,APF2,APFC2);
grid on;
saveas(f2,"stuart_vortex_tp_2_" + add_to_output_string + ".png");

f3 = figure(3);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.5, 0.5]);
plot_component(xcoords,ycoords,b3,lines12,X,Y,APF3,APFC3);
grid on;
saveas(f3,"stuart_vortex_tp_3_" + add_to_output_string + ".png");

f4 = figure(4);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.5, 0.5]);
mesh(X,Y,F1,FC1,'FaceAlpha',0.5);
grid on;
saveas(f4,"stuart_vortex_tp_4_" + add_to_output_string + ".png");

f5 = figure(5);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.5, 0.5]);
mesh(X,Y,F2,FC2,'FaceAlpha',0.5);
grid on;
saveas(f5,"stuart_vortex_tp_5_" + add_to_output_string + ".png");

f6 = figure(6);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.5, 0.5]);
mesh(X,Y,F3,FC3,'FaceAlpha',0.5);
grid on;
saveas(f6,"stuart_vortex_tp_6_" + add_to_output_string + ".png");

subplot(nb_rows,nb_cols,1);
plot_component(xcoords,ycoords,b1,lines23,X,Y,APF1,APFC1);

subplot(nb_rows,nb_cols,3);
plot_component(xcoords,ycoords,b2,lines31,X,Y,APF2,APFC2);

subplot(nb_rows,nb_cols,5);
plot_component(xcoords,ycoords,b3,lines12,X,Y,APF3,APFC3);

subplot(nb_rows,nb_cols,2);
mesh(X,Y,F1,FC1,'FaceAlpha',0.5);

subplot(nb_rows,nb_cols,4);
mesh(X,Y,F2,FC2,'FaceAlpha',0.5);

subplot(nb_rows,nb_cols,6);
mesh(X,Y,F3,FC3,'FaceAlpha',0.5);

