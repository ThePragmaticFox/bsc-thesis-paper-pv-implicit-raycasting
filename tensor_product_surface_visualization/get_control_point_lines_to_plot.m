function lines = get_control_point_lines_to_plot(x,y,b)
%GET_LINES Summary of this function goes here
%   Detailed explanation goes here

    lines = zeros(3,3,6);
    lines(:,:,1) = [x(1),y(1),b(1); x(2),y(2),b(2); x(3),y(3),b(3)];
    lines(:,:,2) = [x(1),y(1),b(1); x(4),y(4),b(4); x(7),y(7),b(7)];
    lines(:,:,3) = [x(3),y(3),b(3); x(8),y(8),b(8); x(9),y(9),b(9)];
    lines(:,:,4) = [x(7),y(7),b(7); x(6),y(6),b(6); x(9),y(9),b(9)];
    lines(:,:,5) = [x(2),y(2),b(2); x(5),y(5),b(5); x(6),y(6),b(6)];
    lines(:,:,6) = [x(4),y(4),b(4); x(5),y(5),b(5); x(8),y(8),b(8)];

end

