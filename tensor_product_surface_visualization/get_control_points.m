function [b1,b2,b3] = get_control_points(x,y,z,dx,dy)
%GET_CONTROL_POINTS Summary of this function goes here
%   Detailed explanation goes here

    b = zeros(9,3);

    v00 = get_v(x,y,z);
    v01 = get_v(x,y+dy,z);
    v10 = get_v(x+dx,y,z);
    v11 = get_v(x+dx,y+dy,z);

    w00 = get_w(x,y,z);
    w01 = get_w(x,y+dy,z);
    w10 = get_w(x+dx,y,z);
    w11 = get_w(x+dx,y+dy,z);
    
    vxw0000 = cross(v00,w00);
    vxw0001 = cross(v00,w01);
    vxw0010 = cross(v00,w10);
    vxw0011 = cross(v00,w11);
    vxw0100 = cross(v01,w00);
    vxw0101 = cross(v01,w01);
    vxw0110 = cross(v01,w10);
    vxw0111 = cross(v01,w11);
    vxw1000 = cross(v10,w00);
    vxw1001 = cross(v10,w01);
    vxw1010 = cross(v10,w10);
    vxw1011 = cross(v10,w11);
    vxw1100 = cross(v11,w00);
    vxw1101 = cross(v11,w01);
    vxw1110 = cross(v11,w10);
    vxw1111 = cross(v11,w11);
    
    for i = 1:3
        
        b(1,i) = vxw0000(i);
        b(3,i) = vxw0101(i);
        b(7,i) = vxw1010(i);
        b(9,i) = vxw1111(i);
        
        b(2,i) = 0.5*( vxw0001(i) + vxw0100(i) );
        b(4,i) = 0.5*( vxw0010(i) + vxw1000(i) );
        b(6,i) = 0.5*( vxw0111(i) + vxw1101(i) );
        b(8,i) = 0.5*( vxw1011(i) + vxw1110(i) );
        
        b(5,i) = 0.25*( vxw0011(i) + vxw0110(i) + vxw1001(i) + vxw1100(i) );
        
    end
    
    b1 = b(:,1);
    b2 = b(:,2);
    b3 = b(:,3);

end

