function vxw = get_vxw(x,y,z)
%CROSSPROD Summary of this function goes here
%   Detailed explanation goes here

    v = get_v(x,y,z);
    w = get_w(x,y,z);

    vxw = [ 
            v(2)*w(3) - v(3)*w(2);
            v(3)*w(1) - v(1)*w(3);
            v(1)*w(2) - v(2)*w(1);
    ];
end

