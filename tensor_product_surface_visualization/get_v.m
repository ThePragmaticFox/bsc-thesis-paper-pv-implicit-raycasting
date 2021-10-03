function v = get_v(x,y,z)
%v Summary of this function goes here
%   Detailed explanation goes here
    v = [(-cos(x - z) + 4.* cosh(y) + 4.* sinh(y))/(-cos(x - z) + 4.* cosh(y)); -(sin(x - z)/(-cos(x - z) + 4.* cosh(y))); 1];
end

