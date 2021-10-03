function C = get_color(color_less_0,color_more_0,color_equl_0,eps,f)
%GET_COLOR Summary of this function goes here
%   Detailed explanation goes here
    if f < -eps
        C = color_less_0; 
    elseif f > eps
        C = color_more_0;
    else
        C = color_equl_0;
    end
end

