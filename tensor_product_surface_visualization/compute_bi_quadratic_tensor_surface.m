function fxyb = compute_bi_quadratic_tensor_surface(x,y,b)
%COMPUTE_THE_8_SURFACES Summary of this function goes here
%   Detailed explanation goes here

    N = 2;
    M = 2;

    fxyb = 0;        
    index = 0;
    for i = 0:N
        for j = 0:M
            index = index + 1;
            fxyb = fxyb + bi_quadratic_tensor_entry(2,i,j,x,y,b(index));
        end
    end
end

