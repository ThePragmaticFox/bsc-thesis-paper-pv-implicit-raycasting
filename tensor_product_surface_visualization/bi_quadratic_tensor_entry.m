function result = bi_quadratic_tensor_entry(n,i,j,x,y,b)
%TENSOR_PRODUCT_SURFACE Summary of this function goes here
%   Detailed explanation goes here
    bix = nchoosek(n,i)*x^(i)*(1-x)^(n-i);
    bjy = nchoosek(n,j)*y^(j)*(1-y)^(n-j);
    result = bix*bjy*b;
end