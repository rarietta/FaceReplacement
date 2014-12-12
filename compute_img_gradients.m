function [ grad_x grad_y ] = compute_img_gradients( img )
%COMPUTE_IMG_GRADIENTS Compute both horizontal and vertical gradients of input image.
%                      The gradient is essentially the change in intensity between neighboring pixels.
%   Parameters: img, input image.
%   Returns: grad_x, horizontal gradient.
%            grad_y, vertical gradient.

% Check for invalid number of arguments.
if nargin ~= 1
    error( 'compute_img_gradients requires exactly 1 argument.' );
else
    kernel_x = [ 0, -1, 1 ];
    kernel_y = [ 0; -1; 1 ];
    
    grad_x = imfilter( img, kernel_x, 'replicate' );
    grad_y = imfilter( img, kernel_y, 'replicate' );
end
    
end