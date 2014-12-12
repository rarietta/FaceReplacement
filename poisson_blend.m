function [ blended_img ] = poisson_blend( img, grad_x, grad_y, mask )
%POISSON_BLEND Blend image using method described in "Poisson Image Editing" [Perez et all. 2003].
%              Uses the Jacobi Method to solve a sparse linear system.
%              Implementation inspired by Chris Tralie: http://www.ctralie.com/Teaching/PoissonImageEditing/
%              Implementation also inspired by Masayuki Tanaka: http://www.mathworks.com/matlabcentral/fileexchange/37224-poisson-image-editing
%   Parameters: img, input image.
%               grad_x, horizontal gradient.
%               grad_y, vertical gradient.
%               mask, defines region to blend.
%   Returns: blended_img, blended result.

% Check for invalid number of arguments.
if nargin ~= 4
    error( 'poisson_blend requires exactly 4 arguments.' );
else
    % Kernel to identify four orthogonal neighbors per pixel.
    neighbor_kernel = [ 0, 1, 0; 1, 0, 1; 0, 1, 0 ];
    laplacian = circshift( grad_x, [ 0, 1 ] ) + circshift( grad_y, [ 1, 0 ] ) - grad_x - grad_y;
    mask_selection_vector = ( mask > 0 );
    
    prev_max_diff = 1E32;
    curr_blend = img;
    prev_blend = img;
    
    for i = 1:2048
        neighbor_sums = imfilter( curr_blend, neighbor_kernel, 'replicate' );
        curr_blend( mask_selection_vector ) = ( laplacian( mask_selection_vector ) + neighbor_sums( mask_selection_vector ) ) / 4;

        blend_diff = abs( curr_blend - prev_blend );
        curr_max_diff = max( blend_diff( : ) );
        
        % DEBUG.
%         fprintf( '%d %g %g\n', i, curr_max_diff, ( prev_max_diff - curr_max_diff ) / prev_max_diff );

        if ( ( prev_max_diff - curr_max_diff ) / prev_max_diff < 1.0e-8 )
            break;
        end

        prev_blend = curr_blend;
        prev_max_diff = curr_max_diff;
    end
    
    blended_img = curr_blend;
end

end