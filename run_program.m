% When difference between blend iterations is less than blend_stop_criteria, blending operation stops.
% If blend_stop_criterea is not reached after blend_max_iterations, then blending operation stops.
% Increase blend_max_iterations and decrease blend_stop_criteria for improved blending results.
blend_max_iterations = 2048;
blend_stop_criteria = 0.0001;

replace_faces( 'SampleSet/easy/', blend_max_iterations, blend_stop_criteria );
%replace_faces( 'SampleSet/hard/', blend_max_iterations, blend_stop_criteria );

%replace_faces( 'TestSet/blending/', blend_max_iterations, blend_stop_criteria );
%replace_faces( 'TestSet/pose/', blend_max_iterations, blend_stop_criteria );
%replace_faces( 'TestSet/more/', blend_max_iterations, blend_stop_criteria );