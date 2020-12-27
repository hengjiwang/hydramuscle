function [stress] = exs_ecto(j)
    % Returns a string of endodermal external stress
    
    stress = ['f_scale_ecto*int', num2str(j), '(t/1[s])'];
    
end

