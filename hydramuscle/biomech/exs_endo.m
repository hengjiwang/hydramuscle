function [stress] = exs_endo(j)
    % Returns a string of endodermal external stress
    
    if mod(j, 20) == 0
        row = 1;
    else
        row = 21 - mod(j, 20);
    end
    
    stress = 'LenTen((intop';
    stress = [stress, num2str(row)];
    stress = [stress, '(1)/1[mm]+intop'];
    stress = [stress, num2str(row+1)];
    stress = [stress, '(1)/1[mm])/(4*pi))*f_scale_endo*int'];
    stress = [stress, num2str(j+200)];
    stress = [stress, '(t/1[s])[N/m^2]'];
    
end

