[W,L] = read_tracks_mex('Tracks10.dat');

Z = W ~= 0;

Z = Z(1:2:end,:);

Z = double(Z);

write_tracks_mex('Tracks10_write.dat',W,Z,L);
