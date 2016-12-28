#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

void readTracks(char *filename, double *measurements, int frames, int tracks, int *labels, int labels_num);
void writeTracks(char *filename, double *measurements, int frames, int tracks, int *labels, int labels_num);
