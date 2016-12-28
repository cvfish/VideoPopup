#include <stdio.h>
#include <vector>
#include <iostream>
#include <string.h>

using namespace std;

class CPoint
{
public:
	CPoint() {}
	float x,y,frame;
};

class CSimpleTrack
{
public:
	CSimpleTrack() {};
	int mLabel;
	vector<CPoint> mPoints;
};

class CTracksIO
{
public:
	void writeTracks(char *filename, vector<CSimpleTrack>& mTracks, int& mSequenceLength);
	void readTracks(char *filename, vector<CSimpleTrack>& mTracks, int& mSequenceLength);
public:
	int a;
	int b;
};
