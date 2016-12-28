#include "mex.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>

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

void writeTracks(char *filename, std::vector<CSimpleTrack>& mTracks, int& mSequenceLength)
{
	// Create a filename and open a file
	// 	char buffer[50];
	// 	sprintf(buffer,"Tracks%d.dat",mSequenceLength);
	// 	std::ofstream aFile(buffer);
	ofstream aFile(filename);
	// write number of frames considered
	aFile<< mSequenceLength << std::endl;
	// write number of tracks
	int aCount = 0;
	for(int i = 0; i < mTracks.size(); i++)
		if (mTracks[i].mLabel >= 0) aCount++;
	aFile<<aCount<<std::endl;
	// write each track
	for(int i = 0; i < mTracks.size(); i++)
	{
		// Ignore tracks marked as outliers
		if (mTracks[i].mLabel < 0) continue;
		// write label and length of track
		int aSize = mTracks[i].mPoints.size();
		aFile<<mTracks[i].mLabel<<" "<<aSize<<std::endl;
		// write x,y coordinates and frame number of the tracked point
		for (int j = 0; j < aSize; j ++)
			aFile<< mTracks[i].mPoints[j].x<<" "<<mTracks[i].mPoints[j].y<<" "<<mTracks[i].mPoints[j].frame<<std::endl;
	}
}

void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *in[])
{
	// input parameters: name of the tracks file, measurement matrix , visibility matrix, label

	char*  filename;
	mwSize buflen;
	int status;

	/* Check for proper input type */
	if(! mxIsChar(in[0]) || mxGetM(in[0]) != 1)
	{
		mexErrMsgTxt("Input argument must be a string");
	}

	buflen = mxGetN(in[0])*sizeof(mxChar)+1;
	filename = (char *)mxMalloc(buflen);

	/* Copy the string data into buf. */
	status = mxGetString(in[0], filename, buflen);

	//mexPrintf("The input string is:  %s\n", filename);

	int frames = mxGetM(in[1])/2;    // # of frames
	int points = mxGetN(in[1]);       // # of points

	double* p2dPointsMatrix = new double[2*frames*points];

	for(int i = 0; i < frames; i++)
	{
		for(int j = 0; j < points; j++)
		{
			p2dPointsMatrix[2*i*points+j] = mxGetPr(in[1])[j*2*frames+2*i];
			p2dPointsMatrix[(2*i+1)*points+j] = mxGetPr(in[1])[j*2*frames+2*i+1];

		}
	}

	int *pVisibilityMatrix;
	pVisibilityMatrix = new int[frames*points];

	for(int i = 0; i < frames; ++i)
	{
		for(int j = 0; j < points; j++)
		{
			pVisibilityMatrix[i*points+j] = mxGetPr(in[2])[j*frames+i];
		}
	}

	int *label;
	label = new int[points];

	for(int i = 0; i < points; ++i)
	{
		label[i] = mxGetPr(in[3])[i];
	}


	vector<CSimpleTrack> mTracks;
	int mSequenceLength;

	mSequenceLength = frames;

	for(int j = 0; j < points; ++j)
	{
		CSimpleTrack pTrack;
		for(int i = 0; i < frames; ++i)
		{
			if(pVisibilityMatrix[i*points + j] == 1)
			{
				CPoint point;
				point.x = p2dPointsMatrix[2*i*points + j];
				point.y = p2dPointsMatrix[(2*i+1)*points + j];
				point.frame = i;
				pTrack.mPoints.push_back(point);
			}
		}
		pTrack.mLabel = label[j];
		mTracks.push_back(pTrack);
	}

	writeTracks(filename,mTracks,mSequenceLength);

	mxFree(filename);

	delete []p2dPointsMatrix;
	delete []pVisibilityMatrix;
	delete []label;


}
