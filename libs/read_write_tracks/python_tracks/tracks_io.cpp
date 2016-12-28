#include "tracks_io.h"
#include "assert.h"

using namespace std;

void readTracks(char *filename, double *measurements, int frames, int tracks, int *labels, int labels_num)
{
  assert(tracks == labels_num);

  // cout << frames << " " << tracks << endl;

	ifstream aFile(filename);
	if(aFile.fail())
	  {
	    printf("%s doesn't exist\n",filename);
	    return;
	  }

  frames = frames / 2;

  int mSequenceLength;
  // Read number of frames considered
	aFile >> mSequenceLength;
  assert(frames == mSequenceLength);

	// Read number of tracks
	int aCount;
	aFile>> aCount;

  assert(tracks == aCount);
	// Read each track
	for (int i = 0; i < tracks; i++)
    {
      // cout << " tracks " << i << endl;
      // Read label and length of track
      int aSize;
      aFile >> labels[i];
      aFile >> aSize;
      // Read x,y coordinates and frame number of the tracked point
      double x,y; int frame;
      for (int j = 0; j < aSize; j++)
        {
          aFile >> x;
          aFile >> y;
          aFile >> frame;
          // cout << " x y frame: " << x << " " << y << " " << frame << endl;
          measurements[ 2*frame*tracks + i ] = x;
          measurements[ (2*frame+1)*tracks + i ] = y;
          // cout << measurements[ 2*frame*tracks + i ] << " " << measurements[ (2*frame+1)*tracks + i ] << endl;
        }
    }
}


void writeTracks(char *filename, double *measurements, int frames, int tracks, int *labels, int labels_num)
{
  assert(tracks == labels_num);

  frames = frames / 2;

	ofstream aFile(filename);
	// write number of frames considered
	aFile << frames << endl;

  // write number of tracks
	aFile << tracks << endl;

  for(int i = 0; i < tracks; ++i)
    {
      int numVisibleFrames = 0;
      for(int frame = 0; frame < frames; ++frame)
        {
          if( measurements[ (2*frame*tracks + i) ] != 0 ||
              measurements[ (2*frame+1)*tracks + i ] != 0)
            numVisibleFrames++;
        }
      aFile << labels[i] << " " << numVisibleFrames << endl;
      for(int frame = 0; frame < frames; ++frame)
        {
          if( measurements[ (2*frame*tracks + i) ] != 0 ||
              measurements[ (2*frame+1)*tracks + i ] != 0)
            {
              aFile << measurements[ (2*frame*tracks + i) ] << " "
                    << measurements[ (2*frame+1)*tracks + i ] << " "
                    << frame << endl;
            }
        }
    }
}
