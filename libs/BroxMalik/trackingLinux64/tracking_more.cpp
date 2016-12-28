#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include "CTensor.h"
#include "CFilter.h"
#include "ldof.h"

class CTrack {
public:
  CTrack() {mStopped = false; mLabel = -1;}
  std::vector<float> mx,my;             // current position of the track
  int mox,moy;                          // original starting point of the track
  int mLabel;                           // assignment to a region (ignored for tracking but makes the tracking files compatible to other tools I have)
  bool mStopped;                        // tracking stopped due to occlusion, etc.
  int mSetupTime;                       // Time when track was created
};

CVector<std::string> mInput;
std::string mFilename;
std::string mInputDir;
std::string mResultDir;
std::vector<CTrack> mTracks;
int mStartFrame;
int mStep;
int mSequenceLength;
int mXSize,mYSize;
CMatrix<float> mColorCode;
std::ofstream mStatus;

// buildColorCode
void buildColorCode() {
  mColorCode.setSize(3,15000);
  for (int i = 0; i < 256; i++) {
    mColorCode(0,i) = 0;
    mColorCode(1,i) = i;
    mColorCode(2,i) = 255;
  }
  for (int i = 0; i < 256; i++) {
    mColorCode(0,i+256) = 0;
    mColorCode(1,i+256) = 255;
    mColorCode(2,i+256) = 255-i;
  }
  for (int i = 0; i < 256; i++) {
    mColorCode(0,i+512) = i;
    mColorCode(1,i+512) = 255;
    mColorCode(2,i+512) = 0;
  }
  for (int i = 0; i < 256; i++) {
    mColorCode(0,i+768) = 255;
    mColorCode(1,i+768) = 255-i;
    mColorCode(2,i+768) = 0;
  }
  for (int i = 0; i < 256; i++) {
    mColorCode(0,i+1024) = 255;
    mColorCode(1,i+1024) = 0;
    mColorCode(2,i+1024) = i;
  }
  for (int i = 1280; i < mColorCode.ySize(); i++) {
    mColorCode(0,i) = 255;
    mColorCode(1,i) = 0;
    mColorCode(2,i) = 255;
  }
}

// readMiddlebury
bool readMiddlebury(const char* aFilename, CTensor<float>& aFlow) {
  FILE *stream = fopen(aFilename, "rb");
  if (stream == 0) {
    std::cout << "Could not open " << aFilename << std::endl;
    return false;
  }
  float help;
  int dummy;
  dummy = fread(&help,sizeof(float),1,stream);
  int aXSize,aYSize;
  dummy = fread(&aXSize,sizeof(int),1,stream);
  dummy = fread(&aYSize,sizeof(int),1,stream);
  aFlow.setSize(aXSize,aYSize,2);
  for (int y = 0; y < aFlow.ySize(); y++)
    for (int x = 0; x < aFlow.xSize(); x++) {
      dummy = fread(&aFlow(x,y,0),sizeof(float),1,stream);
      dummy = fread(&aFlow(x,y,1),sizeof(float),1,stream);
    }
  fclose(stream);
  return true;
}

// writeMiddlebury
bool writeMiddlebury(const char* aFilename, CTensor<float>& aFlow) {
  FILE *stream = fopen(aFilename, "wb");
  if (stream == 0) {
    std::cout << "Could not open " << aFilename << std::endl;
    return false;
  }
  float help=202021.25;
  int dummy;
  dummy = fwrite(&help,sizeof(float),1,stream);
  int aXSize = aFlow.xSize();
  int aYSize = aFlow.ySize();
  fwrite(&aXSize,sizeof(int),1,stream);
  fwrite(&aYSize,sizeof(int),1,stream);
  //aFlow.setSize(aXSize,aYSize,2);
  for (int y = 0; y < aFlow.ySize(); y++)
    for (int x = 0; x < aFlow.xSize(); x++) {
      fwrite(&aFlow(x,y,0),sizeof(float),1,stream);
      fwrite(&aFlow(x,y,1),sizeof(float),1,stream);
    }
  fclose(stream);
  return true;
}

class CPoint {
public:
  CPoint() {}
  float x,y,frame;
};

class CSimpleTrack {
public:
  CSimpleTrack() {}
  int mLabel;
  CVector<CPoint> mPoints;
};

// writeTracks
void writeTracks() {
  char buffer[50];
  sprintf(buffer,"Tracks%d.dat",mSequenceLength);
  std::ofstream aFile((mResultDir+mFilename+buffer).c_str());
  aFile << mSequenceLength << std::endl;
  int aCount = 0;
  for (int i = 0; i < mTracks.size(); i++)
    if (mTracks[i].mLabel >= 0) aCount++;
  aFile << aCount << std::endl;
  for (int i = 0; i < mTracks.size(); i++) {
    if (mTracks[i].mLabel < 0) continue;
    int aSize = mTracks[i].mx.size();
    aFile << mTracks[i].mLabel << " " << aSize+1 << std::endl;
    aFile << mTracks[i].mox << " " << mTracks[i].moy << " " << mTracks[i].mSetupTime << std::endl;
    for (int j = 0; j < aSize; j++)
      aFile << mTracks[i].mx[j] << " " << mTracks[i].my[j] << " " << mTracks[i].mSetupTime+j+1 << std::endl;
  }
}

// computeCorners --------------------------------------------------------------
void computeCorners(CTensor<float>& aImage, CMatrix<float>& aCorners, float aRho) {
  aCorners.setSize(aImage.xSize(),aImage.ySize());
  int aXSize = aImage.xSize();
  int aYSize = aImage.ySize();
  int aSize = aXSize*aYSize;
  // Compute gradient
  CTensor<float> dx(aXSize,aYSize,aImage.zSize());
  CTensor<float> dy(aXSize,aYSize,aImage.zSize());
  CDerivative<float> aDerivative(3);
  NFilter::filter(aImage,dx,aDerivative,1,1);
  NFilter::filter(aImage,dy,1,aDerivative,1);
  // Compute second moment matrix
  CMatrix<float> dxx(aXSize,aYSize,0);
  CMatrix<float> dyy(aXSize,aYSize,0);
  CMatrix<float> dxy(aXSize,aYSize,0);
  int i2 = 0;
  for (int k = 0; k < aImage.zSize(); k++)
    for (int i = 0; i < aSize; i++,i2++) {
      dxx.data()[i] += dx.data()[i2]*dx.data()[i2];
      dyy.data()[i] += dy.data()[i2]*dy.data()[i2];
      dxy.data()[i] += dx.data()[i2]*dy.data()[i2];
    }
  // Smooth second moment matrix
  NFilter::recursiveSmoothX(dxx,aRho);
  NFilter::recursiveSmoothY(dxx,aRho);
  NFilter::recursiveSmoothX(dyy,aRho);
  NFilter::recursiveSmoothY(dyy,aRho);
  NFilter::recursiveSmoothX(dxy,aRho);
  NFilter::recursiveSmoothY(dxy,aRho);
  // Compute smallest eigenvalue
  for (int i = 0; i < aSize; i++) {
    float a = dxx.data()[i];
    float b = dxy.data()[i];
    float c = dyy.data()[i];
    float temp = 0.5*(a+c);
    float temp2 = temp*temp+b*b-a*c;
    if (temp2 < 0.0f) aCorners.data()[i] = 0.0f;
    else aCorners.data()[i] = temp-sqrt(temp2);
  }
}

// Code fragment from Pedro Felzenszwalb  ---------------
// http://people.cs.uchicago.edu/~pff/dt/
void dt(CVector<float>& f, CVector<float>& d, int n) {
  d.setSize(n);
  int *v = new int[n];
  float *z = new float[n+1];
  int k = 0;
  v[0] = 0;
  z[0] = -10e20;
  z[1] = 10e20;
  for (int q = 1; q <= n-1; q++) {
    float s  = ((f[q]+q*q)-(f(v[k])+v[k]*v[k]))/(2*(q-v[k]));
    while (s <= z[k]) {
      k--;
      s  = ((f[q]+q*q)-(f(v[k])+v[k]*v[k]))/(2*(q-v[k]));
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k+1] = 10e20;
  }
  k = 0;
  for (int q = 0; q <= n-1; q++) {
    while (z[k+1] < q)
      k++;
    int help = q-v[k];
    d(q) = help*help + f(v[k]);
  }
  delete[] v;
  delete[] z;
}
// ------------------------------------------------------

// euclideanDistanceTransform
void euclideanDistanceTransform(CMatrix<float>& aMatrix) {
  int aXSize = aMatrix.xSize();
  int aYSize = aMatrix.ySize();
  CVector<float> f(NMath::max(aXSize,aYSize));
  // Transform along columns
  for (int x = 0; x < aXSize; x++) {
    for (int y = 0; y < aYSize; y++)
      f(y) = aMatrix(x,y);
    CVector<float> d;
    dt(f,d,aYSize);
    for (int y = 0; y < aYSize; y++)
      aMatrix(x,y) = d(y);
  }
  // Transform along rows
  for (int y = 0; y < aYSize; y++) {
    int aOffset = y*aXSize;
    for (int x = 0; x < aXSize; x++)
      f(x) = aMatrix.data()[x+aOffset];
    CVector<float> d;
    dt(f,d,aXSize);
    for (int x = 0; x < aXSize; x++)
      aMatrix.data()[x+aOffset] = d(x);
  }
}

int main(int argc, char** args) {

  if(argc < 5)
    {
      std::cout<<args[0]<<" <bmf file> <start frame number (starts from 0)> <Sequence length> <subsampling factor (1 for dense, >1 for subsampled - integer only) > "<<std::endl;
      exit(0);
    }
  clock_t fileopstart,fileop=0;
  CVector<float>color = CVector<float>(3);
  // Determine input directory
  std::string s = args[1];
  s.erase(s.find_last_of("/")+1,s.length());
  mInputDir = s;
  s = args[1];
  s.erase(0,s.find_last_of('.'));
  // Read image sequence in bmf format
  if (s == ".bmf" || s == ".BMF") {
    int aImageCount,aViewCount;
    std::ifstream aStream(args[1]);
    aStream >> aImageCount;
    aStream >> aViewCount;
    mInput.setSize(aImageCount);
    for (int i = 0; i < aImageCount; i++) {
      std::string s;
      aStream >> s;
      mInput(i) = mInputDir+s;
    }
  }
  else {
    std::cout << "Must pass a bmf file as input" << std::endl;
    return -1;
  }
  // Determine image/sequence name
  s = args[1];
  s.erase(0,s.find_last_of("/")+1);
  s.erase(s.find_first_of('.'));
  mFilename = s;
  // Make directory for results
  mResultDir = mInputDir + mFilename + "Results";
  std::string s2 = "mkdir " + mResultDir;
  int dummy = system(s2.c_str());
  mResultDir += "/";
  buildColorCode();
  // Set beginning and end of tracking
  mStartFrame = atoi(args[2]);
  mSequenceLength = atoi(args[3]);
  mStep = atoi(args[4]);
  char buffer[100];
  sprintf(buffer,"Status%d.txt",mSequenceLength);
  mStatus.open((mResultDir+buffer).c_str());
  clock_t start = clock();
  // Load first image
  CTensor<float>* aImage1 = new CTensor<float>;
  CTensor<float>* aImage2;
  fileopstart = clock();
  std::cout<<mInput(mStartFrame).c_str()<<std::endl;
  aImage1->readFromPPM(mInput(mStartFrame).c_str());
  std::cout<<mInput(mStartFrame).c_str()<<std::endl;
  fileop += clock() - fileopstart;
  mXSize = aImage1->xSize();
  mYSize = aImage1->ySize();
  mStatus << "Tracks are being computed..." << std::endl;
  mTracks.clear();
  CMatrix<float> aCorners;
  CMatrix<float> aCovered(mXSize,mYSize);
  int aSize = mXSize*mYSize;
  // Smooth first image (can be removed from optical flow computation then)
  NFilter::recursiveSmoothX(*aImage1,0.8f);
  NFilter::recursiveSmoothY(*aImage1,0.8f);
  // Tracking
  for (int t = 0; t < mSequenceLength-1; t++) {
    // Load next image
    aImage2 = new CTensor<float>;
    fileopstart = clock();
    aImage2->readFromPPM(mInput(mStartFrame+t+1).c_str());
    std::cout<<mInput(mStartFrame+t+1).c_str()<<std::endl;
    fileop += clock() - fileopstart;
    NFilter::recursiveSmoothX(*aImage2,0.8f);
    NFilter::recursiveSmoothY(*aImage2,0.8f);
    // Mark areas sufficiently covered by tracks
    aCovered = 1e20;
    if (t > 0) {
      for (unsigned int i = 0; i < mTracks.size(); i++)
        if (mTracks[i].mStopped == false) aCovered((int)mTracks[i].mx.back(),(int)mTracks[i].my.back()) = 0.0f;
      euclideanDistanceTransform(aCovered);
    }
    // Set up new tracking points in uncovered areas

    // computeCorners(*aImage1,aCorners,3.0f);
    // float aCornerAvg = aCorners.avg();

    for (int ay = 4; ay < mYSize-4; ay+=mStep)
      for (int ax = 4; ax < mXSize-4; ax+=mStep) {
        if (aCovered(ax,ay) < mStep*mStep) continue;

        // float distToImageBnd = exp(-0.1*NMath::min(NMath::min(NMath::min(ax,ay),mXSize-ax),mYSize-ay));
        // if (aCorners(ax,ay) < 1.0* (aCornerAvg*(0.1+distToImageBnd))) continue;
        // if (aCorners(ax,ay) < 1.0*(1.0f+distToImageBnd)) continue;

        mTracks.push_back(CTrack());
        CTrack& newTrack = mTracks.back();
        newTrack.mox = ax;
        newTrack.moy = ay;
        newTrack.mLabel = -1;
        newTrack.mSetupTime = t;
      }
    // Compute bidirectional LDOF or read from file when available
    CTensor<float> aForward,aBackward;
    sprintf(buffer,"ForwardFlow%03d.flo",t+mStartFrame);
    std::string aName1 = mResultDir+buffer;
    sprintf(buffer,"BackwardFlow%03d.flo",t+mStartFrame);
    std::string aName2 = mResultDir+buffer;
    if (readMiddlebury(aName1.c_str(),aForward) == false || readMiddlebury(aName2.c_str(),aBackward) == false)
      {
        mStatus << "Compute optical flow..." << std::endl;
        // Call optical flow
        ldof(*aImage1,*aImage2,aForward,aBackward);
        writeMiddlebury(aName1.c_str(), aForward);
        writeMiddlebury(aName2.c_str(), aBackward);
      }
    // Check consistency of forward flow via backward flow
    CMatrix<float> aUnreliable(mXSize,mYSize,0);
    CTensor<float> dx(mXSize,mYSize,2);
    CTensor<float> dy(mXSize,mYSize,2);
    CDerivative<float> aDev(3);
    NFilter::filter(aForward,dx,aDev,1,1);
    NFilter::filter(aForward,dy,1,aDev,1);
    CMatrix<float> aMotionEdge(mXSize,mYSize,0);
    for (int i = 0; i < aSize; i++) {
      aMotionEdge.data()[i] += dx.data()[i]*dx.data()[i];
      aMotionEdge.data()[i] += dx.data()[aSize+i]*dx.data()[aSize+i];
      aMotionEdge.data()[i] += dy.data()[i]*dy.data()[i];
      aMotionEdge.data()[i] += dy.data()[aSize+i]*dy.data()[aSize+i];
    }
    for (int ay = 0; ay < aForward.ySize(); ay++)
      for (int ax = 0; ax < aForward.xSize(); ax++) {
        float bx = ax+aForward(ax,ay,0);
        float by = ay+aForward(ax,ay,1);
        int x1 = floor(bx);
        int y1 = floor(by);
        int x2 = x1+1;
        int y2 = y1+1;
        if (x1 < 0 || x2 >= mXSize || y1 < 0 || y2 >= mYSize) { aUnreliable(ax,ay) = 1.0f; continue;}
        float alphaX = bx-x1; float alphaY = by-y1;
        float a = (1.0-alphaX)*aBackward(x1,y1,0)+alphaX*aBackward(x2,y1,0);
        float b = (1.0-alphaX)*aBackward(x1,y2,0)+alphaX*aBackward(x2,y2,0);
        float u = (1.0-alphaY)*a+alphaY*b;
        a = (1.0-alphaX)*aBackward(x1,y1,1)+alphaX*aBackward(x2,y1,1);
        b = (1.0-alphaX)*aBackward(x1,y2,1)+alphaX*aBackward(x2,y2,1);
        float v = (1.0-alphaY)*a+alphaY*b;
        float cx = bx+u;
        float cy = by+v;
        float u2 = aForward(ax,ay,0);
        float v2 = aForward(ax,ay,1);
        if (((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay)) >= 0.01*(u2*u2+v2*v2+u*u+v*v)+0.5f) { aUnreliable(ax,ay) = 1.0f; continue;}
        if (aMotionEdge(ax,ay) > 0.01*(u2*u2+v2*v2)+0.002f) { aUnreliable(ax,ay) = 1.0f; continue;}
      }
    CTensor<float> aShow(*aImage2);
    for (unsigned int i = 0; i < mTracks.size(); i++) {
      if (mTracks[i].mStopped) continue;
      float ax,ay,oldvar;
      if (mTracks[i].mSetupTime == t) {
        ax = mTracks[i].mox; ay = mTracks[i].moy;
      }
      else {
        ax = mTracks[i].mx.back(); ay = mTracks[i].my.back();
      }
      int iax = lroundf(ax);
      int iay = lroundf(ay);
      if (aUnreliable(iax,iay) > 0) mTracks[i].mStopped = true;
      else {
        float bx = ax+aForward(iax,iay,0);
        float by = ay+aForward(iax,iay,1);
        int ibx = lroundf(bx);
        int iby = lroundf(by);
        if (ibx < 0 || iby < 0 || ibx >= mXSize || iby >= mYSize) mTracks[i].mStopped = true;
        else {
          mTracks[i].mx.push_back(bx);
          mTracks[i].my.push_back(by);
          mTracks[i].mLabel = 0;
          int t = mTracks[i].mx.size();
          int f = 1300/mSequenceLength;
          f=10;
          color.data()[0] = mColorCode(0,f*t);
          color.data()[1] = mColorCode(1,f*t);
          color.data()[2] = mColorCode(2,f*t);

          //aShow.drawRect(ibx-1,iby-1,ibx+1,iby+1,mColorCode(0,4*t),mColorCode(1,4*t),mColorCode(2,4*t));
          //aShow.drawRect(ibx-1,iby-1,ibx+1,iby+1,mColorCode(0,f*t),mColorCode(1,f*t),mColorCode(2,f*t));
          // draw colored square
          {
            int x1 = ibx-1 < 0 ? 0 : ibx-1;
            int y1 = iby-1 < 0 ? 0 : iby-1;
            int x2 = ibx+1 >= mXSize ? mXSize-1 : ibx+1;
            int y2 = iby+1 >= mYSize ? mYSize-1 : iby+1;
            aShow.fillRect(color, x1,y1,x2,y2);
          }

        }
      }
    }
    sprintf(buffer,"Tracking%03d.ppm",t+mStartFrame);
    aShow.writeToPPM((mResultDir+buffer).c_str());
    // Prepare for next image
    delete aImage1; aImage1 = aImage2;
  }
  delete aImage1;
  // Write tracks to file
  writeTracks();
  clock_t finish = clock();
  mStatus << (double(finish)-double(start)-(double)fileop)/CLOCKS_PER_SEC << std::endl;
  mStatus.close();
  return 0;
}
