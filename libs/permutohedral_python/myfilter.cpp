#include "Python.h"
//include "segmentation.h"
#undef assert
#include "macros.h"
#include "Image.h"
#include "permutohedral.h"
#include "Image.cpp"
extern "C" {
  void _filter1(double* control_image, int cx, int cy,int cd,
               double* perturb_image, int px, int py,int pd,
               double* out_image, int ox, int oy,int od,
               double SpatialStdev,double ColorStdev){
    if(cd!=3){
      PyErr_Format(PyExc_ValueError,"First image must be colour image of dimension w*h*3.");
      return;
    }
    if(pd!=od){
      PyErr_Format(PyExc_ValueError,"Second image and third image must have same final dimension.");
      return;
    }

    if(cx!=px){
      PyErr_Format(PyExc_ValueError,"First and second image have different first image dimensions");
      return;
    }
    if(cy!=py){
      PyErr_Format(PyExc_ValueError,"First and second image have different second image dimensions");
      return;
    }
    if(cx!=ox){
      PyErr_Format(PyExc_ValueError,"First and third(output) image have different first image dimensions");
      return;
    }
    if(cy!=oy){
      PyErr_Format(PyExc_ValueError,"First and third(output) image have different second image dimensions");
      return;
    }
    if(0>=SpatialStdev||0>=ColorStdev){
      PyErr_Format(PyExc_ValueError,"Standard deviations must be strictly greater than 0.");
      return;
    }
    float invSpatialStdev = 1.0f/SpatialStdev;
    float invColorStdev = 1.0f/ColorStdev;

    // Construct the position vectors out of x, y, r, g, and b.
    Image positions(1, cx, cy, 5);
    //First input is # of frames!

    for (int y = 0; y < cy; y++) {
      for (int x = 0; x < cx; x++) {
        positions(x, y)[0] = invSpatialStdev * x;
        positions(x, y)[1] = invSpatialStdev * y;
        positions(x, y)[2] = invColorStdev * control_image[(x*cy+y)*3+0];
        positions(x, y)[3] = invColorStdev *  control_image[(x*cy+y)*3+1];
        positions(x, y)[4] = invColorStdev *  control_image[(x*cy+y)*3+2];
      }
    }
    //create perturb image
    Image perturb(1, cx, cy, pd);
    for (int y = 0; y < cy; y++)
      for (int x = 0; x < cx; x++)
        for (int i = 0; i != pd; ++i)
          perturb(x,y)[i]=perturb_image[(x*cy+y)*pd+i];

    // Filter the input with respect to the position vectors. (see permutohedral.h)
    Image out = PermutohedralLattice::filter(perturb, positions);

    for (int y = 0; y < cy; y++)
      for (int x = 0; x < cx; x++)
        for (int i = 0; i != pd; ++i)
          out_image[(x*cy+y)*pd+i]=out(x,y)[i];

    return ;
  };
};
