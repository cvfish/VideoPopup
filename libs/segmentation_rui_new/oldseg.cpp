extern "C" {  
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"    
#include <TH.h>
#include <luaT.h>
}
#include <iostream>
#include <vector>
#include <map>
static const void* torch_DoubleTensor_id=NULL;

using namespace std;

#include "kmeans.cpp"
#include "class.cpp"
#include "allgc.cpp"
#include "shrink.cpp"
#include "mdl-fast.cpp"
#include "visualise.cpp"
#include "clump.cpp"
#include "invclump.cpp"
static const luaL_reg segs [] = {
  {"kmeans", wrapper_kmeans},
  {"visualise", wrapper_vis},
  {"visualise_col", wrapper_vis_col_2},
  {"visualise_col_bland", wrapper_vis_col},
  {"allgc", wrapper_allgc},
  {"overlap", wrapper_overgc},
  {"alphaexp", wrapper_pairgc},
  {"shrink", wrapper_shrink},
  {"mdl", wrapper_mdl},
  {"clump", wrapper_clump},
  {"invclump", wrapper_inv_clump},
  {NULL, NULL}  /* sentinel */
};
      
extern "C" {
  int luaopen_libsegmentation (lua_State *L) {
    torch_DoubleTensor_id=luaT_checktypename2id(L,"torch.DoubleTensor");
    luaL_openlib(L, "libsegmentation", segs, 0);
    return 1;
  }
}
