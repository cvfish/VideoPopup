// #include "mex.h"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
//include "ibfs.h"
#include "graph.h"
#include "graph.cpp"
//include "ibfs.cpp"
#include "maxflow.cpp"
#include <limits>
//include <mex.h>
#define xstr(s) str(s)
#define str(s) #s
const bool disable_mask=false;
//define DEBUG ___

#ifdef DEBUG
#define debug_test(x) x
#define mexassert(x)                                                    \
  {if(! (x))                                                            \
      mexErrMsgTxt("ERROR!! Mexassert: " #x  " failed\n  on line " xstr(__LINE__) " in file " __FILE__ "\n");}
#else
#define mexassert(x)
#define debug_test(x)
#endif


using namespace std;
//typedef unsigned __int128 gtype;
//typedef unsigned short gtype;
typedef double gtype;
//template class IBFSGraph<gtype,gtype,gtype>;
template class Graph<gtype,gtype,gtype>;

class hypothesis{
public:
  inline int MASK_ACCESS(int node, int lab){
    mexassert(node>=0);
    mexassert(lab>=0);
    mexassert(node<=nodes);
    mexassert(lab<=hyp);
    return node+nodes*lab;
    //((node)*hyp+(lab))
  }
  int max_neigh_p;
  int max_neigh_m;
  int nodes, hyp;
  vector<int>  label;
  inline int& get_label(int node){
    mexassert(node>=0);
    mexassert(node<nodes);
    mexassert(label[node]>=0);
    mexassert(label[node]<hyp);
    return label[node];}
  vector<int>used_hyp;
  int total_used;
  vector<double>  mask;
  // vector<double> un;
  double *un;
  double* models;
  double lambda;
  double *hypweight;
  int maxnodes;
  double *thresh;
  vector<int> neigh_p;
  vector<int> neigh_m;
  //  vector<double> _ncost;
  double *_ncost;
  double *_mcost;
  double overlap_penalty;
  vector <double> mask_backup;
  double scalar;
  gtype max_int;
  inline void set_scalar(double cost){
    //    cost=500;
    mexassert(cost);
    max_int=-1;
    scalar=1;//0.99*max_int/(cost+0.00001);
    //    cout<<"scalar is "<<scalar<<endl;
    //cout<<"cost is "<<cost<<endl;
    //    cout<<"max_int is "<<max_int<<endl;
    //cout<<"numeric_limits<unsigned int>::max() is "<<numeric_limits<unsigned int>::max()<<endl;
  }

  inline void add_tweights(int i,double w1,double w2, char id){
    if ((w1==0)&&(w2==0))
      return;
    mexassert(i>=0);
    debug_test(if (i>=maxnodes)
                 mexPrintf("i is %d and maxnodes is %d id is %c\n",i,maxnodes,id);
               );
    mexassert(i<maxnodes);
    //  mexassert((w1*scalar)<max_int)
    // mexassert((w2*scalar)<max_int)
    gtype s1=static_cast<gtype>(scalar*w1);
    gtype s2=static_cast<gtype>(scalar*w2);
    g-> add_tweights(i,s1,s2);
  }
  inline void add_edge(int i,int j,double w1,double w2,char id){
    debug_test(
               if(!((w2>=0)&&(w1>=0)&&(i!=j)&&(i>=0)&&(j>=0)))
                 printf("w1 %f w2 %f i %d j %d, id %c\n",w1,w2,i,j,id);
               );
    mexassert(i!=j);
    mexassert(i>=0);
    mexassert(j>=0);
    debug_test(
               if (i>=maxnodes ||j >= maxnodes)
                 printf("Add edge is greater than %d: out of bounds i %d j %d %c\n",maxnodes,i,j,id);
               );
    mexassert(i<maxnodes);
    mexassert(j<maxnodes);
    mexassert(w1>=0.0);
    mexassert(w2>=0.0);
    if ((w1==0)&&(w2==0))
      return;
    //  mexassert(scalar*w1<max_int);
    // mexassert(scalar*w2<max_int);
    gtype s1=static_cast<gtype>(scalar*w1);
    gtype s2=static_cast<gtype>(scalar*w2);
    g->add_edge(i,j,s1,s2);
  }
  inline double& unary(int node,int lab){
    mexassert(node<nodes);
    mexassert(node>=0);
    mexassert(lab>=0);
    mexassert(lab<hyp);
    return un[node+lab*nodes];
  }
  inline double get_mask(int node, int lab){
    mexassert(nodes*hyp==mask.size());
    mexassert(node>=0);
    mexassert(lab>=0);
    mexassert(node<nodes);
    mexassert(lab<hyp);
    return mask[MASK_ACCESS(node,lab)];
  }
  inline double hweight(int lab){
    mexassert(hypweight);
    mexassert(lab<hyp);
    mexassert(lab>=0);
    return hypweight[lab];
  }
  inline int &overlap_neighbour(int n,int n2,char loc){
    debug_test(
               if (!((n>=0)&&(n<nodes)))
                 mexPrintf("bad choice of n = %d, loc is %c \n",n,loc);
               if (n2>=max_neigh_m){
                 mexPrintf("bad choice of n = %d, loc is %c \n",n,loc);
                 mexPrintf("bad choice of n2 = %d>= %d \n",n2,max_neigh_m);
               }
               );
    mexassert(n>=0);
    mexassert(n<nodes);
    mexassert(n2<max_neigh_m);
    mexassert(n2>=0);
    mexassert (neigh_m[n*max_neigh_m+n2]<nodes);
    mexassert (neigh_m[n*max_neigh_m+n2]>=-1);
    return neigh_m[n*max_neigh_m+n2];
  }
  int &pair_neighbour(int n,int n2){
    mexassert(n>=0);
    mexassert(n2>=0);
    mexassert(n<nodes);
    mexassert(n2<max_neigh_p);
    return neigh_p[n*max_neigh_p+n2];
    //return neigh_p[n+nodes*n2];
  }
  double &ncost(int n,int n2){
    //mexassert(_ncost);
    mexassert(n<nodes);
    mexassert(n>=0);
    mexassert(n2>=0);
    mexassert(n2<max_neigh_p);
    mexassert(!isnan(_ncost[n*max_neigh_p+n2]));
    mexassert(_ncost[n*max_neigh_p+n2]>=0);
    mexassert(_ncost[n*max_neigh_p+n2]!=INFINITY);
    return _ncost[n*max_neigh_p+n2];
  }

  Graph<gtype,gtype,gtype>* g;

  hypothesis(unsigned int nodes,unsigned int maxn, unsigned int hyp,double l,double* w1,unsigned int maxn_p){
    overlap_penalty=0;
    cout<<"nodes is "<<nodes<<endl;
    cout<<"hyp is "<<hyp<<endl;
    cout<<"l is "<<l<<endl;
    cout<<"maxn_p is "<<maxn_p<<endl;
    _ncost=NULL;
    _mcost=NULL;
    // un.resize(nodes*hyp);
    max_neigh_p=maxn_p;
    hypweight=w1;
    lambda=l;
    max_neigh_m=maxn;
    thresh = NULL;    //Initialize as NULL if no input
    this->nodes=nodes;
    int edges=maxn*nodes;
    this->hyp=hyp;
    label.resize(nodes,0);
    neigh_m.resize(nodes*max_neigh_m,-1);
    neigh_p.resize(nodes*max_neigh_p,-1);
    // _ncost.resize(nodes*max_neigh_p,0);
    //    max_neigh_p=0;
    maxnodes=3*nodes/*nothing which is alpha*/ + edges/* non interior points*/ +hyp/*MDL prior*/;
    printf("nodes %d, edges %d hyp %d\n", nodes, edges, hyp);
    g=new Graph<gtype,gtype,gtype> (maxnodes,edges*2+nodes*3+max_neigh_p*nodes);
  }

  hypothesis(unsigned int nodes,unsigned int maxn, unsigned int hyp,double l,double* w1,unsigned int maxn_p,double *ppthresh){
    overlap_penalty=0;
    cout<<"nodes is "<<nodes<<endl;
    cout<<"hyp is "<<hyp<<endl;
    cout<<"l is "<<l<<endl;
    cout<<"maxn_p is "<<maxn_p<<endl;
    // un.resize(nodes*hyp);
    max_neigh_p=maxn_p;
    hypweight=w1;
    lambda=l;
    _mcost=NULL;
    max_neigh_m=maxn;
    thresh = ppthresh;
    this->nodes=nodes;
    int edges=maxn*nodes;
    this->hyp=hyp;
    label.resize(nodes,0);
    neigh_m.resize(nodes*max_neigh_m,-1);
    neigh_p.resize(nodes*max_neigh_p,-1);
    //_ncost.resize(nodes*max_neigh_p,0);
    //    max_neigh_p=0;
    maxnodes=3*nodes/*nothing which is alpha*/ + edges/* non interior points*/ +hyp/*MDL prior*/;
    printf("nodes %d, edges %d hyp %d\n", nodes, edges, hyp);
    g=new Graph<gtype,gtype,gtype> (maxnodes,edges*2+nodes*3+max_neigh_p*nodes);
  }


  ~hypothesis(){
    delete g;
  }
  inline void multi_construct_helper(const int i,const int j, const int alpha){
    mexassert(i>=0);
    mexassert(j>=0);
    mexassert(alpha>=0);
    int temp=g->add_node(1);
    add_tweights(temp,0,unary(i,j)*lambda,'e');
    for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k)
      if(get_label(overlap_neighbour(i,k,'b'))==j)
        add_edge(overlap_neighbour(i,k,'b'),temp,unary(i,j)*lambda,0,'c');
    if(get_label(i)==j){
      add_edge(i,temp,unary(i,j)*lambda,0,'d');
      add_tweights(i,unary(i,alpha)*(1.0-lambda),unary(i,j)*(1.0-lambda),'v');
    }
  }

  void construct_multi(const int alpha){
    g->add_node(nodes);
    for (int i = 0; i !=nodes; ++i)
      if(get_label(i)!=alpha){
        bool nalpha=false;
        //check if overlap_neighbours contain alpha
        for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'a')!=-1); ++j)
          nalpha|=(get_label(overlap_neighbour(i,j,'a'))==alpha);

        if(!nalpha)
          // if(nalpha) then boundary set must always contain alpha
          //and transition to I=alpha is free
          //and remaining `unary' cost comes in section 2
          //else
          {
            add_tweights(nodes+i,unary(i,alpha)*lambda,0,'d');
            add_edge(i,nodes+i,0,unary(i,alpha)*lambda,'a');
            for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'a')!=-1); ++j)
              add_edge(overlap_neighbour(i,j,'a'),nodes+i,
                       0,lambda*unary(i,alpha),'u');
          }
      }
    //Half way --- we have added all costs associated with changing node n to alpha
    // we now need to add the costs to overlap_neighbours of keeping a node fixed

    vector<bool>nclass(hyp);
    for (int i = 0; i !=nodes; ++i){
      //      //clog<<"i "<<i<<endl;
      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'b')!=-1); ++j)
        nclass[get_label(overlap_neighbour(i,j,'b'))]=true; //check overlap_neighbours
      nclass[get_label(i)]=true;//check self
      nclass[alpha]=false;
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if(nclass[j]){
          nclass[j]=false;
          multi_construct_helper(i,j,alpha);
        }
      }
      if (nclass[get_label(i)])
        multi_construct_helper(i,get_label(i),alpha);
    }
  }
  inline void helper_robust_multi(const int i,const int lab, const int alpha){
    mexassert(lab!=alpha);
    mexassert(i>=0);
    mexassert(i<nodes);
    mexassert(lab<hyp);
    int temp=g->add_node(1);
    add_tweights(temp,0,unary(i,lab)*lambda,'E');
    for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'d')!=-1); ++k)
      if(get_label(overlap_neighbour(i,k,'d'))==lab)
        add_edge(overlap_neighbour(i,k,'d'),temp,_mcost[i*max_neigh_m+k],0,'C');
    if (get_label(i)==lab){
      add_edge(i,temp,unary(i,lab)*lambda*1.1+1+2*overlap_penalty+hweight(lab),0,'d');
      add_tweights(i,unary(i,alpha)*(1.0-lambda),unary(i,lab)*(1.0-lambda),'v');
    }
  }

  void construct_robust_multi(const int alpha){
    /*We also need a robut version of overlapping models.  Instead of
      minimizing the truncated threshold, optimize
      \min(U_i(\alpha),\sum_{j \in \Delta(j \in \alpha) P_{i,j} This
      can be understood as standard overlap cost + the ability to pay
      cost p_{i,j} to remove j from N_i.

      If U_i is the minima, then the point lies in the boundary,
      otherwise it lies outside.  Essentially this is just replacing
      the pairwise edge between the auxillary variable and the
      interior point with a fixed parwise cost.  Uses pointer to
      double _mcost to store these costs, with the same interface as
      _ncost.
    */
    //cout<<"_mcost "<<(unsigned long) _mcost<<endl;
    mexassert(_mcost);
    g->add_node(nodes);
    for (int i = 0; i !=nodes; ++i)
      if(get_label(i)!=alpha){
        add_tweights(nodes+i,unary(i,alpha)*lambda,0,'d');
        add_edge(i,nodes+i,0,unary(i,alpha)*lambda*1.1+1,'a');
        for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'c')!=-1); ++j)
          add_edge(overlap_neighbour(i,j,'c'),nodes+i,
                   0,_mcost[i*max_neigh_m+j],'p');
      }

    //Half way --- we have added all costs associated with changing node n to alpha
    // we now need to add the costs to overlap_neighbours of keeping a node fixed

    vector<double>nclass(hyp);
    for (int i = 0; i !=nodes; ++i){
      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'d')!=-1); ++j)
        nclass[get_label(overlap_neighbour(i,j,'d'))]=true;
      nclass[get_label(i)]=true;//check self
      nclass[alpha]=false;
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if(nclass[j]){
          nclass[j]=false;
          helper_robust_multi(i, j, alpha);
        }
      }
      if (nclass[get_label(i)])
        helper_robust_multi(i, get_label(i), alpha);
      if(get_label(i)==alpha)
        add_tweights(i,unary(i,alpha)*(1.0-lambda),10000000000,'q');
    }
  }
  void construct_robust_multi_with_mask(const int alpha){
    /*We also need a robut version of overlapping models.  Instead of
      minimizing the truncated threshold, optimize
      \min(U_i(\alpha),\sum_{j \in \Delta(j \in \alpha) P_{i,j} This
      can be understood as standard overlap cost + the ability to pay
      cost p_{i,j} to remove j from N_i.

      If U_i is the minima, then the point lies in the boundary,
      otherwise it lies outside.  Essentially this is just replacing
      the pairwise edge between the auxillary variable and the
      interior point with a fixed parwise cost.  Uses pointer to
      double _mcost to store these costs, with the same interface as
      _ncost.
      When we combine this with sparsity, most of the existing shortcuts don't work.
      Disable theshold shortcut.
      alpha check is disabled  -- can be added later
    */
    mexassert(_mcost);
    mexassert(mask.size()!=0);
    g->add_node(nodes);
    for (int i = 0; i !=nodes; ++i){
      add_tweights(nodes+i,unary(i,alpha)*lambda,0,'d');
      add_edge(i,nodes+i,0,unary(i,alpha)*lambda*1.1+1+2*overlap_penalty+hweight(alpha),'a');
      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'c')!=-1); ++j)
        add_edge(overlap_neighbour(i,j,'c'),nodes+i,
                 0,_mcost[i*max_neigh_m+j],'p');
    }

    //Half way --- we have added all costs associated with changing node n to alpha
    // we now need to add the costs to overlap_neighbours of keeping a node fixed

    vector<bool>nclass(hyp);
    for (int i = 0; i !=nodes; ++i){
      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'d')!=-1); ++j)
        nclass[get_label(overlap_neighbour(i,j,'d'))]=true;
      // get_mask(i,get_label(overlap_neighbour(i,j,'d')));
      nclass[get_label(i)]=1;//check self
      nclass[alpha]=false;
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if(nclass[j]){
          nclass[j]=false;
          if (get_mask(i,j))
            helper_robust_multi(i, j, alpha);
          // else
          //   add_tweights(overlap_neighbour(i,k,'b'),_mcost[i*max_neigh_m+j],0,'X');
          //I can't figure out if the above makes sense
        }
      }
      if (nclass[get_label(i)])
        helper_robust_multi(i, get_label(i), alpha);
      if(get_label(i)==alpha)
        add_tweights(i,0,10000000000,'q');
    }
  }
  void backup_mask(){
    debug_test(cout<<"backup_mask"<<endl;)
      mask_backup.resize(nodes*max_neigh_m);
    for (int i = 0; i != nodes; ++i)
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        mask_backup[max_neigh_m*i+k]=get_mask(i,j);
      }
  }
  void void_mask(){
    debug_test(cout<<"void_mask"<<endl;)
      for (int i = 0; i != nodes; ++i){
        mask[MASK_ACCESS(i,get_label(i))]=0;
        for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k)
          mask[MASK_ACCESS(i,get_label(overlap_neighbour(i,k,'b')))]=0;
      }
  }

  void restore_mask(){
    debug_test(cout<<"restore_mask"<<endl;)
      // for (int i = 0; i != nodes*hyp; ++i)
      //   mask[i]=0;
      for(int i = 0; i != nodes; ++i)
        for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
          unsigned int j=get_label(overlap_neighbour(i,k,'b'));
          mask[MASK_ACCESS(i,j)]=mask_backup[i*max_neigh_m+k];
        }
    for (int i = 0; i != nodes; ++i)
      mask[MASK_ACCESS(i,get_label(i))]=1;

  }
  void construct_sparse_overlap(const int alpha){
    /* We want to pay a cost \geq \sum_p \sum_{l_1}\sum_{l_2}
       overlap_penalty\Delta (\exist x: x\in M_{l_1},x \in M_{l_2})

       that is tight at the current location.

       we approximate this cost with the local MDL costs:
       \Delta (\exist x\in M_{l_1}, x \in M^{prev}_{l_2})+
       \Delta (\exist x \in M_{l_2})
       -1

       We reparameterise by adding cost and split into two mdl costs over I_\beta and
       m_\beta.
    */

    int offset_masked_MDL=g->add_node(total_used)-1;

    //add penalty for model j covering the interior of alpha

    vector<bool>nclass(hyp,false);
    //Which classes already neighbour model alpha?
    for (int i = 0; i != nodes; ++i)
      if (get_label(i)==alpha)
        for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
          unsigned int j=get_label(overlap_neighbour(i,k,'b'));
          if (get_mask(i,j))
            nclass[j]= true;
        }
    nclass[alpha]=true;
    //Ignore them, ignore alpha, and impose an MDL cost on neighbouring the rest
    for (int j = 0; j != hyp; ++j)
      nclass[j]=nclass[j]&&!used_hyp[j]; //Also don't care about inactive models

    for (int j=0 ; j!= hyp; ++j)
      if (!nclass[j])
        add_tweights(offset_masked_MDL+used_hyp[j],overlap_penalty,0,'d');

    vector <bool>called(hyp);

    for (int i = 0; i != nodes; ++i){
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        called[get_label(overlap_neighbour(i,k,'b'))]=
          nclass[get_label(overlap_neighbour(i,k,'b'))];
      }
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if ((!called[j])&&get_mask(i,j)){
          called[j]=true;
          add_edge(i,offset_masked_MDL+used_hyp[j],0,overlap_penalty,'Q');
        }
      }
    }
    for (int i = 0; i != nodes; ++i)
      for (int j = 0; j != hyp; ++j)
        if (get_mask(i,j)&&j!=alpha)
          add_edge(i,offset_masked_MDL + used_hyp[j],0,overlap_penalty,'Q');

    offset_masked_MDL=g->add_node(total_used)-1;//New aux_variables
    //add penalty for model alpha covering the interior of model j

    for (int j=0 ; j!= hyp; ++j)
      if ((get_label(j)!=alpha)&&used_hyp[j])
        add_tweights(offset_masked_MDL+used_hyp[j],overlap_penalty,0,'d');
    //Add edge for all auxillary variables over points not taking interior label alpha
    for (int i = 0; i !=nodes; ++i)
      if (get_label(i)!=alpha)
        add_edge(nodes+i,offset_masked_MDL+used_hyp[get_label(i)],0,overlap_penalty,'Q');
  }

  inline void multi_thresh_construct_helper(const int i,const int j, const int alpha){
    mexassert(i>=0);
    mexassert(j>=0);
    mexassert(alpha>=0);
    int temp=g->add_node(1);
    add_tweights(temp,0,unary(i,j)*lambda,'e');
    for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k)
      if(get_label(overlap_neighbour(i,k,'b'))==j)
        add_edge(overlap_neighbour(i,k,'b'),temp,unary(i,j)*lambda,0,'c');
    if(get_label(i)==j){
      add_edge(i,temp,unary(i,j)*lambda,0,'d');
      add_tweights(i,unary(i,alpha)*(1.0-lambda),unary(i,j)*(1.0-lambda),'v');
    }
  }
  void construct_multi_thresh(const int alpha){//This is new
    mexassert(thresh!=0);
    /*
      May as well document this in the code, as it will stop it getting lost.

      What we want is a per point threshold t_p that only applies to
      exterior models. Interior labels still take the same costs.  As
      such this needs to be handled in a mannor analagous to the
      lambda parameter.  The cost of a model belonging to the
      interior/exterior of model, must be split into two components -
      the truncated cost of belonging to the exterior, and the
      residual cost or interior cost -exterior cost.

      We will apply lambda first, and define:
      ex_cost:= min (t_p,\lambda U_p(m))
      int_cost:= U_p(m)
      res_cost:= U_p(m)-min (t_p,\lambda U_p(m))
      as such this is a fairly mechanical switch of
      unary(i,alpha)*lambda with min(unary(i,alpha)*lambda,t[p])
    */
    g->add_node(nodes);
    for (int i = 0; i !=nodes; ++i)
      if(get_label(i)!=alpha){
        bool nalpha=false;
        //check if overlap_neighbours contain alpha
        for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'e')!=-1); ++j)
          nalpha|=(get_label(overlap_neighbour(i,j,'e'))==alpha);

        if(!nalpha)
          // if(nalpha) then boundary set must always contain alpha
          //and transition to I=alpha is free
          //and remaining `unary' cost comes in section 2
          //else
          {
            double update=min(unary(i,alpha)*lambda,thresh[i]);
            mexassert(update>=0);
            add_tweights(nodes+i,update,0,'d');
            add_edge(i,nodes+i,0,update,'a');
            for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'e')!=-1); ++j)
              add_edge(overlap_neighbour(i,j,'e'),nodes+i,0,update,'r');
          }
      }
    //Half way --- we have added all costs associated with changing node n to alpha
    // we now need to add the costs to overlap_neighbours of keeping a node fixed

    vector<bool>nclass(hyp);
    for (int i = 0; i !=nodes; ++i){
      //      //clog<<"i "<<i<<endl;
      for (int j = 0; j !=hyp; ++j)
        nclass[j]=false;
      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'e')!=-1); ++j)
        nclass[get_label(overlap_neighbour(i,j,'e'))]=true; //check overlap_neighbours
      nclass[get_label(i)]=true;//check self

      for (int j = 0; j !=hyp; ++j){//FIXME --SLOW
        if(nclass[j]&&(j!=alpha)){
          int temp=g->add_node(1);
          double update=min(unary(i,j)*lambda,thresh[i]);
          add_tweights(temp,0,update,'e');
          for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'e')!=-1); ++k)
            if(get_label(overlap_neighbour(i,k,'e'))==j)
              add_edge(overlap_neighbour(i,k,'e'),temp,update,0,'c');
          if(get_label(i)==j){
            add_edge(i,temp,update,0,'d');
            add_tweights(i,unary(i,alpha)-min(unary(i,alpha)*lambda,thresh[i]),
                         unary(i,j)-update,'s');
          }
        }
      }
    }
  }

  int active_hyp(){
    used_hyp.resize(hyp);
    for (int i = 0; i != hyp; ++i)
      used_hyp[i]=0;
    for (int i = 0; i != nodes; ++i)
      used_hyp[get_label(i)]=1;
    total_used=0;
    for (int i = 0, ind=0; i != hyp; ++i)
      if (used_hyp[i]){
        used_hyp[i]=++total_used;
        //        cout<<i<<" ";
      }
    //    cout<<endl;
    return total_used;
  }

  void construct_mdl_boykov(const int alpha){
    int temp= g->add_node(active_hyp())-1;
    //Warning active_hyp is important, must be run once for each iteration
    //Removing this function call is *very* bad.

    for (int i = 0; i !=hyp; ++i)
      if (used_hyp[i]&&(i!=alpha)&&((hweight(i)>0)||(overlap_penalty>0)))
        add_tweights(temp+used_hyp[i],0,hweight(i),'z');//2*overlap_penalty

    for (int j = 0; j !=nodes; ++j){
      int beta= get_label(j);
      if (beta!=alpha&&((hweight(beta)>0)||(overlap_penalty>0))){
        add_edge(j,temp+used_hyp[beta],hweight(beta),0,'l');//2*overlap_penalty
        //        +2*
      }
    }
  }

  void construct_mdl_standard(const int alpha){
    //Not needed see Yuri Boykov's work on MDL priors
    //Left for debugging
    construct_mdl_boykov(alpha);

    if(!used_hyp[alpha] && hweight(alpha)){
      int temp=g->add_node();
      add_tweights(temp,hweight(alpha),0,'a');
      for (int i = 0; i !=nodes; ++i)
        add_edge(i,temp,0,hweight(alpha),'m');
    }

  }

  void construct_pairwise(const int alpha){//Must be called after construct_multi
    //printf("Contructing graph for alpha=%d\n",alpha);
    for ( int i = 0; i !=nodes; ++i){
      //printf("Node %d labelled %d \n",i,get_label(i));
      if(get_label(i)!=alpha){
        //printf("not alpha\n");
        mexassert(get_label(i)<hyp);
        double cost;
        cost=(1-lambda)*(unary(i,get_label(i))-unary(i,alpha));
        for ( int j = 0; (j !=max_neigh_p)&&(pair_neighbour(i,j)!=-1); ++j){
          //printf(" %dth neighbour is  %d \n",j,pair_neighbour(i,j));
          if(get_label(pair_neighbour(i,j))==alpha){
            //printf("neighbour %d is taking label alpha\n",j);
            cost+=ncost(i,j);
          }
          else// if (i>pair_neighbour(i,j))
            {
              if (get_label(pair_neighbour(i,j))==get_label(i)){
                add_edge(i,pair_neighbour(i,j),ncost(i,j),ncost(i,j),'a');//Tautologically correct
              }
              else
                {
                  cost+=ncost(i,j);
                  add_edge(i,pair_neighbour(i,j),0,ncost(i,j),'a');
                  /*
                    X|1 0
                    ----
                    1|1 1
                    0|1 0
                    = x_1 + x_2 -x_1 x_2 = x_1 + (1-x_1)x_2
                  */
                }
            }
        }
        if(get_label(i)!=alpha){
          add_tweights(i,max(-cost,0.0),max(cost,0.0),'z'); // Add unaries backward.
          //printf("Added unary cost %f\n",cost);
        }
      }
    }
  }

  void set_labels(const int alpha){
    for (int i = 0; i !=nodes; ++i)
      if ((get_label(i)!=alpha)&&(g->what_segment(i,Graph<gtype,gtype,gtype>::SINK)
                                  ==Graph<gtype,gtype,gtype>::SINK))
        get_label(i)=alpha;
  }

  void debug_dump(){
    cout<<"  ";
    for (int k = 0; k != nodes; ++k)
      cout<<" "<<get_label(k);
    cout<<endl;
    for (int k = 0; k != hyp; ++k){
      int sum=0;
      for (int j = 0; j != nodes; ++j)
        sum+=get_mask(j,k);
      if (sum>0){
        cout<<k<<':';
        for (int j = 0; j != nodes; ++j)
          cout<<' '<<get_mask(j,k);
        cout<<endl;
      }
    }
  }
  void mask_test(){
    cout<<"Testing mask consistency."<<endl;
    for (int i = 0; i != nodes; ++i)
      //   if(get_label(i)==alpha)
      if (!get_mask(i, get_label(i))){
        cout<<"i is "<<i<<endl;
        cout<<"get_label(i) is "<<get_label(i)<<endl;
        //debug_dump();
        cout<<"Nhood"<<endl;
        for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'d')!=-1); ++k)
          cout<<"overlap_neighbour(i,k) is "<<overlap_neighbour(i,k,'d')<<endl;
        mexassert(get_mask(i,get_label(i))==1);
      }
    cout<<"Short test passed."<<endl;
    cout<<"Long test initiating"<<endl;
    for (int i = 0; i != nodes; ++i)
      for (int j = 0; j != hyp; ++j)
        if (get_mask(i, j)){
          bool flag=(get_label(i)==j);
          for (int k = 0; k != max_neigh_m &&overlap_neighbour(i,k,'e'!=-1); ++k)
            flag=flag||(get_label(overlap_neighbour(i,k,'f'))==j);
          if (!flag){
            cout<<"i is "<<i<<endl;
            cout<<"j is "<<j<<endl;
            cout<<"get_mask(i,j) is "<<get_mask(i,j)<<endl;
            cout<<"lambda is "<<lambda<<endl;
            cout<<"lambda*unary(i,j) is "<<lambda*unary(i,j)<<endl;
            for (int k = 0; k != max_neigh_m &&overlap_neighbour(i,k,'e')!=-1; ++k){
              cout<<"k is "<<k<<endl;
              cout<<"overlap_neighbour(i,k,'e') is "<<overlap_neighbour(i,k,'e')<<endl;
              cout<<"get_label(overlap_neighbour(i,k,'e')) is "<<get_label(overlap_neighbour(i,k,'e'))<<endl;
              cout<<"get_mask(i,get_label(overlap_neighbour(i,k,'e'))) is "<<get_mask(i,get_label(overlap_neighbour(i,k,'e')))<<endl;
            }
            mexassert(flag);
          }
        }
  }
  void set_mask(const int alpha){
    //    printf("setting mask for label %d\n", alpha);
    mexassert(_mcost);
    mexassert(mask.size()!=0);
    // cout<<"Init test"<<endl;
    debug_test(mask_test());
    vector<bool> nclass(hyp);
    int temp=2*nodes;
    for (int i = 0; i !=nodes; ++i){

      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'d')!=-1); ++j)
        nclass[get_label(overlap_neighbour(i,j,'d'))]=disable_mask||get_mask(i,get_label(overlap_neighbour(i,j,'d')));
      nclass[get_label(i)]=disable_mask||get_mask(i,get_label(i));//check self
      nclass[alpha]=false;
      //      for (int j = 0; j != hyp; ++j){
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if(nclass[j]){
          nclass[j]=false;
          mask[i+nodes*j]=((g->what_segment(temp++,Graph<gtype,gtype,gtype>::SINK)
                            ==Graph<gtype,gtype,gtype>::SOURCE)?lambda:0);
        }
      }
      if (nclass[get_label(i)])
        mask[i+nodes*get_label(i)]=((g->what_segment(temp++,Graph<gtype,gtype,gtype>::SINK)
                                     ==Graph<gtype,gtype,gtype>::SOURCE)?lambda:0);
    }

    for (int i = 0; i !=nodes; ++i)
      // if (get_label(i)!=alpha)
      mask[i+nodes*alpha]=((g->what_segment(nodes+ i, Graph<gtype,gtype,gtype>::SINK)
                            ==Graph<gtype,gtype,gtype>::SINK)?lambda:0);

    set_labels(alpha);
    if (lambda!= 1)
      for (int i = 0; i != nodes; ++i)
        mask[i+nodes*get_label(i)]=1;
    //    cout<<"final mask test"<<endl;
  }

  void expand(const int alpha) {
    //    disp_labels();
    //printf("Expanding on %d:\n",alpha);
    //printf(".");
    mexassert(alpha<hyp);
    g->reset();
    g->add_node(nodes);
    //    cout<<"In expand: overlap_penalty is "<<overlap_penalty<<endl;
    debug_test(cout<<"calling construct_robust_multi_with_mask"<<endl;)
      if (overlap_penalty)
        construct_robust_multi_with_mask(alpha);
      else if (_mcost)
        construct_robust_multi(alpha);
      else if (thresh)
        construct_multi_thresh(alpha);
      else
        construct_multi(alpha);
    debug_test(cout<<"calling construct_pairwise"<<endl;)
      if(max_neigh_p)
        construct_pairwise(alpha);

    //construct_mdl_boykov(alpha,overlap_penalty);
    debug_test(cout<<"calling construct_mdl_standard"<<endl;)
      construct_mdl_standard(alpha);
    debug_test(cout<<"calling construct_sparse_overlap"<<endl;)
      if (overlap_penalty)
        construct_sparse_overlap(alpha);
    debug_test(cout<<"calling maxflow"<<endl;)
      g->maxflow();
    debug_test(cout<<"calling set_mask"<<endl;)
      if (overlap_penalty)
        set_mask(alpha);
      else
        set_labels(alpha);

  }

  void disp_labels(){
    for (int i = 0; i !=nodes; ++i)
      printf("%d ",get_label(i));
    printf("\n");
  }

  void annotate (double* out){
    mexassert(out);

    int num_breaking = 0;

    if (overlap_penalty){
      for (int i = 0; i != nodes*hyp; ++i)
        out[i]=mask[i];
      return;
    }

    for (int i = 0; i !=nodes*hyp; ++i)
      out[i]=0;

    if (_mcost){
      vector<double> thresh(hyp);
      for (int i = 0; i !=nodes; ++i)
        { //robust cost
          for (int h =0;h!=hyp;++h)
            thresh[h]=0;
          for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'f')!=-1); ++j)
            thresh[get_label(overlap_neighbour(i,j,'f'))]+=_mcost[i*max_neigh_m+j];
          for (int h =0;h!=hyp;++h)
            if (lambda*unary(i,h)<thresh[h])
              out[MASK_ACCESS(i,h)]=lambda;
          out[MASK_ACCESS(i,get_label(i))]=1;
        }
    }
    else if (!thresh)//default
      for (int i = 0; i !=nodes; ++i){
        for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'f')!=-1); ++j)
          out[i+nodes*get_label(overlap_neighbour(i,j,'f'))]
            =max(out[MASK_ACCESS(i,get_label(overlap_neighbour(i,j,'f')))],lambda);
        out[MASK_ACCESS(i,get_label(i))]=1;
      }
    else//per neighbourhood rejection
      for (int i = 0; i !=nodes; ++i){
        for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'f')!=-1); ++j)
          {
            out[MASK_ACCESS(i,get_label(overlap_neighbour(i,j,'f')))]
              = max(out[i+nodes*get_label(overlap_neighbour(i,j,'f'))],
                    unary(i,get_label(overlap_neighbour(i,j,'f')))<thresh[i]?lambda:0);
            if(thresh[i]<=unary(i,get_label(overlap_neighbour(i,j,'f')))){
              cout<<"break edges, node and neighbor: "<<i<<" "<<overlap_neighbour(i,j,'f')<<endl;
              num_breaking++;
            }
          }
        out[MASK_ACCESS(i,get_label(i))]=1;
      }
    return ;
  }

  void debug_overlap(int alpha){
    cout<<"Expanding on "<<alpha<<endl;
    cout<<"The models: ";
    cout<<endl;

    double count=0;
    vector <bool> pairs(hyp*hyp,0);
    //use neighbours to avoid searching over all hyp
    //
    //   for (int j = 0; j != hyp; ++j)
    //     if (j!=get_label(i)&&get_mask(i,j))
    //       pairs[get_label(i)*hyp+j]=1;
    for (int i = 0; i != nodes; ++i){
      int l2=get_label(i);
      if(l2==alpha)
        for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'g')!=-1); ++j){
          int l1=get_label(overlap_neighbour(i,j,'g'));
          if ((l1!=l2)&&get_mask(i,l1))
            if(!pairs[l1*hyp+l2]){
              pairs[l1*hyp+l2]=1;
              cout<<l1<<' ';
              count++;
            }
        }
    }
    cout<<" overlap with the interior of "<<alpha<<endl;

    double cost=0;
    cout<<"The interior of classes"<<endl;
    //use neighbours to avoid searching over all hyp
    //
    //   for (int j = 0; j != hyp; ++j)
    //     if (j!=get_label(i)&&get_mask(i,j))
    //       pairs[get_label(i)*hyp+j]=1;
    for (int i = 0; i != nodes; ++i){
      int l2=get_label(i);
      for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'g')!=-1); ++j){
        int l1=get_label(overlap_neighbour(i,j,'g'));
        if (l1==alpha)
          if ((l1!=l2)&&get_mask(i,l1))
            if(!pairs[l1*hyp+l2]){
              pairs[l1*hyp+l2]=1;
              cout<<l2<<' ';
              count++;
            }
      }
    }
    cout<<" contain model "<<alpha<<endl;
    cout<<"Total violations on "<<alpha<<" : "<<count<<endl;
    cout<<"All other violations:"<<endl;
    cout<<"B  I"<<endl;
    int count2=0;
    for (int i = 0; i != nodes; ++i){
      int l2=get_label(i);
      if (l2!=alpha)
        for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'g')!=-1); ++j){
          int l1=get_label(overlap_neighbour(i,j,'g'));
          if (l1!=alpha)
            if ((l1!=l2)&&get_mask(i,l1))
              if(!pairs[l1*hyp+l2]){
                pairs[l1*hyp+l2]=1;
                cout<<l1<<' '<<l2<<endl;
                count2++;
              }
        }
    }
    cout<<"All other violations "<<count2<<endl;
  }
  double overlap_pen_cost(){
    double cost=0;
    vector <bool> pairs(hyp*hyp,0);
    //use neighbours to avoid searching over all hyp
    //
    //   for (int j = 0; j != hyp; ++j)
    //     if (j!=get_label(i)&&get_mask(i,j))
    //       pairs[get_label(i)*hyp+j]=1;
    for (int i = 0; i != nodes; ++i)
      for (int j = 0; (j !=max_neigh_m&&overlap_neighbour(i,j,'g')!=-1); ++j){
        int l1=get_label(overlap_neighbour(i,j,'g'));
        int l2=get_label(i);
        if ((l1!=l2)&&get_mask(i,l1))
          if(!pairs[l1*hyp+l2]){
            pairs[l1*hyp+l2]=1;
            cost++;
          }
      }

    // for (int i = 0; i != hyp; ++i)
    //   for (int j = 0; j != hyp; ++j)
    //     cost+=pairs[i*hyp+j];

    // for (int i = 0; i != hyp; ++i){

    // }
    debug_test(cout<<"There are "<<cost<<" overlapping models."<<endl;)
      cost*=overlap_penalty;

    return cost;
  }
  double mask_cost(){
    double cost=0;
    debug_test(mask_test());
    vector<bool>nclass(hyp);

    for (int i = 0; i != nodes; ++i) {//Unary costs
      for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'b')!=-1); ++j)
        nclass[get_label(overlap_neighbour(i,j,'b'))]=true; //check overlap_neighbours
      nclass[get_label(i)]=false;//ignore self

      for(int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if (!get_mask(i,j))
          cost+=_mcost[i*max_neigh_m+k];
        else if(nclass[j]){
          nclass[j]=false;
          cost+=lambda*unary(i,j);
        }
      }
      cost+=unary(i,get_label(i));
      debug_test(if(get_mask(i,get_label(i))!=1){
          cout<<"i is "<<i<<endl;
          cout<<"get_label(i) is "<<get_label(i)<<endl;
          cout<<"get_mask(i,get_label(i)) is "<<get_mask(i,get_label(i))<<endl;
        }
        mexassert(get_mask(i,get_label(i))==1);)
        }

    debug_test(cout<<"unary cost is "<<cost<<endl);
    return cost;
  }

  double thresh_cost(){
    double cost=0;
    vector<double> lweight(hyp);
    vector<bool> nclass(hyp);
    for ( int i = 0; i != nodes ; ++i){
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        lweight[j]=0;
        nclass[j]=true;
      }
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        lweight[j]+=_mcost[i*max_neigh_m+k];
      }
      nclass[get_label(i)]=false;
      for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
        unsigned int j=get_label(overlap_neighbour(i,k,'b'));
        if (nclass[j]){
          nclass[j]=false;
          cost+=min(lambda*unary(i,j),lweight[j]);
        }
      }
      cost+=unary(i,get_label(i));
    }
    return cost;
  }
  double cost() {
    /* Start with this function if you want to understand the code.
     */
    double cost=0;
    vector<bool>nclass(hyp);
    if (overlap_penalty){
      cost = mask_cost();
      //double epsilon=1.000001;
      // double c2= thresh_cost();
      // if ((cost>c2*epsilon)||(c2>cost*epsilon)){
      //   cout<<"mask_cost is "<<cost<<endl;
      //   cout<<"thresh_cost is "<<c2<<endl;
      //   debug_dump();
      // }
      // mexassert(cost<=c2*epsilon);
      // mexassert(c2<=cost*epsilon);
      cost+=overlap_pen_cost();
    }
    else if (_mcost)
      cost=thresh_cost();
    else
      for ( int i = 0; i != nodes ; ++i){
        for (int j = 0; (j !=max_neigh_m)&&(overlap_neighbour(i,j,'b')!=-1); ++j)
          nclass[get_label(overlap_neighbour(i,j,'b'))]=true; //check overlap_neighbours
        nclass[get_label(i)]=false;//deal with self in next line
        cost+=unary(i,get_label(i));
        if(!thresh) //default case
          for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
            unsigned int j=get_label(overlap_neighbour(i,k,'b'));
            if(nclass[j]){
              nclass[j]=false;
              cost+=lambda*unary(i,j);
            }
          }
        else //per neighbourhood rejection
          for (int k = 0; (k !=max_neigh_m)&&(overlap_neighbour(i,k,'b')!=-1); ++k){
            unsigned int j=get_label(overlap_neighbour(i,k,'b'));
            if(nclass[j]){
              nclass[j]=false;
              cost+=min(thresh[i],lambda*unary(i,j));
            }
          }
      }

    //pairwise cost
    for ( int i = 0; i != nodes ; ++i)
      for ( int j = 0; (j !=max_neigh_p)&&(pair_neighbour(i,j)!=-1); ++j)
        if (get_label(i)!=get_label(pair_neighbour(i,j)))
          cost+=ncost(i,j);

    vector<bool>lcost(hyp);
    for (int i = 0; i != hyp; ++i)
      lcost[i]=false;

    for (int i = 0; i !=nodes; ++i){
      mexassert(get_label(i)<hyp);
      mexassert(get_label(i)>=0);
      lcost[get_label(i)]=true;
    }
    // //mdl cost
    for (int i = 0; i !=hyp; ++i)
      if (lcost[i])
        cost+=hweight(i);
    return cost;
  }

  void debug_solve(){
    if (overlap_penalty){
      mask.resize(nodes*hyp);
      double temp=overlap_penalty;
      overlap_penalty=0;
      annotate(&mask[0]);
      overlap_penalty=temp;
    }
    double c=cost();
    set_scalar(c);
    double c2=INFINITY;
    int oldlabel[nodes];
    for (int i = 0; i !=nodes; ++i)
      oldlabel[i]=get_label(i);
    if(overlap_penalty)
      backup_mask();

    int i=0;
    while ((c2>c)||(c==INFINITY)){
      c2=c;
      double temp=0;

      for (int j = 0; j !=hyp; ++j){
        printf("i: %d, j: %d, hyp %d\n",i,j,hyp);
        //        cout<<<<endl;
        // cout<<"--------------------"<<endl;
        debug_overlap(j);
        expand(j);
        //      annotate(grid);
        temp=cost();
        cout<<"cost is "<<temp<<endl;
        // for (int k = 0; k != nodes; ++k)
        //   cout<<get_label(k)<<endl;
        //mexassert(temp==new_cost());
        debug_overlap(j);
        // cout<<"--------------------"<<endl;
        if (temp>c){
          char a[80];
          // mexPrintf("Cost increasing j: %d i: %d c: %f temp: %f",j,i,c,temp);
          printf("Cost increasing j: %d i: %d c: %f temp: %f",j,i,c,temp);
          mexassert(temp<c*1.00001);
          if(overlap_penalty)
            void_mask();
          for (int i = 0; i !=nodes; ++i)
            get_label(i)=oldlabel[i];
          if(overlap_penalty)
            restore_mask();
          temp=c;
          mexassert(temp==cost())
            }
        else {
          if (temp<c){
            // mexPrintf ("Cost decreasing to %f\n",temp);
            printf("Cost decreasing to %f\n",temp);
            i=0;
          }
          for (int i = 0; i !=nodes; ++i)
            oldlabel[i]=get_label(i);
          if(overlap_penalty)
            backup_mask();
          c=temp;
          set_scalar(c);
        }
        ++i;
        if (i==hyp){
          c2=c;
          return;
        }
      }
    }
  }

  void solve(){
    if (overlap_penalty){
      mask.resize(nodes*hyp);
      double temp=overlap_penalty;
      overlap_penalty=0;
      annotate(&mask[0]);
      overlap_penalty=temp;
    }
    double c=cost();
    set_scalar(c);
    double c2=INFINITY;
    int oldlabel[nodes];
    for (int i = 0; i !=nodes; ++i)
      oldlabel[i]=get_label(i);
    if(overlap_penalty)
      backup_mask();

    int i=0;
    while ((c2>c)||(c==INFINITY)){
      c2=c;
      double temp=0;

      for (int j = 0; j !=hyp; ++j){
        expand(j);
        temp=cost();
        if (temp>c){
          if(overlap_penalty)
            void_mask();
          for (int i = 0; i !=nodes; ++i)
            get_label(i)=oldlabel[i];
          if(overlap_penalty)
            restore_mask();
          temp=c;
        }
        else
          {
            if (temp<c){
              //printf("Cost decreasing to %f\n",temp);
              // cout<<"Cost decreasing to "<< temp << endl;
              i=0;
            }
            for (int i = 0; i !=nodes; ++i)
              oldlabel[i]=get_label(i);
            if(overlap_penalty)
              backup_mask();
            c=temp;
            set_scalar(c);
          }
        ++i;
        if (i==hyp){
          c2=c;
          return;
        }
      }
    }
  }

  void solve_fast(){
    if (overlap_penalty){
      mask.resize(nodes*hyp);
      cout<<"overlap_penalty is still"<<overlap_penalty<<endl;
      double temp=overlap_penalty;
      overlap_penalty=0;
      annotate(&mask[0]);
      overlap_penalty=temp;
    }

    double c=cost();
    set_scalar(c);

    double c2=INFINITY;
    int oldlabel[nodes];
    for (int i = 0; i !=nodes; ++i)
      oldlabel[i]=get_label(i);

    c2=c;
    double temp=0;

    for (int j = 0; j !=hyp; ++j){
      expand(j);
      temp=cost();

      if (temp>c){
        for (int i = 0; i !=nodes; ++i)
          get_label(i)=oldlabel[i];
        if(overlap_penalty)
          restore_mask();
        temp=c;
      }
      else
        {
          for (int i = 0; i !=nodes; ++i)
            oldlabel[i]=get_label(i);
          if(overlap_penalty)
            backup_mask();
          c=temp;
          set_scalar(c);
        }

    }
  }
};
