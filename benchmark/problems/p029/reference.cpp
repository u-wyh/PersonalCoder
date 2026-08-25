#include <bits/stdc++.h>
using namespace std;
int main(){int n;cin>>n;vector<vector<int>>a(2,vector<int>(n));int hi=0;for(auto&r:a)for(int&x:r)cin>>x,hi=max(hi,x);auto ok=[&](int lim){for(auto&r:a){int pending=0;for(int x:r)if(x>lim){if(!pending)pending=x;else if(pending==x)pending=0;else return false;}if(pending)return false;}return true;};int lo=-1;while(hi-lo>1){int mid=lo+(hi-lo)/2;if(ok(mid))hi=mid;else lo=mid;}cout<<hi<<'\n';}
