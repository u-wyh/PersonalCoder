#include <bits/stdc++.h>
using namespace std;
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);int n,m;cin>>n>>m;vector<int>p(n+1),sz(n+1,1);iota(p.begin(),p.end(),0);function<int(int)>f=[&](int x){return p[x]==x?x:p[x]=f(p[x]);};while(m--){int z,x,y;cin>>z>>x>>y;x=f(x);y=f(y);if(z==1){if(x!=y){if(sz[x]<sz[y])swap(x,y);p[y]=x;sz[x]+=sz[y];}}else cout<<(x==y?'Y':'N')<<'\n';}}
