#include <bits/stdc++.h>
using namespace std;using ll=long long;
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);int n,m;cin>>n>>m;vector<ll>t(n+1);auto add=[&](int x,ll v){for(;x<=n;x+=x&-x)t[x]+=v;};auto sum=[&](int x){ll r=0;for(;x;x-=x&-x)r+=t[x];return r;};for(int i=1;i<=n;++i){ll x;cin>>x;add(i,x);}while(m--){int op,x;ll y;cin>>op>>x>>y;if(op==1)add(x,y);else cout<<sum((int)y)-sum(x-1)<<'\n';}}
