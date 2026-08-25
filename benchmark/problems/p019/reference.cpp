#include <bits/stdc++.h>
using namespace std;
int main(){int n;cin>>n;long long last,x;cin>>last;int cur=1,ans=1;for(int i=1;i<n;++i){cin>>x;cur=x>=last?cur+1:1;ans=max(ans,cur);last=x;}cout<<ans<<'\n';}
