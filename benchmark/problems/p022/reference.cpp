#include <bits/stdc++.h>
using namespace std;const long long NEG=-(1LL<<60);
int main(){int n,k;while(cin>>n>>k&&n){vector<array<long long,3>>dp(k+1),ndp(k+1);for(auto&x:dp)x={NEG,NEG,NEG};dp[0][0]=0;for(int i=0;i<n;++i){long long l,r;cin>>l>>r;for(auto&x:ndp)x={NEG,NEG,NEG};for(int c=0;c<=k;++c)for(int s=0;s<3;++s)if(dp[c][s]>NEG){ndp[c][0]=max(ndp[c][0],dp[c][s]+l+r);if(c<k&&s!=2)ndp[c+1][1]=max(ndp[c+1][1],dp[c][s]+r);if(c<k&&s!=1)ndp[c+1][2]=max(ndp[c+1][2],dp[c][s]+l);}dp.swap(ndp);}cout<<max({dp[k][0],dp[k][1],dp[k][2]})<<'\n';}}
