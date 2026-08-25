#include <bits/stdc++.h>
using namespace std;
int main(){int n;cin>>n;vector<int>c(n);for(int&x:c)cin>>x;int lim=n==1?2:c[n-1]+c[n-2],INF=1e9;vector<int>dp(lim+1,INF);dp[0]=0;for(int x=1;x<=lim;++x)for(int v:c)if(v<=x)dp[x]=min(dp[x],dp[x-v]+1);for(int x=1;x<=lim;++x){int y=x,g=0;for(int i=n-1;i>=0;--i)g+=y/c[i],y%=c[i];if(g!=dp[x]){cout<<"non-canonical\n";return 0;}}cout<<"canonical\n";}
