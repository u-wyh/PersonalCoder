#include <bits/stdc++.h>
using namespace std;using ll=long long;const ll INF=4e18;
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);int n,m,s;cin>>n>>m>>s;vector<vector<pair<int,int>>>g(n+1);while(m--){int u,v,w;cin>>u>>v>>w;g[u].push_back({v,w});}vector<ll>d(n+1,INF);priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<pair<ll,int>>>q;d[s]=0;q.push({0,s});while(!q.empty()){auto [du,u]=q.top();q.pop();if(du!=d[u])continue;for(auto [v,w]:g[u])if(d[v]>du+w)d[v]=du+w,q.push({d[v],v});}for(int i=1;i<=n;++i)cout<<(d[i]==INF?2147483647:d[i])<<(i==n?'\n':' ');}
