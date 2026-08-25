#include <bits/stdc++.h>
using namespace std;
int main(){int h,w;cin>>h>>w;vector<string>a(h+2,string(w+2,'0'));for(int i=1;i<=h;++i){string s;cin>>s;for(int j=1;j<=w;++j)a[i][j]=s[j-1];}vector<vector<int>>vis(h+2,vector<int>(w+2));queue<pair<int,int>>q;q.push({0,0});vis[0][0]=1;int ans=0,dx[4]={1,-1,0,0},dy[4]={0,0,1,-1};while(!q.empty()){auto [x,y]=q.front();q.pop();for(int d=0;d<4;++d){int nx=x+dx[d],ny=y+dy[d];if(nx<0||nx>h+1||ny<0||ny>w+1)continue;if(a[nx][ny]=='1')++ans;else if(!vis[nx][ny])vis[nx][ny]=1,q.push({nx,ny});}}cout<<ans<<'\n';}
