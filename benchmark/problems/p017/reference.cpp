#include <bits/stdc++.h>
using namespace std;
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);string s,p;cin>>s>>p;int m=p.size();vector<int>pi(m);for(int i=1,j=0;i<m;++i){while(j&&p[i]!=p[j])j=pi[j-1];if(p[i]==p[j])++j;pi[i]=j;}for(int i=0,j=0;i<(int)s.size();++i){while(j&&s[i]!=p[j])j=pi[j-1];if(s[i]==p[j])++j;if(j==m){cout<<i-m+2<<'\n';j=pi[j-1];}}for(int i=0;i<m;++i)cout<<pi[i]<<(i+1==m?'\n':' ');}
