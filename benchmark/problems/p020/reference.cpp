#include <bits/stdc++.h>
using namespace std;
int main(){ios::sync_with_stdio(false);cin.tie(nullptr);int n;cin>>n;vector<long long>s(n);for(int i=0;i<n;++i){cin>>s[i];if(i)s[i]+=s[i-1];}int m;cin>>m;while(m--){long long x;cin>>x;cout<<lower_bound(s.begin(),s.end(),x)-s.begin()+1<<'\n';}}
