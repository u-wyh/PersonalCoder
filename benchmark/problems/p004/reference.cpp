#include <bits/stdc++.h>
using namespace std;
int main(){int n,k;cin>>n>>k;vector<int>a(n);for(int&x:a)cin>>x;int ans=0;for(int x:a)ans+=x>0&&x>=a[k-1];cout<<ans<<'\n';}
