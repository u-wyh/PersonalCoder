#include <bits/stdc++.h>
using namespace std;
int main(){int a[10],h,ans=0;for(int&x:a)cin>>x;cin>>h;for(int x:a)ans+=x<=h+30;cout<<ans<<'\n';}
