#include <bits/stdc++.h>
using namespace std;
int main(){int n,k;cin>>n>>k;vector<int>a(n);for(int&x:a)cin>>x;sort(a.begin(),a.end());if(k==0)cout<<(a[0]>1?1:-1);else if(k==n)cout<<a.back();else cout<<(a[k-1]<a[k]?a[k-1]:-1);cout<<'\n';}
