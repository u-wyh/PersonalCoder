#include <bits/stdc++.h>
using namespace std;
int main(){int n,x=0;cin>>n;while(n--){string s;cin>>s;x+=s.find('+')!=string::npos?1:-1;}cout<<x<<'\n';}
