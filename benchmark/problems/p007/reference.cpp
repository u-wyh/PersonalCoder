#include <bits/stdc++.h>
using namespace std;
int main(){string s,t;cin>>s;for(char c:s)if(isdigit(c))t+=c;sort(t.begin(),t.end());for(int i=0;i<(int)t.size();++i)cout<<(i?"+":"")<<t[i];cout<<'\n';}
