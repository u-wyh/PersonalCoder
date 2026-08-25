#include <bits/stdc++.h>
using namespace std;
int main(){priority_queue<long long>lo;priority_queue<long long,vector<long long>,greater<long long>>hi;long long x;while(cin>>x){if(lo.empty()||x<=lo.top())lo.push(x);else hi.push(x);if(lo.size()<hi.size())lo.push(hi.top()),hi.pop();if(lo.size()>hi.size()+1)hi.push(lo.top()),lo.pop();cout<<(lo.size()==hi.size()?(lo.top()+hi.top())/2:lo.top())<<'\n';}}
