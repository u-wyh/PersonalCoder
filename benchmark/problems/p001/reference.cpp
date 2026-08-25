#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    long long sum = 0, value;
    while (n--) {
        cin >> value;
        sum += value;
    }
    cout << sum << '\n';
    return 0;
}
