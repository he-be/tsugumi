C++における `std::vector` は**「可変長配列」**、`std::map` は**「連想コンテナ（辞書型）」**です。

それぞれの特徴と使い方を簡潔に説明します。

---

### 1. std::vector<int> （動的配列）
データを順番に並べて保持するためのものです。配列のように「インデックス（0, 1, 2...）」を使ってアクセスします。

*   **特徴:** 要素の追加・削除が柔軟。インデックスによるアクセスが非常に高速。
*   **主な操作:** `push_back()` (追加), `size()` (要素数), `[]` (要素へのアクセス)。

#### コード例
```cpp
#include <iostream>
#include <vector> // vectorを使うために必要

int main() {
    // 初期化
    std::vector<int> numbers = {10, 20, 30};

    // 要素の追加
    numbers.push_back(40);
    numbers.push_back(50);

    // 要素へのアクセス (インデックスを使用)
    std::cout << "2番目の要素: " << numbers[1] << std::endl; // 20

    // 要素数の取得
    std::cout << "要素数: " << numbers.size() << std::endl; // 5

    // 全要素のループ処理
    std::cout << "全要素: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

---

### 2. std::map<std::string, int> （連想コンテナ）
「キー（Key）」と「値（Value）」をペアにして保持するものです。今回の場合は「名前（string）」で「数値（int）」を管理する辞書のようなイメージです。

*   **特徴:** キーを使って値を高速に検索できる。自動的にキーの順（辞書順）にソートされる。
*   **主な操作:** `[]` (値の代入・アクセス), `find()` (検索), `erase()` (削除)。

#### コード例
```cpp
#include <iostream>
#include <map>    // mapを使うために必要
#include <string> // stringを使うために必要

int main() {
    // 初期化 (名前 -> 点数 というイメージ)
    std::map<std::string, int> scores;

    // 要素の追加・更新
    scores["Alice"] = 90;
    scores["Bob"] = 85;
    scores["Charlie"] = 70;

    // 値の取得
    std::cout << "Aliceのスコア: " << scores["Alice"]