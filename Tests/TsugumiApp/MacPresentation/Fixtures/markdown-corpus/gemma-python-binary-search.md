Pythonで二分探索（Binary Search）を実装する方法を、コードと詳細な解説、および計算量とともに説明します。

### 二分探索とは？

二分探索は、**ソート（並び替え）済みのリスト**の中から、特定の要素を効率的に探すアルゴリズムです。
探索範囲を毎回「半分」に絞り込んでいくため、非常に高速に動作するのが特徴です。

---

### Pythonによる実装（逐次的な実装）

一般的に、メモリ効率が良い「繰り返し（while文）を用いた実装」がよく使われます。

```python
def binary_search(arr, target):
    """
    二分探索を行う関数
    
    Args:
        arr (list): ソート済みのリスト
        target: 探したい値
        
    Returns:
        int: 見つかった場合はそのインデックス、見つからない場合は -1
    """
    left = 0                  # 探索範囲の左端のインデックス
    right = len(arr) - 1      # 探索範囲の右端のインデックス

    while left <= right:
        # 中央のインデックスを計算（切り捨て）
        mid = (left + right) // 2
        
        # 中央の値がターゲットと同じ場合
        if arr[mid] == target:
            return mid
        
        # 中央の値がターゲットより小さい場合、右半分を探索
        elif arr[mid] < target:
            left = mid + 1
            
        # 中央の値がターゲットより大きい場合、左半分を探索
        else:
            right = mid - 1

    # 見つからなかった場合
    return -1

# --- テストコード ---
if __name__ == "__main__":
    # 二分探索は必ず「ソート済み」である必要がある
    sorted_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target_value = 7

    result = binary_search(sorted_list, target_value)

    if result != -1:
        print(f"ターゲット {target_value} はインデックス {result} にあります。")
    else:
        print(f"ターゲット {target_value} はリスト内に見つかりませんでした。")
```

---

### コードの解説

1.  