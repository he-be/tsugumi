# 23. Phase 5 — XML 形のツール呼び出しと GBNF (実測(手元)、2026-08-22)

[04-PHASES.md](04-PHASES.md) 次の一手 **#22**。Phase 5 に残っていた
**ツール呼び出しのパーサと GBNF ビルダ**、および `GrammarVocabulary` の
piece 復元を ByteLevel に切り替えた。[22 §4-2](22-PHASE5-TOKENIZER.md) の
`tools` の JSON もここで再訪し、**記述が 1 つ間違っていたので訂正する** (§5-1)。

```
swift run -c release TurboFieldfareKernelCheck \
  --qwen-tools scratch/ornith-oq4e-g64.gturbo
```

| | |
| --- | --- |
| 中心の主張 | **テンプレート自身が描いた呼び出しを、文法が受理し、パーサが同じ引数に戻す。**呼び出しはこの検査が書くのではなく、`toolCalls` を持つ `Message` から `chat_template.jinja` が描く — サーバーが次のターンに送り返すのと同じ描画である (INV-1) |
| 書いたもの | `QwenToolCallParser` / `QwenStructuredAssistantDecoder` (§1)、`QwenToolCallGrammar` + `QwenChatGrammarBuilder` (§2)、`GrammarVocabulary` の ByteLevel 入口 (§3) |
| 検査 | `--qwen-tools` が **36 本すべて緑、うち 6 本は負例** (§4)。`swift test --no-parallel` は **1,329 件** (1,297 + 新規 32) すべて緑 |
| Gemma | **1 行も動かしていない。**`ChatGrammarBuilder` / `GemmaToolCallParser` / `StructuredAssistantDecoder` は無変更で、`GrammarVocabulary` は既存の入口を私有 init に畳んだだけ (§3)。既定 69 本と `swift test` は緑のまま |
| 残り | サーバーの結線 (Phase 8) と CLI の `--tools`。**閉じていない往復が 2 つある** (§5-2) |

---

## 1. パーサ — 値の綴りは**宣言された型**で決まる

形式はテンプレートが固定している (実測(上流)):

```
<tool_call>
<function=get_weather>
<parameter=city>
Kyoto
</parameter>
<parameter=days>
3
</parameter>
</function>
</tool_call>
```

`GemmaToolCallParser` との構造的な差は 1 つだけで、それが全部を決める。
テンプレートはこう書く:

```jinja
{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
```

**文字列の引数は生で書かれる** — 引用符もエスケープも無い。`Kyoto` であって
`"Kyoto"` ではない。それ以外は JSON で書かれる。したがって:

- 文字列 `"3"` と整数 `3` は、**綴りだけでは区別が付かない**
- 文字列 `hello` は、JSON として読めば構文エラーである

**だからパーサはツールのスキーマを要る。**`QwenToolCallParser(tools:)` は
名前の集合ではなく `name → parameters` の表を持ち、`type == "string"` の
パラメータだけを生として読む。Gemma 側が `allowedTools: Set<String>` で
足りていたのは、あちらの値が常に JSON 形だったからである。

型が分からないときの規則は 1 段だけ:

| そのキーのスキーマ | 読み方 |
| --- | --- |
| `type: "string"` (または全部 string の union) | **生**。引用符も空白も値の一部 |
| その他の `type` が宣言されている | JSON。読めなければ `unparsableArgument` で**落とす** (推測しない) |
| `type` が無い / 宣言されていないキー | JSON を試し、駄目なら文字列 |

値の終端は **最初の `\n</parameter>`** を採る。「最後の」ではない —
文法が値に閉じ札を含ませないので (§2) 最初と最後は同じものであり、
最後を採る実装は**文法が出せない呼び出しを受理して、値の終わりについて
文法と食い違う**。

`QwenStructuredAssistantDecoder` は `StructuredAssistantDecoder` の兄弟で、
生成トークンを 1 個ずつ受けて reasoning / content / toolCall に分ける。
Gemma と違う点が 3 つある:

1. **推論ブロックは既に開いている。**テンプレートが生成プロンプトに
   `<think>\n` を書くので、最初の生成トークンが既に推論の中にあり、
   `</think>` だけが見える。`startsInReasoning` は呼び出し側が
   プロンプトの末尾のマーカーから読む
2. **マーカーは special ではない。**`tokenizer.json` は `<think>` /
   `</think>` / `<tool_call>` / `</tool_call>` を `special: false` と
   宣言しているので、`QwenDetokenizer` はその綴りを**テキストとして出す**。
   delta は「抱えていたバイト + マーカーの綴り」なので、綴りを剥がして
   前のバイトだけを**マーカーが状態を変える前のチャンネル**に流す
3. 本文はトークン ID で集め、`</tool_call>` で 1 度だけ復号する
   (Gemma と同じ。閉じマーカーをまたいで割れた符号点が本文に残る)

## 2. GBNF — 生の文字列は「**閉じ札を含まない任意のテキスト**」

`QwenToolCallGrammar` は `TurboFieldfare` 側に置いた (`ChatGrammarBuilder` は
サーバー側にある)。形式はチェックポイントのものであって HTTP 層のもの
ではなく、**実物の語彙に当てるのは `KernelCheck`** だからである。
サーバー側に残したのは要求の話 (`tool_choice` がどのツールを選ぶか、
lazy かどうか、`response_format` との衝突) だけで、`TurboFieldfareServerCore`
は NIO を引くので `KernelCheck` からは触れない。

規則の形:

```
root      ::= tool-call                             (lazy)
root      ::= !<[248058]>* tool-call                (非 lazy)
tool-call ::= tool-get-weather ("\n" tool-get-weather)*
tool-get-weather ::= <[248058]> "\n<function=get_weather>\n"
                     tool-get-weather-p-city tool-get-weather-p-days?
                     "</function>\n" <[248059]>
```

**マーカーは `<[ID]>` (TOKEN 要素) で書く。**綴りは通常トークンの列でも
到達できる (`<`, `tool`, `_`, `call`, `>`) ので、リテラルで書くと
その列も同じように受理してしまう。負例 `markersAsText` がこれを押さえる
(§4)。

### 2-1. 生の値 — 13 状態のオートマトン

文字列パラメータの値に禁じられているのは **`\n</parameter>` という列だけ**
である。素直な近似 `[^<]*` は文字列の中の `<` を全部禁じることになり、
マークアップやコードを編集するツールはそれで使えなくなる。

代わりに「その列を含まない任意のテキスト」を右線形文法で書いた。状態 `i` は
「末尾 `i` 文字が閉じ札の先頭 `i` 文字」の意味で、閉じ札の改行は先頭の
1 個しか無いので、外れたときの戻り先は **改行なら状態 1、それ以外は状態 0** に
なる:

```
text-value-0  ::= ( [^\n] text-value-0 | "\n" text-value-1 )?
text-value-1  ::= ( [^\n<] text-value-0 | "\n" text-value-1 | "<" text-value-2 )?
…
text-value-12 ::= ( [^\n>] text-value-0 | "\n" text-value-1 )?
```

**閉じ札を完成させる遷移が 1 本も無い**ことが、この構成そのものである。
どの状態も受理状態なので (値が `…\n</param` で終わってよい) 規則は全部
`?` が付く。分岐はすべて末尾再帰なので、照合器のスタックは値の長さで
伸びない。

`enum` の文字列は生で書かれるので、`JSONSchemaGrammar` が出す引用符付きの
選択肢ではなく**裸のリテラル**にする。

### 2-2. 文字列でない値は `.json` 方言のまま

`tojson` が書いたものなので JSON の展開がそのまま正しい。ただし `.json`
方言は**テンプレートが書かない空白を許す** (`space` 規則)。Gemma 側は
GEN-8 で空白を一切許さない方言を作ってこれを閉じたが、Qwen で同じことを
しても往復は閉じない (§5-2) ので、**方言を増やさず所見として登録する**方を
採った。空白が入り得るのは `array` / `object` 型の引数だけで、スカラーの
引数 (実際の大半) には入る場所が無い。

### 2-3. 非 lazy の前置きは「セクション開始でない任意のトークン」

thinking on だと最初の生成トークンは推論の中にいるので、`root` が呼び出し
だけを綴ると**許されるトークンが 1 つも無くなる** (GEN-7 が 500 にする)。
Gemma は思考チャンネルのブロックを前置きにした。Qwen では
`!<[thinkEnd]>* <[thinkEnd]>` という精密な形ではなく
**`!<[toolCallStart]>*`** にした。理由は 2 つあり、どちらも単独で決定的:

1. `</think>` の後にテンプレートは `\n\n` を書き、**このチェックポイント
   自身のシステムプロンプトが「関数呼び出しの前に自然言語で理由を書いて
   よい」と明記している**。それを禁じる文法は、モデルが読んでいる
   プロンプトと矛盾する
2. 拘束するものが減らない。前置きの中では文法が未完成なので `mayEndHere` が
   偽で**停止トークンが拒まれる** — 出口は呼び出しを書くことしか無い

## 3. `GrammarVocabulary` の piece を ByteLevel に

piece 表 (トークン ID → 出力に足すバイト列) の作り方は家族ごとに違うので、
`GFTokenizer` 版に分岐を足すのではなく**別の入口**にした
(`init(_ tokenizer: QwenTokenizer)`)。表から先 — 足切り (空 piece と NUL
始まり)、候補の採番、UTF-8 の事前復号 — は参照実装の
`llama_grammar_apply_impl` と同じ 1 本のままで、私有 init に畳んである。

| | Gemma | Ornith |
| --- | --- | --- |
| 通常トークン | metaspace `▁` を空白に直した綴り | **GPT-2 の byte↔unicode 表を逆引き** |
| バイト | `<0xXX>` の run だけ | **全部** |
| 追加トークン | 綴りそのもの | 綴りそのもの |

**Gemma の規則で表を作っても落ちない。**`こんにちは` の piece が
`ãģĵãĤĵãģ«ãģ¡ãģ¯` になり、文法が日本語を 1 文字も通さなくなるだけである。
落ちずに静かに通らなくなるので、負例 `gemmaPieceRules` でしか捕まらない
(§4)。

## 4. 検査

**純粋な部分は `swift test`** (トークナイザも語彙も要らない):

| 束 | 本数 | 中身 |
| --- | ---: | --- |
| `QwenToolCallParserTests` | 13 | 形・型付け・値の終端・壊れた枠 7 通り・重複キー・大きすぎる本文 |
| `QwenChatGrammarBuilderTests` | 19 | `tool_choice` 4 値、XML の形、キー昇順、必須と任意、`enum`、生の値 (`<b>bold</b>` など 7 種)、閉じ札の密輸、並列、`response_format` |

**語彙が要る部分は `--qwen-tools`** (36 本、うち 6 本は負例):

| 見るもの | 本数 | 中身 |
| --- | ---: | --- |
| piece 表 | 5 | 語彙数、マーカー 3 本の piece、**piece の連結 == 復号したテキスト** (5 検体) |
| `<tools>` の描画 | 2 | 区切りとキー順 (§5-1) |
| 往復 | 21 | 7 検体 × (XML の形 / 文法が受理 / パーサが同じ引数に戻す) |
| マスク | 1 | `fillAllowedMask` == `allows()` を全語彙 248,070 本 × 4 地点で |
| lazy の性質 | 1 | §7 |
| **負例** | **6** | 下表 |

| 負例 | 検出したもの |
| --- | --- |
| `gemmaPieceRules` (metaspace の規則を ByteLevel の語彙に) | 5 検体すべてで不一致。`"ãģĵãĤĵãģ«ãģ¡ãģ¯ãĢģä¸"` ≠ `"こんにちは、世界。"` |
| `markersAsText` (マーカーを通常トークン 4 + 4 本で綴る) | 非 lazy の文法が拒否 |
| `smuggledCloser` (値が自分のブロックを閉じる) | 文法もパーサも拒否 |
| `gemmaCallForm` (`call:name{…}`) | 拒否 |
| `undeclaredTool` | 拒否 |
| `missingRequired` (`city` を落とす) | 拒否 |

往復の 7 検体は ASCII / 日本語 / マークアップ (`<b>Kyoto</b> & co`) /
**閉じ札の一歩手前** (`a\n</paramete>\nb`) / 複数行 / 絵文字 / 文字列のみ。
日本語と絵文字は piece 表が ByteLevel でなければ通らず、マークアップと
「一歩手前」は値の規則が `[^<]*` なら通らない。

## 5. テンプレートとの往復 (INV-1)

### 5-1. **[22 §4-2](22-PHASE5-TOKENIZER.md) の訂正: swift-jinja のキー順は不定ではない**

[22 §4-2](22-PHASE5-TOKENIZER.md) は swift-jinja のキー順を
「辞書由来で**不定**」と書いた。**これは間違いである。**swift-jinja は
両端で並べ替える:

- `Jinja.Value(any:)` は Swift の `[String: Any?]` を `OrderedDictionary` に
  写すときに**キーを昇順に並べる** (`Value.swift`)
- `tojson` は `JSONEncoder` に `.sortedKeys` を入れて書き出す
  (`Filters.swift`)

実際の描画 (`--qwen-tools` が印字する):

| | `<tools>` の 1 行 |
| --- | --- |
| swift-jinja | `{"function":{"description":"Get the weather for a city.","name":"get_weather","parameters":{…}},"type":"function"}` |
| Python (`transformers`) | `{"type": "function", "function": {"name": "get_weather", "description": …, "parameters": {…}}}` |

**どちらも決定的で、綴りが 2 通りあるだけ**である。「片方が不定」ではない。

この訂正が効く場所がある。テンプレートは引数を
`tool_call.arguments|items` で回すが、`items` が見る `Value` は既に
昇順に並べ替えられている。**つまり再描画は必ずキー昇順である。**だから
文法が「プロパティを昇順で綴る」ことにしても、再描画とずれない
(§2 の並び)。不定だったら、この設計は成り立たなかった。

### 5-2. **閉じていない往復が 2 つ**

「生成した通りに描き直せる」(INV-1) は、Qwen ではまだバイト一致では
成り立たない。**どちらも「サーバーが次のターンにそのアシスタントターンを
描き直す」ときにしか効かない**ので、Phase 8 の宿題として置く。

| 残っているもの | 中身 | 閉じるなら |
| --- | --- | --- |
| **入れ子 JSON の空白** | `.json` 方言は `{"a": 1}` を許すが、swift-jinja は `{"a":1}` を書く。オブジェクト型・配列型の引数にだけ起きる | 空白を許さない方言を足す (Gemma の GEN-8 と同じ手) |
| **非 ASCII の `\uXXXX`** | swift-jinja の `tojson` は非 ASCII を `\uXXXX` に逃がす (`Filters.swift` の `escapeNonASCII`、既定 `ensure_ascii: true`)。**`transformers` は逃がさない** — `tojson` を `ensure_ascii=False` で上書きしている (`utils/chat_template_utils.py:449`、実測(上流))。入れ子の中の `東京` が片方だけ `\u6771\u4eac` になる | 描画側で逃がさないようにする |

**空白だけ閉じても往復は閉じない**ので、方言を増やす手は今回採らなかった。
非 ASCII の方は `<tools>` ブロックにも同じようにかかる (§5-1) — 説明文が
日本語のツールは、Swift 側だけ `\uXXXX` で見せることになる。
なお**トップレベルの文字列引数は生で書かれるので、この 2 つはどちらも
かからない** — 日本語の都市名は素通りする (§4 の往復検体がそれを見ている)。

`<tools>` ブロックの綴り (§5-1) は往復の話ではなく**学習分布の話**である。
モデルは Python の綴りで学習しているので、こちらは別の綴りのシステム
プロンプトを見せている。影響の測定は Phase 6 で品質を測るときに一緒に。

## 6. 見つかったもの — システムプロンプト自身が `<tool_call>` トークンを含む

`tools` を渡すと、テンプレートは呼び出し規約を**動く例**として
システムプロンプトに書く:

```
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
…
```

この `<tool_call>` は**追加トークンとして符号化される** — 語彙にあるので、
テキストとしてではなくマーカー ID 248058 になる。したがって:

- 描画済みのプロンプトからマーカーを探すコードは**後ろから探さなければ
  ならない。**最初の `<tool_call>` は説明文の中にある。実際に踏んだ
  (検査が `GBNF: piece 'example' … 候補スタックが残らなかった` で落ちた)
- lazy 文法のトリガはこれに引っかからない。トリガはプロンプトではなく
  **生成トークン**にしかかからないので

## 7. lazy 文法はテキスト綴りのマーカーを拒めない

負例 `markersAsText` は**非 lazy の文法**に当てている。これは都合ではなく
性質である: lazy 文法 (GEN-5) はトリガのトークンが出るまで適用されないので、
マーカーを通常トークンで綴った列は**そもそも拘束されない** —
`allows` は全部真を返し、`mayEndHere` も真になる。

**`tool_choice: auto` は「呼び出しが整形式である」ことを約束できない。
約束できるのは「始まった呼び出しは整形式である」までである。**
`required` (最初のトークンから適用) だけがこれを拒める。この 1 行も
検査に入れてある (§4「lazy の性質」)。

## 8. ついでに直したもの — CLI がマーカーの直前のバイトを落としていた

`RunQwen.swift` は `<think>` / `</think>` のとき delta を丸ごと捨てていた。
`QwenDetokenizer` は追加トークンに対して **`run.commit() + 綴り`** を返す
ので、捨てられていたのは綴りだけでなく**その前に抱えていたバイト**でもある。
`</think>` の直前のトークンで符号点が割れていると、その文字が答えから
消える。綴りだけを剥がして残りを流すようにした。

現象としては稀 (符号点がその 1 か所で割れる必要がある) だが、
§1 の decoder が同じ問題を正面から扱っているので、同じ規則にした。
CLI の出力は変わっていない (日本語の smoke は推論 192 文字 / 答え 13 文字で
[22 §5](22-PHASE5-TOKENIZER.md) と同じ)。

## 9. この文書が動かした結論と、残したもの

| 対象 | 更新 |
| --- | --- |
| [04](04-PHASES.md) 次の一手 #22 | **完了。**パーサ・GBNF ビルダ・ByteLevel の piece が入り、`--qwen-tools` が 36 本緑 |
| [04](04-PHASES.md) Phase 5 | **これで Phase 5 の箇条書きは全部片づいた。**残るのはサーバー結線 (Phase 8) と CLI の `--tools` |
| [22 §4-2](22-PHASE5-TOKENIZER.md) | **訂正 (§5-1)。**swift-jinja のキー順は不定ではなく昇順。両方決定的 |
| INV-1 (描き直し == 生成) | **文字列引数については閉じた** (§4 の往復 7 検体)。入れ子 JSON では**閉じていない** (§5-2) |

**残したもの:**

- **サーバー** (Phase 8): `ServerInference` / `ChatRequestParser` /
  `ServerGenerationPlan` は Gemma の型を通っており、Ornith の分岐は無い。
  `QwenChatGrammarBuilder` は書けたが**誰も呼んでいない**
- **CLI の `--tools`**: `RunQwen.swift` にツールの入口は無い
- **入れ子 JSON の往復** (§5-2) と `<tools>` の綴り (§5-1)
- 並列呼び出しは文法としては通るが、**実物のモデルが並列で書くかは
  見ていない** (Phase 6 以降)
