# FastTokenizer Pipeline

当我们使用 Tokenizer 的 `Tokenizer.encode` 或者 `Tokenizer.encode_batch` 方法进行分词时，会经历如下四个阶段：Normalize、PreTokenize、 Model 以及 PostProcess。针对这四个阶段，FastTokenizer 提供 Normalizer、PreTokenizer、Model 以及 PostProcessor 四个组件分别完成四个阶段所需要的工作。下面将详细介绍四大组件具体负责的工作，并通过示例介绍如何组合四个组件定义一个 Tokenizer。

## Normalizer

Normalizer 组件主要用于将原始字符串标准化，输出标准化的字符串，常见的标准化字符串操作有大小写转换、半角全角转换等。 FastTokenizer 所有 Normalizer 类都继承自 `normalizers.Normalizer`，命名方式均为 `normalizers.*Normalizer`。 FastTokenizer 还支持将现有 Normalizer 类进行组合得到一个 Normalizer 序列，用户可以通过调用 `normalizers.SequenceNormalizer` 使用已有的 Normalizer 自定义新的 Normalizer。下面将分别展示 Python 以及 C++ 上使用示例。

### Python 示例

```python
import fast_tokenizer
from fast_tokenizer.normalizers import LowercaseNormalizer, SequenceNormalizer, NFDNormalizer, StripAccentsNormalizer

normalizer = SequenceNormalizer([NFDNormalizer(), StripAccentsNormalizer() LowercaseNormalizer()])
print(normalizer.normalize_str("Héllò hôw are ü?"))
# hello how are u?
```

### C++ 示例

```c++

#include <iostream>
#include "fast_tokenizer/normalizers/normalizers.h"
using namespace paddlenlp::fast_tokenizer;

int main() {
  normalizers::NFDNormalizer n1;
  normalizers::StripAccentsNormalizer n2;
  normalizers::LowercaseNormalizer n3;
  normalizers::SequenceNormalizer normalizer({&n1, &n2, &n3});
  normalizers::NormalizedString normalized("Héllò hôw are ü?");
  normalizer(&normalized);
  // Expected output
  // normalized string: hello how are u?
  // original string: Héllò hôw are ü?
  std::cout << "normalized string: " << normalized.GetStr() << std::endl;
  std::cout << "original string: " << normalized.GetOrignalStr() << std::endl;
}

```

## PreTokenizer

PreTokenizer 组件主要使用简单的分词方法，将标准化的字符串进行预切词，得到较大粒度的词组（word），例如按照标点、空格等方式进行分词。FastTokenizer 所有 PreTokenizer 类都继承自 `normalizers.PreTokenizer`，命名方式均为 `normalizers.*PreTokenizer`。 下面将分别展示 Python 以及 C++ 上使用空格对文本进行分词的使用示例。

### Python 示例

```python
import fast_tokenizer
from fast_tokenizer.pretokenizers import WhitespacePreTokenizer
pretokenizer = WhitespacePreTokenizer()
print(pretokenizer.pretokenize_str("Hello! How are you? I'm fine, thank you."))
# [('Hello!', (0, 6)), ('How', (7, 10)), ('are', (11, 14)), ('you?', (15, 19)), ("I'm", (20, 23)), ('fine,', (24, 29)), ('thank', (30, 35)), ('you.', (36, 40))]
```

### C++ 示例

```c++

#include <iostream>
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

using namespace paddlenlp::fast_tokenizer;

int main() {
  pretokenizers::WhitespacePreTokenizer pretokenizer;
  pretokenizers::PreTokenizedString pretokenized(
      "Hello! How are you? I'm fine, thank you.");
  pretokenizer(&pretokenized);
  auto&& splits = pretokenized.GetSplits(true, core::OffsetType::CHAR);
  for (auto&& split : splits) {
    auto&& value = std::get<0>(split);
    auto&& offset = std::get<1>(split);
    std::cout << "(" << value << ", (" << offset.first << ", " << offset.second
              << ")"
              << ")" << std::endl;
  }
  return 0;
}

// (Hello!, (0, 6))
// (How, (7, 10))
// (are, (11, 14))
// (you?, (15, 19))
// (I'm, (20, 23))
// (fine,, (24, 29))
// (thank, (30, 35))
// (you., (36, 40))

```

## Model

Model 组件是 FastTokenizer 核心模块，用于将粗粒度词组按照一定的算法进行切分，得到细粒度的 Token（word piece）及其对应的在词表中的 id，目前支持的切词算法包括 FastWordPiece[1]、WordPiece、BPE 以及 Unigram。其中，`FastWordPiece` 是 "Fast WordPiece Tokenization" 提出的基于`MinMaxMatch`匹配算法的一种分词算法。原有 `WordPiece` 算法的时间复杂度与序列长度为二次方关系，在对长文本进行分词操作时，时间开销比较大。而 `FastWordPiece` 算法通过 `Aho–Corasick` 算法避免 Token 失配时从头匹配，将 `WordPiece` 算法的时间复杂度降低为与序列长度的线性关系，大大提升了分词效率。下面是 `FastWordPiece` 类的初始化示例。

### Python 示例

```python
import fast_tokenizer
from fast_tokenizer.models import FastWordPiece

# Initialize model from ernie 3.0 vocab file
model = FastWordPiece.from_file("ernie-3.0-medium-vocab.txt", with_pretokenization=True)
print(model.tokenize("我爱中国!"))
# [id: 75    value:我    offset: (0, 3), id: 329    value:爱    offset: (3, 6), id: 12    value:中    offset: (6, 9), id: 20    value:国    offset: (9, 12), id: 12046    value:!    offset: (12, 13)]
```

### C++ 示例

```c++

#include <iostream>
#include <vector>

#include "fast_tokenizer/models/models.h"

using namespace paddlenlp::fast_tokenizer;

int main() {
  std::string text = "我爱中国！";
  auto model = models::FastWordPiece::GetFastWordPieceFromFile(
      "ernie_vocab.txt", "[UNK]", 100, "##", true);
  std::vector<core::Token> results = model.Tokenize(text);
  for (const core::Token& token : results) {
    std::cout << "id: " << token.id_ << ", value: " << token.value_
              << ", offset: (" << token.offset_.first << ", "
              << token.offset_.second << ")." << std::endl;
  }
  return 0;
}

// id: 75, value: 我, offset: (0, 3).
// id: 329, value: 爱, offset: (3, 6).
// id: 12, value: 中, offset: (6, 9).
// id: 20, value: 国, offset: (9, 12).
// id: 12044, value: ！, offset: (12, 15).
```

## PostProcessor

PostProcess 组件主要执行 Transformer 类模型的文本序列的后处理逻辑，比如添加 [SEP] 等特殊 Token，并且会将前面分词得到的结果转为一个 `Encoding` 的结构体，包含 token_ids, type_ids, offset, position_ids 等模型所需要的信息。FastTokenizer 所有 PostProcessor 类都继承自 `normalizers.PostProcessor`，命名方式均为 `normalizers.*PostProcessor`。

## Tokenizer

Tokenizer 对象在运行`Tokenizer.encode` 或者 `Tokenizer.encode_batch` 方法进行分词时，通过调用各个阶段组件的回调函数运行不同阶段的处理逻辑。所以我们定义 Tokenizer 对象时，需要设置各个阶段的组件。下面将通过代码示例展示如何定义 ERNIE 模型的 Tokenizer。

### Python 示例

```python

import fast_tokenizer
from fast_tokenizer import Tokenizer
from fast_tokenizer.models import FastWordPiece
from fast_tokenizer.normalizers import BertNormalizer
from fast_tokenizer.pretokenizers import BertPreTokenizer

# 1. Initialize model from ernie 3.0 vocab file
model = FastWordPiece.from_file("ernie-3.0-medium-vocab.txt")

# 2. Use model to initialize a tokenizer object
tokenizer = Tokenizer(model)

# 3. Set a normalizer
tokenizer.normalizer = BertNormalizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)

# 4. Set a pretokenizer
tokenizer.pretokenizer = BertPreTokenizer()

print(tokenizer.encode("我爱中国!"))

# The Encoding content:
# ids: 75, 329, 12, 20, 12046
# type_ids: 0, 0, 0, 0, 0
# tokens: 我, 爱, 中, 国, !
# offsets: (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)
# special_tokens_mask: 0, 0, 0, 0, 0
# attention_mask: 1, 1, 1, 1, 1
# sequence_ranges:
```

针对 ERNIE、BERT 这类常见模型，FastTokenizer Python 库 已经定义好这类模型的 Tokenizer，可以通过 `from fast_tokenizer import ErnieFastTokenizer` 直接使用。

### C++ 示例

```c++

#include <iostream>
#include <vector>

#include "fast_tokenizer/core/tokenizer.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/postprocessors/postprocessors.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"

using namespace paddlenlp::fast_tokenizer;

int main() {
  std::vector<std::string> texts{"我爱中国！"};
  core::Tokenizer tokenizer;

  // 1. Set model
  auto model = models::FastWordPiece::GetFastWordPieceFromFile(
      "ernie_vocab.txt", "[UNK]", 100, "##", true);
  tokenizer.SetModel(model);

  // 2. Set Normalizer
  normalizers::BertNormalizer normalizer(
      /* clean_text = */ true,
      /* handle_chinese_chars = */ true,
      /* strip_accents= */ true,
      /* lowercase = */ true);
  tokenizer.SetNormalizer(normalizer);

  // 3. Set Pretokenizer
  pretokenizers::BertPreTokenizer pretokenizer;
  tokenizer.SetPreTokenizer(pretokenizer);

  // 4. Set PostProcessor
  postprocessors::BertPostProcessor postprocessor;
  tokenizer.SetPostProcessor(postprocessor);

  std::vector<core::Encoding> encodings;
  tokenizer.EncodeBatchStrings(texts, &encodings);

  for (auto encoding : encodings) {
    std::cout << encoding.DebugString() << std::endl;
  }
  return 0;
}

// The Encoding content:
// ids: 101, 75, 329, 12, 20, 12044, 102
// type_ids: 0, 0, 0, 0, 0, 0, 0
// tokens: [CLS], 我, 爱, 中, 国, ！, [SEP]
// offsets: (0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 0)
// special_tokens_mask: 1, 0, 0, 0, 0, 0, 1
// attention_mask: 1, 1, 1, 1, 1, 1, 1
// sequence_ranges: {0 : (1, 6) },
```

针对 ERNIE、BERT 这类常见模型，FastTokenizer C++ 库 已经定义好这类模型的 Tokenizer，可以通过 `paddlenlp::fast_tokenizer::tokenizers_impl::ErnieFastTokenizer` 直接使用。


## 参考文献

- [1] Xinying Song, Alex Salcianuet al. "Fast WordPiece Tokenization", EMNLP, 2021
