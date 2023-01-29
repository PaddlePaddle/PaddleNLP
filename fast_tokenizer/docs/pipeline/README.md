# FastTokenizer Pipeline

当我们使用 Tokenizer 的`Tokenizer.encode` 或者 `Tokenizer.encode_batch` 方法进行分词时，会经历如下四个阶段：Normalize、PreTokenize、 Model 以及 PostProcess。针对这四个阶段，FastTokenizer 提供 Normalizer、PreTokenizer、Model以及PostProcessor四个组件分别完成四个阶段所需要的工作。下面将详细介绍四大组件具体负责的工作。

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

Model 组件是 FastTokenizer 核心模块，用于将粗粒度词组按照一定的算法进行切分，得到细粒度的 Token（word piece），目前支持的切词算法包括 FastWordPiece、WordPiece、BPE 以及 Unigram。

## PostProcessor

PostProcess 组件为后处理，主要执行 Transformer 类模型的文本序列的处理，比如添加 [SEP] 等特殊 Token。
