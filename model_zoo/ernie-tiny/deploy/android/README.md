# FastDeploy ERNIE 3.0 Tiny 模型Android部署示例

本目录提供了快速完成在 Android 的车载语音场景下的口语理解（Spoken Language Understanding，SLU）任务的部署示例。

## 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

## App示例编译和使用步骤

1. ERNIE 3.0 Tiny Android 部署示例位于`PaddleNLP/model_zoo/ernie-tiny/deploy/android`目录
2. 用 Android Studio 打开 ernie-tiny/deploy/android 工程
3. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/210742200-596f1585-9d4c-46e4-acae-5956a798c7ce.png">


工程内容说明：  
```bash
.
├── README.md
├── app                   # App示例代码
├── build.gradle
├── ernie_tiny            # ERNIE Tiny JNI & Java封装代码
# ...                     # 一些和gradle相关的工程配置文件
├── local.properties
└── ui                    # 一些辅助用的UI代码
```

> **注意：**
>> 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod NDK location` 为您本机配置的 NDK 所在路径。本工程默认使用的NDK版本为20.
>> 如果您是通过 Android Studio 的 SDK Tools 下载的 NDK (见本章节"环境准备")，可以直接点击下拉框选择默认路径。
>> 还有一种 NDK 配置方法，你可以在 `local.properties` 文件中手动完成 NDK 路径配置，如下图所示
>> 如果以上步骤仍旧无法解决 NDK 配置错误，请尝试根据 Android Studio 官方文档中的[更新 Android Gradle 插件](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin)章节，尝试更新Android Gradle plugin版本。


4. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载预编译的 FastDeploy Android 库 以及 模型文件，需要联网)
   成功后效果如下，图一：APP 安装到手机；图二： APP 打开后的效果，输入文本后点击"开始分析意图"后即会自动进行意图识别和槽位分析；图三：APP 设置选项，点击右上角的设置图片，可以设置不同选项进行体验。

| APP 效果 | APP 演示 | APP设置项 |
  | ---     | --- | --- |
| <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/210737269-0fe175c9-7b87-40b3-8249-1c6378e4a5e9.jpg">  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/211800602-2790c799-0823-4f91-8ce3-7b4a45896b41.gif"> | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/211800645-bc274593-26a4-4ed0-a258-c0615fcafce1.jpg"> |  

5. 点击 APP 右上角的设置选项，可以跳转到设置页。在设置页，您可以选择不同的模型和不同的推理精度，即是否开启 FP16 和 Int8 推理，两种推理精度只能二选一。FP32各模型都支持的，只要设置项中的 FP16 和 Int8 选项都为 false 时，即使用FP32进行推理。各模型FP16和Int8推理的支持情况为：  

|模型选项|模型名称|FP16|Int8|  
|---|---|---|---|  
|models/ernie-tiny|原模型|✅|-|   
|models/ernie-tiny-clip|原模型+裁剪（词表+模型宽度）|✅|-|   
|models/ernie-tiny-clip-qat|原模型+裁剪（词表+模型宽度）+量化（矩阵乘）|-|✅|   
|models/ernie-tiny-clip-qat-embedding-int8|原模型+裁剪（词表+模型宽度）+量化（矩阵乘+Embedding）|-|✅|   


## ERNIE Tiny Java SDK 说明和使用

本工程除了可以直接编译 App 体验之外，还可以编译 ERNIE 3.0 Tiny 的`Java SDK`，方便用户开箱即用。 如下图所示，编译 Java SDK 的步骤为：  
   - 先在 Android Studio 中打开`ernie_tiny/build.gradle`工程文件；  
   - 选择 Build->Make Module 'android:ernietiny'；  
   - 从`ernie_tiny/build/outputs/aar`目录中获取编译后得到的 SDK，即`ernie_tiny-debug.aar`.

<img width="1073" alt="image" src="https://user-images.githubusercontent.com/31974251/210746163-41d39478-8d5d-4138-9f76-75ff5c25c295.png">

在获取到`ernie_tiny-debug.aar`后，可将其拷贝到您自己的工程中进行使用。在 Android Studio 中的配置步骤如下：  
（1）首先，将`ernie_tiny-debug.aar`拷贝到您 Android 工程的libs目录下。
```bash
├── build.gradle
├── libs
│   └── ernie_tiny-debug.aar
├── proguard-rules.pro
└── src
```
（2）在您的 Android 工程中的 build.gradle 引入 ERNIE 3.0 Tiny SDK，如下
```groovy
dependencies {
    implementation fileTree(include: ['ernie_tiny-debug.aar'], dir: 'libs')
    implementation 'com.android.support:appcompat-v7:28.0.0'
    // ...
}
```

### ERNIE 3.0 Tiny Java API说明如下  

- ERNIE 3.0 Tiny `Predictor` 初始化 API: 模型初始化 API 包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用 init 函数，在合适的程序节点进行初始化。ERNIE 3.0 Tiny Predictor 初始化参数说明如下：
   - modelFile: String, paddle 格式的模型文件路径，如 infer_model.pdmodel
   - paramFile: String, paddle 格式的参数文件路径，如 infer_model.pdiparams
   - vocabFile: String, 词表文件，如 vocab.txt 每一行包含一个词
   - slotLabelsFile: String, 槽位标签文件，如 slots_label.txt
   - intentLabelsFile: String, 意图标签文件，如 intent_label.txt 每一行包含一个标签
   - addedTokensFile: String, 额外词表文件，如 added_tokens.json，json文件
   - runtimeOption: RuntimeOption，可选参数，模型初始化 option。如果不传入该参数则会使用默认的运行时选项。
   - maxLength: 最大序列长度，默认为16

```java
public Predictor(); // 空构造函数，之后可以调用init初始化
public Predictor(String modelFile, String paramsFile, String vocabFile, String slotLabelsFile, String intentLabelsFile, String addedTokensFile);
public Predictor(String modelFile, String paramsFile, String vocabFile, String slotLabelsFile, String intentLabelsFile, String addedTokensFile, RuntimeOption runtimeOption, int maxLength);
public boolean init(String modelFile, String paramsFile, String vocabFile, String slotLabelsFile, String intentLabelsFile, String addedTokensFile, RuntimeOption runtimeOption, int maxLength);
```  

- ERNIE 3.0 Tiny `Predictor` 预测 API：Predictor 提供 predict 接口对输出的文本进行意图识别。  
```java
public IntentDetAndSlotFillResult[] predict(String[] texts);
```

- ERNIE 3.0 Tiny Predictor 资源释放 API: 调用 release() API 可以释放模型资源，返回 true 表示释放成功，false 表示失败；调用 initialized() 可以判断 Predictor 是否初始化成功，true 表示初始化成功，false 表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查Predictor是否初始化成功
```

- RuntimeOption设置说明

```java
public class RuntimeOption {
  public void enableLiteFp16();                       // 开启fp16精度推理
  public void disableLiteFP16();                      // 关闭fp16精度推理（默认关闭）
  public void enableLiteInt8();                       // 开启int8精度推理（需要先准备好量化模型）
  public void disableLiteInt8();                      // 关闭int8精度推理（默认关闭）
  public void setCpuThreadNum(int threadNum);         // 设置线程数
  public void setLitePowerMode(LitePowerMode mode);   // 设置能耗模式
  public void setLitePowerMode(String modeStr);       // 通过字符串形式设置能耗模式
}
```

- 意图和槽位识别结果`IntentDetAndSlotFillResult`说明  

```java
public class IntentDetAndSlotFillResult {
   public String mStr;                    // 可用于debug的字符串 拼接了意图识别和槽位识别的结果
   public boolean mInitialized = false;
   public IntentDetResult mIntentResult;  // 意图识别结果
   public SlotFillResult[] mSlotResult;   // 槽位识别结果
   
   static class IntentDetResult {
      public String mIntentLabel;         // 意图识别结果文本标签
      public float mIntentConfidence;     // 意图识别结果置信度
   }
   static class SlotFillResult {
      public String mSlotLabel;           // 槽位识别结果文本标签
      public String mEntity;              // 槽位识别的实体
      public int[] mPos; // [2]           // 在原始文本对应的位置 [start,end] 
   }
}
```  

- ERNIE 3.0 Tiny `Predictor` FP32/FP16 推理示例

```java
import com.baidu.paddle.paddlenlp.ernie_tiny.RuntimeOption;
import com.baidu.paddle.paddlenlp.ernie_tiny.Predictor;
import com.baidu.paddle.paddlenlp.ernie_tiny.IntentDetAndSlotFillResult;
import android.app.Activity;

// 以下为伪代码
class TestERNIETiny extends Activity {
  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    Predictor predictor = new Predictor();

    // 设置模型文件和标签文件
    String modelFile = "ernie-tiny/infer_model.pdmodel";
    String paramsFile = "ernie-tiny/infer_model.pdiparams";
    String vocabFile = "ernie-tiny/vocab.txt";
    String slotLabelsFile = "ernie-tiny/slots_label.txt";
    String intentLabelsFile = "ernie-tiny/intent_label.txt";
    String addedTokensFile = "ernie-tiny/added_tokens.json";

    // RuntimeOption 设置 
    RuntimeOption option = new RuntimeOption();
    option.setCpuThreadNum(2);  // 设置线程数
    option.enableLiteFp16();    // 是否开启FP16精度推理

    // Predictor初始化        
    predictor.init(modelFile, paramsFile, vocabFile, slotLabelsFile, intentLabelsFile, addedTokensFile, option, 16);

    // 进行意图识别和槽位分析
    String[] inputTexts = new String[]{"来一首周华健的花心", "播放我们都一样", "到信阳市汽车配件城"};

    IntentDetAndSlotFillResult[] results = predictor.predict(inputTexts);
  }
}
```  

- ERNIE 3.0 Tiny `Predictor` Int8 量化模型推理示例

```java
import com.baidu.paddle.paddlenlp.ernie_tiny.RuntimeOption;
import com.baidu.paddle.paddlenlp.ernie_tiny.Predictor;
import com.baidu.paddle.paddlenlp.ernie_tiny.IntentDetAndSlotFillResult;
import android.app.Activity;

// 以下为伪代码
class TestERNIETiny extends Activity {
  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    Predictor predictor = new Predictor();

    // 设置模型文件和标签文件
    String modelFile = "ernie-tiny-clip-qat-embedding-int8/infer_model.pdmodel";
    String paramsFile = "ernie-tiny-clip-qat-embedding-int8/infer_model.pdiparams";
    String vocabFile = "ernie-tiny-clip-qat-embedding-int8/vocab.txt";
    String slotLabelsFile = "ernie-tiny-clip-qat-embedding-int8/slots_label.txt";
    String intentLabelsFile = "ernie-tiny-clip-qat-embedding-int8/intent_label.txt";
    String addedTokensFile = "ernie-tiny-clip-qat-embedding-int8/added_tokens.json";

    // RuntimeOption 设置 
    RuntimeOption option = new RuntimeOption();
    option.setCpuThreadNum(2);  // 设置线程数
    option.enableLiteInt8();    // 开启int8精度推理（需要先准备好量化模型）

    // Predictor初始化        
    predictor.init(modelFile, paramsFile, vocabFile, slotLabelsFile, intentLabelsFile, addedTokensFile, option, 16);

    // 进行意图识别和槽位分析
    String[] inputTexts = new String[]{"来一首周华健的花心", "播放我们都一样", "到信阳市汽车配件城"};

    IntentDetAndSlotFillResult[] results = predictor.predict(inputTexts);
  }
}
```

更详细的用法请参考 [ERNIETinyMainActivity](./app/src/main/java/com/baidu/paddle/paddlenlp/app/ernie_tiny/ERNIETinyMainActivity.java) 中的用法

## 替换 App 示例中的 ERNIE 3.0 Tiny 模型    

替换 App 示例中的模型的步骤非常简单，模型所在的位置为 `app/src/main/assets/models`。替换模型之前请确保您的模型目录中包含 vocab.txt、slots_label.txt、intent_label.txt 以及 added_token.json 等意图识别和槽位分析所需要的词表和标签文件。替换模型的步骤为：  
  - 将您的 ERNIE 3.0 Tiny 模型放在 `app/src/main/assets/models` 目录下；
  - 修改 `app/src/main/res/values/strings.xml` 中模型路径的默认值，如：

```xml
<!-- 将这个路径指修改成您的模型，如 models/ernie-tiny-clip-qat-embedding-int8 -->
<string name="ERNIE_TINY_MODEL_DIR_DEFAULT">models/ernie-tiny-clip-qat-embedding-int8</string>
```  

## 关于 ERNIE 3.0 Tiny JNI 的封装  

如果您对 ERNIE 3.0 Tiny JNI 封装的实现感兴趣，可以参考 [ernie_tiny_jni/predictor_jni.cc](./ernie_tiny/src/main/cpp/ernie_tiny_jni/predictor_jni.cc), 关于如何使用 JNI 进行模型封装并和 Java 通信，可以参考 [FastDeploy/use_cpp_sdk_on_android.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_cpp_sdk_on_android.md) 文档中的说明。

## 相关文档

[ERNIE 3.0 Tiny 模型详细介绍](../../README.md)

[ERNIE 3.0 Tiny 模型C++部署方法](../cpp/README.md)

[ERNIE 3.0 Tiny 模型Python部署方法](../python/README.md)

[FastDeploy SDK 安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md)