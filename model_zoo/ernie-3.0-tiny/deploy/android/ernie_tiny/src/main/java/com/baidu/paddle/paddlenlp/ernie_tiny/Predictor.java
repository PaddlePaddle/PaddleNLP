package com.baidu.paddle.paddlenlp.ernie_tiny;

public class Predictor {
    protected long mCxxContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public Predictor() {
        mInitialized = false;
    }

    public Predictor(String modelFile,
                     String paramsFile,
                     String vocabFile,
                     String slotLabelsFile,
                     String intentLabelsFile,
                     String addedTokensFile) {
        init_(modelFile, paramsFile, vocabFile, slotLabelsFile,
                intentLabelsFile, addedTokensFile, new RuntimeOption(), 16);
    }

    public Predictor(String modelFile,
                     String paramsFile,
                     String vocabFile,
                     String slotLabelsFile,
                     String intentLabelsFile,
                     String addedTokensFile,
                     RuntimeOption runtimeOption,
                     int maxLength) {
        init_(modelFile, paramsFile, vocabFile, slotLabelsFile,
                intentLabelsFile, addedTokensFile, runtimeOption, maxLength);
    }

    public boolean init(String modelFile,
                        String paramsFile,
                        String vocabFile,
                        String slotLabelsFile,
                        String intentLabelsFile,
                        String addedTokensFile,
                        RuntimeOption runtimeOption,
                        int maxLength) {
        return init_(modelFile, paramsFile, vocabFile, slotLabelsFile,
                intentLabelsFile, addedTokensFile, runtimeOption, maxLength);
    }

    public boolean release() {
        mInitialized = false;
        if (mCxxContext == 0) {
            return false;
        }
        return releaseNative(mCxxContext);
    }

    public boolean initialized() {
        return mInitialized;
    }

    // Fetch text information (will call predict from native)
    public IntentDetAndSlotFillResult[] predict(String[] texts) {
        if (mCxxContext == 0) {
            return null;
        }
        return predictNative(mCxxContext, texts);
    }

    private boolean init_(String modelFile,
                          String paramsFile,
                          String vocabFile,
                          String slotLabelsFile,
                          String intentLabelsFile,
                          String addedTokensFile,
                          RuntimeOption runtimeOption,
                          int maxLength) {
        if (!mInitialized) {
            mCxxContext = bindNative(
                    modelFile,
                    paramsFile,
                    vocabFile,
                    slotLabelsFile,
                    intentLabelsFile,
                    addedTokensFile,
                    runtimeOption,
                    maxLength
            );
            if (mCxxContext != 0) {
                mInitialized = true;
            }
            return mInitialized;
        } else {
            // release current native context and bind a new one.
            if (release()) {
                mCxxContext = bindNative(
                        modelFile,
                        paramsFile,
                        vocabFile,
                        slotLabelsFile,
                        intentLabelsFile,
                        addedTokensFile,
                        runtimeOption,
                        maxLength
                );
                if (mCxxContext != 0) {
                    mInitialized = true;
                }
                return mInitialized;
            }
            return false;
        }
    }

    // Bind predictor from native context.
    private native long bindNative(String modelFile,
                                   String paramsFile,
                                   String vocabFile,
                                   String slotLabelsFile,
                                   String intentLabelsFile,
                                   String addedTokensFile,
                                   RuntimeOption runtimeOption,
                                   int maxLength);

    // Call prediction from native context.
    private native IntentDetAndSlotFillResult[] predictNative(long CxxContext,
                                                              String[] texts);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long CxxContext);

    // Initializes at the beginning.
    static {
        Initializer.init();
    }
}
