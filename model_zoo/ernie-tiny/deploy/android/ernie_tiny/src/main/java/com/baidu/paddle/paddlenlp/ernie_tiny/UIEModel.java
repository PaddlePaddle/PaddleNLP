package com.baidu.paddle.paddlenlp.ernie_tiny;

import java.util.HashMap;

public class UIEModel {
    protected long mCxxContext = 0; // Context from native.
    protected boolean mInitialized = false;

    public UIEModel() {
        mInitialized = false;
    }

    // Constructor with default runtime option
    public UIEModel(String modelFile,
                    String paramsFile,
                    String vocabFile,
                    String[] schema) {
        init_(modelFile, paramsFile, vocabFile, 0.5f, 128,
                schema, 64, new RuntimeOption(), SchemaLanguage.ZH);
    }

    // Constructor with custom runtime option
    public UIEModel(String modelFile,
                    String paramsFile,
                    String vocabFile,
                    float positionProb,
                    int maxLength,
                    String[] schema,
                    int batchSize,
                    RuntimeOption runtimeOption,
                    SchemaLanguage schemaLanguage) {
        init_(modelFile, paramsFile, vocabFile, positionProb, maxLength,
                schema, batchSize, runtimeOption, schemaLanguage);
    }

    // Call init manually with label file
    public boolean init(String modelFile,
                        String paramsFile,
                        String vocabFile,
                        String[] schema) {
        return init_(modelFile, paramsFile, vocabFile, 0.5f, 128,
                schema, 64, new RuntimeOption(), SchemaLanguage.ZH);
    }

    public boolean init(String modelFile,
                        String paramsFile,
                        String vocabFile,
                        float positionProb,
                        int maxLength,
                        String[] schema,
                        int batchSize,
                        RuntimeOption runtimeOption,
                        SchemaLanguage schemaLanguage) {
        return init_(modelFile, paramsFile, vocabFile, positionProb, maxLength,
                schema, batchSize, runtimeOption, schemaLanguage);
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

    // Set schema for Named Entity Recognition
    public boolean setSchema(String[] schema) {
        if (schema == null || schema.length == 0
                || mCxxContext == 0) {
            return false;
        }
        return setSchemaStringNative(mCxxContext, schema);
    }

    // Set schema for Cross task extraction
    public boolean setSchema(SchemaNode[] schema) {
        if (schema == null || schema.length == 0
                || mCxxContext == 0) {
            return false;
        }
        return setSchemaNodeNative(mCxxContext, schema);
    }

    // Fetch text information (will call predict from native)
    public HashMap<String, UIEResult[]>[] predict(String[] texts) {
        if (mCxxContext == 0) {
            return null;
        }
        return predictNative(mCxxContext, texts);
    }

    private boolean init_(String modelFile,
                          String paramsFile,
                          String vocabFile,
                          float positionProb,
                          int maxLength,
                          String[] schema,
                          int batchSize,
                          RuntimeOption runtimeOption,
                          SchemaLanguage schemaLanguage) {
        if (!mInitialized) {
            mCxxContext = bindNative(
                    modelFile,
                    paramsFile,
                    vocabFile,
                    positionProb,
                    maxLength,
                    schema,
                    batchSize,
                    runtimeOption,
                    schemaLanguage.ordinal()
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
                        positionProb,
                        maxLength,
                        schema,
                        batchSize,
                        runtimeOption,
                        schemaLanguage.ordinal()
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
                                   float positionProb,
                                   int maxLength,
                                   String[] schema,
                                   int batchSize,
                                   RuntimeOption runtimeOption,
                                   int schemaLanguage);

    // Call prediction from native context.
    private native HashMap<String, UIEResult[]>[] predictNative(long CxxContext,
                                                                String[] texts);

    // Release buffers allocated in native context.
    private native boolean releaseNative(long CxxContext);

    // Set schema from native for different tasks.
    private native boolean setSchemaStringNative(long CxxContext,
                                                 String[] schema);

    private native boolean setSchemaNodeNative(long CxxContext,
                                               SchemaNode[] schema);

    // Initializes at the beginning.
    static {
        Initializer.init();
    }
}
