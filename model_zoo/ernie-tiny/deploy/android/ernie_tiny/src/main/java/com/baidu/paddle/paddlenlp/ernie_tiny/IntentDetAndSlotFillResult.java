package com.baidu.paddle.paddlenlp.ernie_tiny;

public class IntentDetAndSlotFillResult {
    public String mStr;
    public boolean mInitialized = false;
    public IntentDetResult mIntentResult;
    public SlotFillResult[] mSlotResult;

    public IntentDetAndSlotFillResult() {
        mInitialized = false;
    }

    static class IntentDetResult {
        public IntentDetResult() {}
        public String mIntentLabel;
        public float mIntentConfidence;
    }

    static class SlotFillResult {
        public SlotFillResult() {}
        public String mSlotLabel;
        public String mEntity;
        public int[] mPos; // [2]
    }
}
