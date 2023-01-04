package com.baidu.paddle.paddlenlp.ernie_tiny;

import android.support.annotation.NonNull;

import java.util.HashMap;
import java.util.Map;

public class UIEResult {
    public long mStart;
    public long mEnd;
    public double mProbability;
    public String mText;
    public HashMap<String, UIEResult[]> mRelation;
    public boolean mInitialized = false;

    public UIEResult() {
        mInitialized = false;
    }

    public static String printResult(@NonNull UIEResult result, int tabSize) {
        final int TAB_OFFSET = 4;
        StringBuilder os = new StringBuilder();
        StringBuilder tabStr = new StringBuilder();
        for (int i = 0; i < tabSize; ++i) {
            tabStr.append(" ");
        }
        os.append(tabStr).append("text: ").append(result.mText).append("\n");
        os.append(tabStr).append("probability: ").append(result.mProbability).append("\n");
        if (result.mStart != 0 || result.mEnd != 0) {
            os.append(tabStr).append("start: ").append(result.mStart).append("\n");
            os.append(tabStr).append("end: ").append(result.mEnd).append("\n");
        }
        if (result.mRelation == null) {
            os.append("\n");
            return os.toString();
        }
        if (result.mRelation.size() > 0) {
            os.append(tabStr).append("relation:\n");
            for (Map.Entry<String, UIEResult[]> currRelation : result.mRelation.entrySet()) {
                os.append(" ").append(currRelation.getKey()).append(":\n");
                for (UIEResult uieResult : currRelation.getValue()) {
                    os.append(printResult(uieResult, tabSize +  2 * TAB_OFFSET));
                }
            }
        }
        os.append("\n");
        return os.toString();
    }

    public static String printResult(@NonNull HashMap<String, UIEResult[]>[] results) {
        StringBuilder os = new StringBuilder();
        os.append("The result:\n");
        for (HashMap<String, UIEResult[]> result : results) {
            for (Map.Entry<String, UIEResult[]> currResult : result.entrySet()) {
                os.append(currResult.getKey()).append(": \n");
                for (UIEResult uie_result : currResult.getValue()) {
                    os.append(printResult(uie_result, 4));
                }
            }
            os.append("\n");
        }
        return os.toString();
    }
}
