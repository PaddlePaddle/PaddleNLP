package com.baidu.paddle.paddlenlp.app.ernie_tiny;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;

import com.baidu.paddle.paddlenlp.app.R;
import com.baidu.paddle.paddlenlp.ernie_tiny.RuntimeOption;
import com.baidu.paddle.paddlenlp.ernie_tiny.Predictor;
import com.baidu.paddle.paddlenlp.ernie_tiny.IntentDetAndSlotFillResult;
import com.baidu.paddle.paddlenlp.ui.Utils;


public class ERNIETinyMainActivity extends Activity implements View.OnClickListener {
    private static final String TAG = ERNIETinyMainActivity.class.getSimpleName();
    private ImageView back;
    private ImageButton btnSettings;
    private EditText etERNIETinyInput;
    private EditText etERNIETinyOutput;
    private Button btnERNIETinyAnalysis;
    private String[] inputTexts;

    // Call 'init' and 'release' manually later
    Predictor predictor = new Predictor();

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.ernie_tiny_activity_main);

        // Clear all setting items to avoid app crashing due to the incorrect settings
        initSettings();

        // Check and request WRITE_EXTERNAL_STORAGE permissions
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }

        // Init the camera preview and UI components
        initView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Reload settings and re-initialize the predictor
        checkAndUpdateSettings();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        super.onDestroy();
    }

    private void initView() {
        // Back from setting page to main page
        back = findViewById(R.id.iv_back);
        back.setOnClickListener(this);
        // Apply ERNIE Tiny predict
        btnERNIETinyAnalysis = findViewById(R.id.btn_ernie_tiny_analysis);
        btnERNIETinyAnalysis.setOnClickListener(this);
        // ERNIE Tiny input and output texts
        etERNIETinyInput = findViewById(R.id.et_ernie_tiny_input);
        etERNIETinyOutput = findViewById(R.id.et_ernie_tiny_output);
        // Setting page
        btnSettings = findViewById(R.id.btn_settings);
        btnSettings.setOnClickListener(this);
    }

    @SuppressLint("NonConstantResourceId")
    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.btn_settings:
                startActivity(new Intent(ERNIETinyMainActivity.this, ERNIETinySettingsActivity.class));
                break;
            case R.id.iv_back:
                finish();
                break;
            case R.id.btn_ernie_tiny_analysis:
                extractTextsIntentAndSlot();
                break;
            default:
                break;
        }

    }

    public void extractTextsIntentAndSlot() {
        if (updateInputTexts()) {
            IntentDetAndSlotFillResult[] results = predictor.predict(inputTexts);
            updateOutputTexts(results);
        }
    }

    public void updateOutputTexts(IntentDetAndSlotFillResult[] results) {
        if (results == null) {
            etERNIETinyOutput.setText("分析结果为空");
            return;
        }
        if (inputTexts == null) {
            etERNIETinyOutput.setText("输入文本为空");
            return;
        }
        if (inputTexts.length != results.length) {
            String info = "输入文本数量与分析结果数量不一致！"
                    + inputTexts.length + "!=" + results.length;
            etERNIETinyOutput.setText(info);
            return;
        }
        // Merge Result Str
        StringBuilder resultStrBuffer = new StringBuilder();
        for (int i = 0; i < results.length; ++i) {
            resultStrBuffer
                    .append("NO.")
                    .append(i)
                    .append(" text = ")
                    .append(inputTexts[i])
                    .append("\n")
                    .append(results[i].mStr)
                    .append("\n");
        }
        // Update output text view (EditText)
        etERNIETinyOutput.setText(resultStrBuffer.toString());
    }

    public boolean updateInputTexts() {
        String combinedInputText = etERNIETinyInput.getText().toString();
        if (combinedInputText == null || combinedInputText.length() == 0) {
            // Use default text if no custom text
            combinedInputText = getString(R.string.ERNIE_TINY_INPUT_TEXTS_DEFAULT);
        }
        String[] texts = combinedInputText.split("[。！!：；:;]");
        if (texts.length <= 0) {
            return false;
        }
        for (int i = 0; i < texts.length; ++i) {
            texts[i] = texts[i].trim();
        }
        // Update input texts
        inputTexts = texts;
        return true;
    }

    @SuppressLint("ApplySharedPref")
    public void initSettings() {
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.commit();
        ERNIETinySettingsActivity.resetSettings();
    }

    public void checkAndUpdateSettings() {
        if (ERNIETinySettingsActivity.checkAndUpdateSettings(this)) {
            // Clear output text first
            etERNIETinyOutput.setText("");

            // Update predictor
            String realModelDir = getCacheDir() + "/" + ERNIETinySettingsActivity.modelDir;
            Utils.copyDirectoryFromAssets(this, ERNIETinySettingsActivity.modelDir, realModelDir);

            String modelFile = realModelDir + "/" + "infer_model.pdmodel";
            String paramsFile = realModelDir + "/" + "infer_model.pdiparams";
            String vocabFile = realModelDir + "/" + "vocab.txt";
            String slotLabelsFile = realModelDir + "/" + "slots_label.txt";
            String intentLabelsFile = realModelDir + "/" + "intent_label.txt";
            String addedTokensFile = realModelDir + "/" + "added_tokens.json";
            RuntimeOption option = new RuntimeOption();
            option.setCpuThreadNum(ERNIETinySettingsActivity.cpuThreadNum);
            option.setLitePowerMode(ERNIETinySettingsActivity.cpuPowerMode);
            if (Boolean.parseBoolean(ERNIETinySettingsActivity.enableLiteInt8)) {
                option.enableLiteInt8(); // For quantized models
            } else {
                // Enable FP16 if Int8 option is not ON.
                if (Boolean.parseBoolean(ERNIETinySettingsActivity.enableLiteFp16)) {
                    option.enableLiteFp16();
                }
            }
            predictor.init(modelFile, paramsFile, vocabFile, slotLabelsFile,
                    intentLabelsFile, addedTokensFile, option, 16);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(ERNIETinyMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            ERNIETinyMainActivity.this.finish();
                        }
                    }).show();
        }
    }

    private void requestAllPermissions() {
        ActivityCompat.requestPermissions(
                this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                0);
    }

    private boolean checkAllPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED;
    }

}
